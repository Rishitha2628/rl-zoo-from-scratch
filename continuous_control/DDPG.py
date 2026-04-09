import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time


# replay buffer 

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=1_000_000):
        self.capacity = capacity
        self.pos      = 0
        self.size     = 0

        # pre-allocate — faster than appending to a list
        self.obs     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1),       dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos]      = obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos]    = done
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx      = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def __len__(self):
        return self.size


# networks 

class Actor(nn.Module):
    """
    Deterministic policy: obs → action in [-1, 1].
    tanh at the output maps unbounded logits into the action range.
    """
    def __init__(self, obs_dim, act_dim, act_limit, hidden=256):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs) * self.act_limit


class Critic(nn.Module):
    """
    Q(s, a) — concatenate obs and action at the input.
    Single Q-value output; DDPG doesn't need the twin-critic trick (SAC does).
    """
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),             nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


def soft_update(target, source, tau):
    # polyak averaging — slowly pulls target toward online network
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


# DDPG 

def ddpg(env, total_steps=1_000_000, gamma=0.99, tau=0.005,
         lr_actor=1e-3, lr_critic=1e-3, batch_size=256,
         buffer_size=1_000_000, warmup_steps=10_000,
         noise_std=0.1, log_every=5_000):
    """
    Deep Deterministic Policy Gradient — off-policy actor-critic for
    continuous action spaces. Uses Gaussian exploration noise during training;
    the policy is deterministic at evaluation time.

    Target networks (updated via polyak averaging) stabilise the Q-targets
    that the critic is regressing toward — without them training diverges.
    warmup_steps fills the buffer with random transitions before learning starts.
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor        = Actor(obs_dim, act_dim, act_limit).to(device)
    critic       = Critic(obs_dim, act_dim).to(device)
    actor_target = Actor(obs_dim, act_dim, act_limit).to(device)
    critic_target = Critic(obs_dim, act_dim).to(device)

    # targets start identical to online networks
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    opt_actor  = optim.Adam(actor.parameters(),  lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    returns_history = []
    mean_returns    = []
    steps_done      = 0
    ep              = 0

    print(f"Training DDPG for {total_steps:,} steps on {device}...")

    obs, _ = env.reset()
    ep_return = 0.0

    while steps_done < total_steps:
        if steps_done < warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = actor(obs_t).cpu().numpy()[0]
                # Gaussian noise for exploration — simpler than OU, works just as well
                action = np.clip(action + np.random.normal(0, noise_std, act_dim),
                                 -act_limit, act_limit)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done      = terminated or truncated
        ep_return += reward

        # store terminated, not done — truncation shouldn't zero the bootstrap
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs        = next_obs
        steps_done += 1

        if done:
            ep += 1
            returns_history.append(ep_return)
            ep_return = 0.0
            obs, _    = env.reset()

            if ep % (log_every // 200) == 0:
                mean_r = np.mean(returns_history[-20:])
                mean_returns.append(mean_r)
                print(f"ep {ep:5d} | steps {steps_done:>9,} | avg20 return: {mean_r:9.2f}")

        if len(buffer) < warmup_steps:
            continue

        # update critic 
        obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size, device)

        with torch.no_grad():
            next_act   = actor_target(next_obs_b)
            q_target   = rew_b + gamma * critic_target(next_obs_b, next_act) * (1 - done_b)

        q_pred      = critic(obs_b, act_b)
        critic_loss = nn.MSELoss()(q_pred, q_target)

        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        # update actor 
        # maximise Q(s, μ(s)) — gradient flows through critic into actor
        actor_loss = -critic(obs_b, actor(obs_b)).mean()

        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        soft_update(actor_target,  actor,  tau)
        soft_update(critic_target, critic, tau)

    return actor, critic, returns_history, mean_returns


# evaluation 

def evaluate(actor, env, episodes=10):
    device  = next(actor.parameters()).device
    rewards = []

    actor.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            total  = 0.0
            done   = False
            while not done:
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = actor(obs_t).cpu().numpy()[0]   # deterministic — no noise
                obs, r, terminated, truncated, _ = env.step(action)
                total += r
                done   = terminated or truncated
            rewards.append(total)
    actor.train()

    return float(np.mean(rewards)), float(np.std(rewards))


# model persistence 

def save_models(actor, critic, path="ddpg_halfcheetah.pt"):
    torch.save({
        "actor_state":  actor.state_dict(),
        "critic_state": critic.state_dict(),
        "obs_dim":      actor.net[0].in_features,
        "act_dim":      actor.net[-2].out_features,
        "act_limit":    actor.act_limit,
        "hidden":       actor.net[0].out_features,
    }, path)
    print(f"Saved -> {path}")


def load_models(path="ddpg_halfcheetah.pt"):
    ckpt   = torch.load(path, map_location="cpu")
    actor  = Actor(ckpt["obs_dim"], ckpt["act_dim"], ckpt["act_limit"], ckpt["hidden"])
    critic = Critic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"])
    actor.load_state_dict(ckpt["actor_state"])
    critic.load_state_dict(ckpt["critic_state"])
    actor.eval()
    critic.eval()
    print(f"Loaded <- {path}")
    return actor, critic


# render visualisation 

def watch(actor, episodes=3):
    env_render = gym.make("HalfCheetah-v4", render_mode="human")
    device     = next(actor.parameters()).device

    actor.eval()
    with torch.no_grad():
        for ep in range(1, episodes + 1):
            obs, _ = env_render.reset()
            total  = 0.0
            done   = False
            steps  = 0
            while not done:
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = actor(obs_t).cpu().numpy()[0]
                obs, r, terminated, truncated, _ = env_render.step(action)
                total += r
                done   = terminated or truncated
                steps += 1
            print(f"episode {ep} | steps: {steps:4d} | return: {total:.2f}")

    env_render.close()
    actor.train()


# entry point 

def run_agent():
    env = gym.make("HalfCheetah-v4")
    np.random.seed(42)
    torch.manual_seed(42)

    actor, critic, returns_hist, mean_returns = ddpg(env, total_steps=1_000_000)

    mean_r, std_r = evaluate(actor, env)
    print(f"\neval: {mean_r:.2f} ± {std_r:.2f}")

    save_models(actor, critic)
    env.close()

    watch(actor, episodes=3)


if __name__ == "__main__":
    run_agent()
    pass