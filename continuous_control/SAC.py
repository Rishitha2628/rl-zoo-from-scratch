import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym


# replay buffer 

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=1_000_000):
        self.capacity  = capacity
        self.pos       = 0
        self.size      = 0
        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards   = np.zeros((capacity, 1),       dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones     = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos]      = obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos]    = done
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, batch_size)
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

LOG_STD_MIN = -5
LOG_STD_MAX =  2


class Actor(nn.Module):
    """
    Stochastic policy: obs → (μ, log_σ) → action via reparameterisation trick.
    tanh squashes the Gaussian sample into [-act_limit, act_limit].
    The log_prob correction accounts for the tanh transform (change of variables).
    """
    def __init__(self, obs_dim, act_dim, act_limit, hidden=256):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        features = self.net(obs)
        mu       = self.mu_head(features)
        log_std  = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        # reparameterisation: action = tanh(μ + ε·σ), ε ~ N(0,1)
        # allows gradients to flow through the sampling step
        eps    = torch.randn_like(mu)
        raw    = mu + eps * std
        action = torch.tanh(raw) * self.act_limit

        # log_prob correction for tanh squashing
        log_prob = (
            torch.distributions.Normal(mu, std).log_prob(raw)
            - torch.log(1 - (action / self.act_limit) ** 2 + 1e-6)
        ).sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, obs):
        # tanh(μ) at eval time — no sampling, no noise
        mu, _ = self(obs)
        return torch.tanh(mu) * self.act_limit


class Critic(nn.Module):
    """
    Twin Q-networks in one module — take the minimum at update time to
    suppress overestimation bias (same problem Double Q-learning fixes, at scale).
    """
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),             nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),             nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs, action):
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


# SAC 

def sac(env, total_steps=1_000_000, gamma=0.99, tau=0.005,
        lr=3e-4, batch_size=256, buffer_size=1_000_000,
        warmup_steps=10_000, alpha=0.2, autotune_alpha=True,
        log_every=5_000):
    """
    Soft Actor-Critic — off-policy, maximum entropy RL.
    The entropy term α·H(π) in the objective encourages exploration throughout
    training, not just at the start. This makes SAC more robust than DDPG to
    hyperparameter choices and avoids premature convergence.

    autotune_alpha adjusts α automatically to hit a target entropy of -act_dim,
    which removes the most sensitive hyperparameter from manual tuning.
    Twin critics suppress Q overestimation (same fix as Double Q-learning).
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim   = env.observation_space.shape[0]
    act_dim   = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor         = Actor(obs_dim, act_dim, act_limit).to(device)
    critic        = Critic(obs_dim, act_dim).to(device)
    critic_target = Critic(obs_dim, act_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    opt_actor  = optim.Adam(actor.parameters(),  lr=lr)
    opt_critic = optim.Adam(critic.parameters(), lr=lr)

    # automatic entropy tuning — learns log_alpha so we don't have to set alpha manually
    if autotune_alpha:
        target_entropy = -float(act_dim)   # heuristic from the SAC paper
        log_alpha      = torch.zeros(1, requires_grad=True, device=device)
        opt_alpha      = optim.Adam([log_alpha], lr=lr)
        alpha          = log_alpha.exp().item()

    buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    returns_history = []
    mean_returns    = []
    steps_done      = 0
    ep              = 0

    print(f"Training SAC for {total_steps:,} steps on {device}...")

    obs, _    = env.reset()
    ep_return = 0.0

    while steps_done < total_steps:
        if steps_done < warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _ = actor.sample(obs_t)
                action    = action.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done      = terminated or truncated
        ep_return += reward

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
                print(f"ep {ep:5d} | steps {steps_done:>9,} | avg20 return: {mean_r:9.2f} | α: {alpha:.4f}")

        if len(buffer) < warmup_steps:
            continue

        obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size, device)

        # update critic 
        with torch.no_grad():
            next_act, next_log_prob = actor.sample(next_obs_b)
            # entropy bonus in the target — the max-entropy objective
            q_next  = critic_target.q_min(next_obs_b, next_act)
            q_target = rew_b + gamma * (1 - done_b) * (q_next - alpha * next_log_prob)

        q1, q2      = critic(obs_b, act_b)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        # update actor 
        new_act, log_prob = actor.sample(obs_b)
        # maximise Q - α·log π — the entropy-regularised objective
        actor_loss = (alpha * log_prob - critic.q_min(obs_b, new_act)).mean()

        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        # update alpha 
        if autotune_alpha:
            alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
            opt_alpha.zero_grad()
            alpha_loss.backward()
            opt_alpha.step()
            alpha = log_alpha.exp().item()

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
                action = actor.deterministic_action(obs_t).cpu().numpy()[0]
                obs, r, terminated, truncated, _ = env.step(action)
                total += r
                done   = terminated or truncated
            rewards.append(total)
    actor.train()

    return float(np.mean(rewards)), float(np.std(rewards))


# model persistence 

def save_models(actor, critic, path="sac_halfcheetah.pt"):
    torch.save({
        "actor_state":  actor.state_dict(),
        "critic_state": critic.state_dict(),
        "obs_dim":      actor.net[0].in_features,
        "act_dim":      actor.mu_head.out_features,
        "act_limit":    actor.act_limit,
        "hidden":       actor.net[0].out_features,
    }, path)
    print(f"Saved -> {path}")


def load_models(path="sac_halfcheetah.pt"):
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
                action = actor.deterministic_action(obs_t).cpu().numpy()[0]
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

    actor, critic, returns_hist, mean_returns = sac(env, total_steps=1_000_000)

    mean_r, std_r = evaluate(actor, env)
    print(f"\neval: {mean_r:.2f} ± {std_r:.2f}")

    save_models(actor, critic)
    env.close()

    watch(actor, episodes=3)


if __name__ == "__main__":
    run_agent()
    pass