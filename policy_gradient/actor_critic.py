import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical


# networks 

class ActorNetwork(nn.Module):
    """
    Maps observation → action logits.
    Separate from the critic so actor and critic can use different learning rates —
    critic needs to track a moving target, actor updates should be more cautious.
    """
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    """
    Maps observation → V(s).
    Single scalar output — no softmax, just a value estimate.
    """
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_action(actor, obs, device):
    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
    dist   = Categorical(logits=actor(obs_t))
    action = dist.sample()
    return action.item(), dist.log_prob(action)


# actor-critic 

def actor_critic(env, episodes=3_000, gamma=0.99,
                 lr_actor=1e-3, lr_critic=5e-3,
                 max_steps=1_000, log_every=100):
    """
    Online actor-critic with TD(0) advantage.
    Updates happen at every step, not at episode end — this is the key difference
    from REINFORCE, which waits for the full return.

    Separate learning rates: critic needs to move faster to track a moving value
    target; a slower actor prevents policy collapse while critic is still warm.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor     = ActorNetwork(obs_dim, n_actions).to(device)
    critic    = CriticNetwork(obs_dim).to(device)
    opt_actor = optim.Adam(actor.parameters(),  lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    returns_history = []
    mean_returns    = []

    for ep in range(1, episodes + 1):
        obs, _  = env.reset()
        done    = False
        steps   = 0
        total_r = 0.0

        while not done and steps < max_steps:
            action, log_prob = get_action(actor, obs, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done    = terminated or truncated
            total_r += reward

            obs_t      = torch.FloatTensor(obs).unsqueeze(0).to(device)
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)

            v_curr = critic(obs_t)
            # don't bootstrap from terminal state — only truncation bootstraps
            v_next = critic(next_obs_t).detach() * (not terminated)
            delta  = reward + gamma * v_next - v_curr   # TD error = advantage estimate

            # critic minimises TD error
            critic_loss = delta ** 2
            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            opt_critic.step()

            # actor maximises expected return, weighted by advantage
            actor_loss = -log_prob * delta.detach()     # detach so actor doesn't move critic
            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            opt_actor.step()

            obs    = next_obs
            steps += 1

        returns_history.append(total_r)

        if ep % log_every == 0:
            mean_r = np.mean(returns_history[-100:])
            mean_returns.append(mean_r)
            print(f"ep {ep:5d} | mean return (last 100): {mean_r:8.2f}")

    return actor, critic, returns_history, mean_returns


# evaluation 

def evaluate(actor, env, episodes=100):
    device  = next(actor.parameters()).device
    rewards = []

    actor.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            total  = 0.0
            done   = False
            steps  = 0
            while not done and steps < 1_000:
                action, _ = get_action(actor, obs, device)
                obs, r, terminated, truncated, _ = env.step(action)
                total += r
                done   = terminated or truncated
                steps += 1
            rewards.append(total)
    actor.train()

    return float(np.mean(rewards)), float(np.std(rewards))


# model persistence 

def save_models(actor, critic, path="actor_critic_lunarlander.pt"):
    torch.save({
        "actor_state":    actor.state_dict(),
        "critic_state":   critic.state_dict(),
        "obs_dim":        actor.net[0].in_features,
        "n_actions":      actor.net[-1].out_features,
        "hidden":         actor.net[0].out_features,
    }, path)
    print(f"Saved -> {path}")


def load_models(path="actor_critic_lunarlander.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    actor  = ActorNetwork(checkpoint["obs_dim"], checkpoint["n_actions"], checkpoint["hidden"])
    critic = CriticNetwork(checkpoint["obs_dim"], checkpoint["hidden"])
    actor.load_state_dict(checkpoint["actor_state"])
    critic.load_state_dict(checkpoint["critic_state"])
    actor.eval()
    critic.eval()
    print(f"Loaded <- {path}")
    return actor, critic


# render visualisation 

def watch(actor, episodes=3):
    env_render = gym.make("LunarLander-v2", render_mode="human")
    device     = next(actor.parameters()).device

    actor.eval()
    with torch.no_grad():
        for ep in range(1, episodes + 1):
            obs, _ = env_render.reset()
            total  = 0.0
            done   = False
            steps  = 0

            while not done and steps < 1_000:
                action, _ = get_action(actor, obs, device)
                obs, r, terminated, truncated, _ = env_render.step(action)
                total += r
                done   = terminated or truncated
                steps += 1

            print(f"episode {ep} | steps: {steps:4d} | return: {total:.2f}")

    env_render.close()
    actor.train()


# entry point 

def run_agent():
    env = gym.make("LunarLander-v2")
    np.random.seed(42)
    torch.manual_seed(42)

    actor, critic, returns_hist, mean_returns = actor_critic(env, episodes=3_000)
    mean_r, std_r = evaluate(actor, env)
    print(f"\neval: {mean_r:.2f} ± {std_r:.2f}")

    save_models(actor, critic)
    env.close()

    watch(actor, episodes=3)


if __name__ == "__main__":
    run_agent()
    pass