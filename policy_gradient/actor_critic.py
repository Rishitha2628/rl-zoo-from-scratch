import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical
from collections import deque


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
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    logits = actor(obs_t)
    dist   = Categorical(logits=logits)
    action = dist.sample()
    # FIX: also return entropy for regularisation
    return action.item(), dist.log_prob(action), dist.entropy()


# actor-critic

def actor_critic(env, episodes=10000, gamma=0.99,
                 lr_actor=3e-4,   # FIX: lowered from 1e-3 — actor needs more caution
                 lr_critic=3e-3,  # FIX: lowered from 5e-3 — too high caused instability
                 entropy_coef=0.02,  # FIX: added entropy bonus to maintain exploration
                 max_steps=1_000, log_every=100):
    """
    Online actor-critic with TD(0) advantage.
    Updates happen at every step, not at episode end — this is the key difference
    from REINFORCE, which waits for the full return.

    Key fixes vs original:
      1. Entropy regularisation — prevents premature policy collapse.
      2. Lower, more conservative learning rates — AC is sensitive to lr.
      3. Reward normalisation via running stats — stabilises the critic's target.
      4. Looser gradient clipping (1.0) — 0.5 was starving early learning.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor      = ActorNetwork(obs_dim, n_actions).to(device)
    critic     = CriticNetwork(obs_dim).to(device)
    opt_actor  = optim.Adam(actor.parameters(),  lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    # FIX: running reward normaliser — stabilises the critic's regression target
    reward_running_mean = 0.0
    reward_running_var  = 1.0
    reward_count        = 1e-8

    returns_history = []
    mean_returns    = []

    for ep in range(1, episodes + 1):
        obs, _  = env.reset()
        done    = False
        steps   = 0
        total_r = 0.0

        while not done and steps < max_steps:
            action, log_prob, entropy = get_action(actor, obs, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done    = terminated or truncated
            total_r += reward

            # FIX: normalise reward using Welford running stats
            reward_count        += 1
            delta_rw             = reward - reward_running_mean
            reward_running_mean += delta_rw / reward_count
            reward_running_var  += delta_rw * (reward - reward_running_mean)
            reward_std           = max(np.sqrt(reward_running_var / reward_count), 1e-8)
            reward_norm          = reward / reward_std

            obs_t      = torch.FloatTensor(obs).unsqueeze(0).to(device)
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)

            v_curr = critic(obs_t)
            # don't bootstrap from terminal state — only truncation bootstraps
            v_next = critic(next_obs_t).detach() * (not terminated)
            delta  = reward_norm + gamma * v_next - v_curr   # TD error = advantage estimate

            # critic minimises TD error
            critic_loss = delta ** 2
            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)  # FIX: loosened from 0.5
            opt_critic.step()

            # FIX: actor loss now includes entropy bonus to sustain exploration
            # -log_prob * advantage  →  encourage better-than-expected actions
            # -entropy_coef * entropy →  penalise over-confident distributions
        
            actor_loss = -log_prob * delta.detach() - entropy_coef * entropy
            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
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
                action, _, _ = get_action(actor, obs, device)  # FIX: unpack 3 values now
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
        "actor_state":  actor.state_dict(),
        "critic_state": critic.state_dict(),
        "obs_dim":      actor.net[0].in_features,
        "n_actions":    actor.net[-1].out_features,
        "hidden":       actor.net[0].out_features,
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
    env_render = gym.make("LunarLander-v3", render_mode="human")
    device     = next(actor.parameters()).device

    actor.eval()
    with torch.no_grad():
        for ep in range(1, episodes + 1):
            obs, _ = env_render.reset()
            total  = 0.0
            done   = False
            steps  = 0

            while not done and steps < 1_000:
                action, _, _ = get_action(actor, obs, device)  # FIX: unpack 3 values
                obs, r, terminated, truncated, _ = env_render.step(action)
                total += r
                done   = terminated or truncated
                steps += 1

            print(f"episode {ep} | steps: {steps:4d} | return: {total:.2f}")

    env_render.close()
    actor.train()


# entry point

def run_agent():
    env = gym.make("LunarLander-v3")
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