import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical


# policy network 

class PolicyNetwork(nn.Module):
    """
    Maps observation → action logits.
    Two hidden layers with ReLU; softmax is applied at sample time via Categorical.
    128-128 is wide enough for LunarLander's 8-dim obs without overfitting.
    """
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def get_action(policy, obs, device):
    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
    logits = policy(obs_t)
    dist   = Categorical(logits=logits)   # handles softmax + sampling cleanly
    action = dist.sample()
    return action.item(), dist.log_prob(action)


# REINFORCE 

def reinforce(env, episodes=3_000, gamma=0.99, lr=2e-3,
              baseline=True, max_steps=1_000, log_every=100):
    """
    Monte-Carlo policy gradient for continuous-observation environments.
    gamma=0.99 instead of 0.9 — LunarLander rewards landing, which comes
    late in the episode, so high gamma is needed to propagate that signal back.

    Returns the trained policy network, full return history, and a smoothed log.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy    = PolicyNetwork(obs_dim, n_actions).to(device)
    optimiser = optim.Adam(policy.parameters(), lr=lr)

    returns_history = []
    mean_returns    = []
    baseline_val    = 0.0

    for ep in range(1, episodes + 1):
        obs, _    = env.reset()
        log_probs = []
        rewards   = []
        done      = False
        steps     = 0

        while not done and steps < max_steps:
            action, log_prob = get_action(policy, obs, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done   = terminated or truncated
            steps += 1

        # single backward pass over the episode
        G       = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns_history.append(returns[0])

        # rolling baseline — same logic as tabular REINFORCE
        if baseline:
            baseline_val = np.mean(returns_history[-100:])

        returns_t  = torch.FloatTensor(returns).to(device)
        advantages = returns_t - baseline_val if baseline else returns_t

        # normalise advantages — reduces sensitivity to reward scale
        if len(returns_t) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # policy gradient: maximise E[log π(a|s) * advantage]
        log_probs_t = torch.stack(log_probs)
        loss        = -(log_probs_t * advantages).sum()

        optimiser.zero_grad()
        loss.backward()
        # clip gradients — LunarLander returns can be large early in training
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimiser.step()

        if ep % log_every == 0:
            mean_r = np.mean(returns_history[-100:])
            mean_returns.append(mean_r)
            print(f"ep {ep:5d} | mean return (last 100): {mean_r:8.2f}")

    return policy, returns_history, mean_returns


# evaluation 

def evaluate(policy, env, episodes=100):
    device  = next(policy.parameters()).device
    rewards = []

    policy.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            total  = 0.0
            done   = False
            steps  = 0
            while not done and steps < 1_000:
                action, _ = get_action(policy, obs, device)
                obs, r, terminated, truncated, _ = env.step(action)
                total += r
                done   = terminated or truncated
                steps += 1
            rewards.append(total)
    policy.train()

    return float(np.mean(rewards)), float(np.std(rewards))


# entry point 

def run_agent():
    env = gym.make("LunarLander-v3")
    np.random.seed(42)
    torch.manual_seed(42)

    results = {}
    best_mean, best_net = -np.inf, None

    for use_baseline, key in [(True, "baseline"), (False, "no_baseline")]:
        print(f"\n{'─'*40}")
        print(f"REINFORCE {'w/' if use_baseline else 'no'} baseline")
        print(f"{'─'*40}")
        net, returns_hist, mean_returns = reinforce(
            env, episodes=3_000, baseline=use_baseline
        )
        mean_r, std_r = evaluate(net, env)
        results[key] = (net, returns_hist, mean_returns, mean_r, std_r)
        print(f"eval: {mean_r:.2f} ± {std_r:.2f}")

        if mean_r > best_mean:
            best_mean, best_net = mean_r, net

    # only save the better of the two runs
    save_policy(best_net)

    env.close()
    return results



# model persistence 

def save_policy(policy, path="reinforce_lunarlander.pt"):
    # save weights and network shape so we can rebuild without keeping the class in scope
    torch.save({
        "state_dict": policy.state_dict(),
        "obs_dim":    policy.net[0].in_features,
        "n_actions":  policy.net[-1].out_features,
        "hidden":     policy.net[0].out_features,
    }, path)
    print(f"Saved policy -> {path}")


def load_policy(path="reinforce_lunarlander.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    policy = PolicyNetwork(
        obs_dim   = checkpoint["obs_dim"],
        n_actions = checkpoint["n_actions"],
        hidden    = checkpoint["hidden"],
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    print(f"Loaded policy <- {path}")
    return policy


# render visualisation 

def watch(policy, episodes=3):
    # separate env with render_mode so training env stays unaffected
    env_render = gym.make("LunarLander-v3", render_mode="human")
    device     = next(policy.parameters()).device

    policy.eval()
    with torch.no_grad():
        for ep in range(1, episodes + 1):
            obs, _  = env_render.reset()
            total   = 0.0
            done    = False
            steps   = 0

            while not done and steps < 1_000:
                action, _ = get_action(policy, obs, device)
                obs, r, terminated, truncated, _ = env_render.step(action)
                total += r
                done   = terminated or truncated
                steps += 1

            print(f"episode {ep} | steps: {steps:4d} | return: {total:.2f}")

    env_render.close()
    policy.train()


if __name__ == "__main__":
    run_agent()
    watch()
    pass