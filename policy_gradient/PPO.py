import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical


# shared actor-critic network 

class ActorCritic(nn.Module):
    """
    Shared backbone — actor and critic read the same feature representation.
    Orthogonal init keeps gradients well-scaled at the start of training;
    small gain on the actor head keeps the initial policy close to uniform.
    """
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        # small gain → near-uniform initial policy, large gain → V estimates start near 0
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x):
        features = self.backbone(x)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)

    def get_action(self, x):
        logits, value = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_action(self, x, action):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


def to_tensor(x, device):
    return torch.FloatTensor(x).unsqueeze(0).to(device)


# generalised advantage estimation 

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    GAE interpolates between TD(0) (lam=0, low variance, biased) and
    Monte Carlo (lam=1, high variance, unbiased). lam=0.95 from the PPO paper.
    """
    advantages = []
    gae        = 0.0
    next_value = last_value

    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        if d:
            next_value = 0.0
            gae        = 0.0
        delta      = r + gamma * next_value - v
        gae        = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v

    advantages = torch.FloatTensor(advantages)
    returns    = advantages + torch.FloatTensor(values)   # A(s,a) + V(s) ≈ G
    return advantages, returns


# PPO 

def ppo(env, total_steps=3_000_000, gamma=0.99, lam=0.95,
        lr=2e-4, clip_eps=0.2, n_steps=2048, n_epochs=10,
        batch_size=64, vf_coef=0.5, ent_coef=0.01,
        max_grad_norm=0.5, log_every=50, solve_threshold=200):
    """
    Proximal Policy Optimisation with GAE and linear LR annealing.
    Trains for total_steps environment steps, not episodes — buffer size
    (n_steps) and total_steps are the natural units for on-policy methods.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model     = ActorCritic(obs_dim, n_actions).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    returns_history = []
    mean_returns    = []
    steps_done      = 0
    ep              = 0
    solved          = False

    print(f"Training for {total_steps:,} steps on {device}...")

    while steps_done < total_steps and not solved:

        # collect one rollout buffer 
        states, actions, rewards       = [], [], []
        log_probs_old, values, dones   = [], [], []

        state, _       = env.reset()
        episode_reward = 0.0

        for _ in range(n_steps):
            state_t = to_tensor(state, device)

            with torch.no_grad():
                action, log_prob, value = model.get_action(state_t)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs_old.append(log_prob.item())
            values.append(value.item())
            dones.append(done)

            episode_reward += reward
            state           = next_state
            steps_done     += 1

            if done:
                ep += 1
                returns_history.append(episode_reward)
                episode_reward = 0.0

                if ep % log_every == 0:
                    avg = np.mean(returns_history[-log_every:])
                    mean_returns.append(avg)
                    print(f"ep {ep:5d} | steps {steps_done:>9,} | avg{log_every} return: {avg:8.2f}")

                    # check solve condition outside the collection loop
                    if avg >= solve_threshold and ep >= 100:
                        solved = True
                        print(f"\nSolved at episode {ep} (avg{log_every} = {avg:.2f})")
                        break

                state, _ = env.reset()

        # bootstrap final value if episode didn't end 
        with torch.no_grad():
            _, last_value = model(to_tensor(state, device))
            last_value    = 0.0 if dones[-1] else last_value.item()

        # GAE + normalise advantages 
        advantages, returns = compute_gae(rewards, values, dones, last_value, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t   = torch.FloatTensor(np.array(states)).to(device)
        actions_t  = torch.stack(actions).squeeze(1).to(device)
        returns_t  = returns.to(device)
        old_lps_t  = torch.FloatTensor(log_probs_old).to(device)
        adv_t      = advantages.to(device)

        # linearly decay lr — keeps updates small as policy matures
        frac = 1.0 - steps_done / total_steps
        for pg in optimiser.param_groups:
            pg["lr"] = lr * frac

        #  PPO update epochs 
        n_collected = len(states)
        for _ in range(n_epochs):
            for idx in torch.randperm(n_collected).split(batch_size):
                if len(idx) < 2:
                    continue

                new_lps, entropy, values_new = model.evaluate_action(
                    states_t[idx], actions_t[idx]
                )

                ratio = torch.exp(new_lps - old_lps_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[idx]

                # clip keeps the update inside a trust region
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values_new, returns_t[idx])
                # entropy bonus discourages premature convergence to a deterministic policy
                entropy_loss = entropy.mean()

                loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy_loss

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimiser.step()

    return model, returns_history, mean_returns


# evaluation 

def evaluate(model, env, episodes=100):
    device  = next(model.parameters()).device
    rewards = []

    model.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            total  = 0.0
            done   = False
            while not done:
                action, _, _ = model.get_action(to_tensor(obs, device))
                obs, r, terminated, truncated, _ = env.step(action.item())
                total += r
                done   = terminated or truncated
            rewards.append(total)
    model.train()

    return float(np.mean(rewards)), float(np.std(rewards))


# model persistence 

def save_model(model, path="ppo_lunarlander.pt"):
    torch.save({
        "state_dict": model.state_dict(),
        "obs_dim":    model.backbone[0].in_features,
        "n_actions":  model.actor_head.out_features,
        "hidden":     model.backbone[0].out_features,
    }, path)
    print(f"Saved -> {path}")


def load_model(path="ppo_lunarlander.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    model = ActorCritic(
        obs_dim   = checkpoint["obs_dim"],
        n_actions = checkpoint["n_actions"],
        hidden    = checkpoint["hidden"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Loaded <- {path}")
    return model


# render visualisation 

def watch(model, episodes=50):
    env_render = gym.make("LunarLander-v3", render_mode="human")
    device     = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for ep in range(1, episodes + 1):
            obs, _ = env_render.reset()
            total  = 0.0
            done   = False
            steps  = 0
            while not done:
                action, _, _ = model.get_action(to_tensor(obs, device))
                obs, r, terminated, truncated, _ = env_render.step(action.item())
                total += r
                done   = terminated or truncated
                steps += 1
            print(f"episode {ep} | steps: {steps:4d} | return: {total:.2f}")

    env_render.close()
    model.train()


# entry point 

def run_agent():
    env = gym.make("LunarLander-v3")
    np.random.seed(42)
    torch.manual_seed(42)

    model, returns_hist, mean_returns = ppo(env, total_steps=3_000_000)

    mean_r, std_r = evaluate(model, env)
    print(f"\neval: {mean_r:.2f} ± {std_r:.2f}")
    print(f"solved: {'yes' if mean_r >= 200 else 'not yet'}")

    save_model(model)
    env.close()

    watch(model, episodes=50                                                                                                                                          )


if __name__ == "__main__":
    run_agent()
    pass