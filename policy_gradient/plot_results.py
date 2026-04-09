import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

from reinforce    import reinforce, evaluate as evaluate_reinforce, get_action as reinforce_action
from actor_critic import actor_critic, evaluate as evaluate_ac, get_action as ac_action
from PPO          import ppo, evaluate as evaluate_ppo, to_tensor


# setup 

env = gym.make("LunarLander-v2")
np.random.seed(42)
torch.manual_seed(42)

SOLVE_THRESHOLD = 200   # LunarLander-v2 considered solved above this


# train all methods 

print("REINFORCE w/ baseline...")
policy_b,  returns_b,  mr_b  = reinforce(env, episodes=3_000, baseline=True)
mean_b,  std_b  = evaluate_reinforce(policy_b, env)

print("\nREINFORCE no baseline...")
policy_nb, returns_nb, mr_nb = reinforce(env, episodes=3_000, baseline=False)
mean_nb, std_nb = evaluate_reinforce(policy_nb, env)

print("\nActor-Critic...")
actor, critic, returns_ac, mr_ac = actor_critic(env, episodes=3_000)
mean_ac, std_ac = evaluate_ac(actor, env)

print("\nPPO...")
ppo_model, returns_ppo, mr_ppo = ppo(env, total_steps=3_000_000)
mean_ppo, std_ppo = evaluate_ppo(ppo_model, env)

env.close()


# collect raw episode returns for the histogram panel 

def collect_returns(action_fn, n=200):
    """Roll out n episodes using a callable action_fn(obs) -> int."""
    e       = gym.make("LunarLander-v2")
    rewards = []
    for _ in range(n):
        obs, _ = e.reset()
        total  = 0.0
        done   = False
        while not done:
            action = action_fn(obs)
            obs, r, terminated, truncated, _ = e.step(action)
            total += r
            done   = terminated or truncated
        rewards.append(total)
    e.close()
    return np.array(rewards)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nCollecting eval distributions...")
with torch.no_grad():
    dist_b   = collect_returns(lambda o: reinforce_action(policy_b,  o, device)[0])
    dist_nb  = collect_returns(lambda o: reinforce_action(policy_nb, o, device)[0])
    dist_ac  = collect_returns(lambda o: ac_action(actor,            o, device)[0])
    dist_ppo = collect_returns(lambda o: ppo_model.get_action(to_tensor(o, device))[0].item())


# figure 
#
#  Four panels (2x2):
#    TL – smoothed learning curves, all methods (normalised x-axis)
#    TR – eval bar chart vs solve threshold
#    BL – rolling return std (variance story across methods)
#    BR – eval return distributions (200-episode histogram per method)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Policy Gradient Methods — LunarLander-v2", fontsize=13, fontweight="bold")

colours = {
    "REINFORCE+b":  "steelblue",
    "REINFORCE":    "cornflowerblue",
    "Actor-Critic": "darkorange",
    "PPO":          "seagreen",
}

# TL: smoothed learning curves 
ax = axes[0, 0]
for name, mr in [("REINFORCE+b",  mr_b),
                 ("REINFORCE",    mr_nb),
                 ("Actor-Critic", mr_ac),
                 ("PPO",          mr_ppo)]:
    ax.plot(np.linspace(0, 1, len(mr)), mr,
            label=name, color=colours[name], linewidth=1.2)
ax.axhline(SOLVE_THRESHOLD, color="black", linestyle="--",
           linewidth=0.8, label=f"Solve ({SOLVE_THRESHOLD})")
ax.set_xlabel("Training progress (normalised)")
ax.set_ylabel("Mean return")
ax.set_title("Learning curves")
ax.legend(fontsize=8)

# TR: eval bar chart 
ax    = axes[0, 1]
names = ["REINFORCE+b", "REINFORCE", "Actor-Critic", "PPO"]
means = [mean_b,  mean_nb,  mean_ac,  mean_ppo]
stds  = [std_b,   std_nb,   std_ac,   std_ppo]
bars  = ax.bar(names, means, yerr=stds, capsize=4,
               color=[colours[n] for n in names], width=0.5)
ax.axhline(SOLVE_THRESHOLD, color="black", linestyle="--",
           linewidth=0.8, label=f"Solve ({SOLVE_THRESHOLD})")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
ax.set_ylabel("Eval return (100 episodes)")
ax.set_title("Final policy evaluation")
ax.tick_params(axis="x", labelsize=8)
ax.legend(fontsize=8)

# BL: rolling return std — the variance story 
# REINFORCE baseline → lower variance than without
# Actor-Critic → lower still (online TD reduces trajectory-level noise)
# PPO → lowest (large rollout buffer + multiple update epochs)
ax     = axes[1, 0]
window = 100
for name, raw in [("REINFORCE+b",  np.array(returns_b)),
                  ("REINFORCE",    np.array(returns_nb)),
                  ("Actor-Critic", np.array(returns_ac)),
                  ("PPO",          np.array(returns_ppo))]:
    if len(raw) > window:
        xs  = np.arange(window, len(raw))
        std = [raw[max(0, i - window):i].std() for i in xs]
        ax.plot(np.linspace(0, 1, len(std)), std,
                label=name, color=colours[name], linewidth=1.0)
ax.set_xlabel("Training progress (normalised)")
ax.set_ylabel("Return std (rolling 100 ep)")
ax.set_title("Return variance over training")
ax.legend(fontsize=8)

# BR: eval return distributions 
ax   = axes[1, 1]
bins = np.linspace(-400, 350, 50)
for name, dist in [("REINFORCE+b",  dist_b),
                   ("REINFORCE",    dist_nb),
                   ("Actor-Critic", dist_ac),
                   ("PPO",          dist_ppo)]:
    ax.hist(dist, bins=bins, alpha=0.5,
            label=f"{name} (μ={dist.mean():.0f})",
            color=colours[name], edgecolor="none")
ax.axvline(SOLVE_THRESHOLD, color="black", linestyle="--",
           linewidth=0.9, label=f"Solve ({SOLVE_THRESHOLD})")
ax.set_xlabel("Episode return")
ax.set_ylabel("Count")
ax.set_title("Eval return distributions (200 episodes)")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("policy_gradient_comparison.png", dpi=150)
plt.close()
print("\nSaved policy_gradient_comparison.png")