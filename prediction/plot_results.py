import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from monte_carlo_prediction import monte_carlo_prediction
from TD_prediction import td_prediction
from TD_lambda_prediction import td_lambda_prediction

# policy_evaluation gives us the exact V(π_random) to validate against
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dynamic_programming.policy_iteration import policy_evaluation 


# setup 

env = gym.make("Taxi-v3")
np.random.seed(42)

N_STATES  = env.observation_space.n
N_ACTIONS = env.action_space.n

random_policy = np.random.randint(0, N_ACTIONS, size=N_STATES)


# ground truth — exact V(π_random) via DP 
# this is what MC / TD(0) / TD(λ) should converge to

print("Computing ground truth V(π_random) via policy evaluation...")
V_true, _ = policy_evaluation(random_policy, env)


# run all three prediction methods 

print("\nMonte Carlo prediction...")
V_mc,  means_mc  = monte_carlo_prediction(random_policy, env, episodes=50_000)

print("\nTD(0) prediction...")
V_td,  means_td, _ = td_prediction(random_policy, env, episodes=10_000)

print("\nTD(λ=0.9) prediction...")
V_tdl, means_tdl = td_lambda_prediction(random_policy, env, episodes=10_000, lam=0.9)


# lambda sweep 
# run once per lambda value — shows how credit assignment horizon affects accuracy

print("\nLambda sweep...")
lambda_vals   = [0.0, 0.3, 0.5, 0.9, 1.0]
lambda_errors = {}   # mean |V - V_true| per lambda

for lam in lambda_vals:
    V_lam, _ = td_lambda_prediction(random_policy, env,
                                    episodes=10_000, lam=lam, log_every=999_999)
    lambda_errors[lam] = np.abs(V_lam - V_true).mean()
    print(f"  λ={lam:.1f}  mean |error|: {lambda_errors[lam]:.4f}")


# completion rate 
# random policy rarely finishes — worth reporting so V values aren't misread

def check_completion_rate(policy, env, episodes=5_000, max_steps=200):
    completed = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = policy[state]
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        if terminated:
            completed += 1
    return completed / episodes

completion = check_completion_rate(random_policy, env)
print(f"\nRandom policy completion rate: {100*completion:.1f}%")


# figure 
#
#  Four panels:
#    TL – learning curves (mean V over episodes)
#    TR – final V distribution vs DP ground truth
#    BL – per-state |V - V_true| histogram for each method
#    BR – lambda sweep: mean error vs λ

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f"Prediction — Taxi-v3 (random policy, completion rate: {100*completion:.1f}%)",
    fontsize=12, fontweight="bold"
)

# TL: learning curves 
ax = axes[0, 0]
# MC runs more episodes so x-axes differ — normalise to [0,1] for fair comparison
ax.plot(np.linspace(0, 1, len(means_mc)),  means_mc,  label="MC",       color="steelblue")
ax.plot(np.linspace(0, 1, len(means_td)),  means_td,  label="TD(0)",     color="darkorange")
ax.plot(np.linspace(0, 1, len(means_tdl)), means_tdl, label="TD(λ=0.9)", color="seagreen")
ax.axhline(V_true.mean(), color="black", linestyle="--",
           linewidth=0.8, label=f"DP truth ({V_true.mean():.3f})")
ax.set_xlabel("Training progress (normalised)")
ax.set_ylabel("Mean V")
ax.set_title("Learning curves")
ax.legend(fontsize=8)

# TR: final V distributions 
ax = axes[0, 1]
bins = np.linspace(V_true.min(), V_true.max(), 40)
ax.hist(V_true, bins=bins, alpha=0.4, color="black",      label="DP truth",  edgecolor="none")
ax.hist(V_mc,   bins=bins, alpha=0.5, color="steelblue",  label="MC",        edgecolor="none")
ax.hist(V_td,   bins=bins, alpha=0.5, color="darkorange", label="TD(0)",     edgecolor="none")
ax.hist(V_tdl,  bins=bins, alpha=0.5, color="seagreen",   label="TD(λ=0.9)", edgecolor="none")
ax.set_xlabel("V(s)")
ax.set_ylabel("State count")
ax.set_title("Final V distributions vs ground truth")
ax.legend(fontsize=8)

# BL: per-state absolute error 
ax = axes[1, 0]
err_mc  = np.abs(V_mc  - V_true)
err_td  = np.abs(V_td  - V_true)
err_tdl = np.abs(V_tdl - V_true)
ax.hist(err_mc,  bins=40, alpha=0.5, color="steelblue",  label=f"MC      (mean {err_mc.mean():.3f})",  edgecolor="none")
ax.hist(err_td,  bins=40, alpha=0.5, color="darkorange", label=f"TD(0)   (mean {err_td.mean():.3f})",  edgecolor="none")
ax.hist(err_tdl, bins=40, alpha=0.5, color="seagreen",   label=f"TD(λ)   (mean {err_tdl.mean():.3f})", edgecolor="none")
ax.set_xlabel("|V_estimated(s) − V_true(s)|")
ax.set_ylabel("State count")
ax.set_title("Per-state estimation error")
ax.legend(fontsize=8)

# BR: lambda sweep 
ax = axes[1, 1]
lams   = list(lambda_errors.keys())
errors = list(lambda_errors.values())
ax.plot(lams, errors, marker="o", color="seagreen", linewidth=1.5)
ax.set_xlabel("λ")
ax.set_ylabel("Mean |V_estimated − V_true|")
ax.set_title("Effect of λ on estimation error")
ax.set_xticks(lams)

plt.tight_layout()
plt.show()