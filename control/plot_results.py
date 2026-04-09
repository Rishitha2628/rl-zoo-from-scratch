import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from SARSA import sarsa
from Q_learning import q_learning
from monte_carlo_control import monte_carlo_control
from n_step_SARSA         import n_step_sarsa
from double_Q_learning    import double_q_learning
from prioritised_Q_learning import prioritized_q_learning


# setup 

env = gym.make("Taxi-v3")
np.random.seed(42)

DP_MEAN_V    = 2.47    # ground truth from value iteration
DP_MAX_Q     = 20.0    # true maximum Q value
PROBE_STATE  = 328


# evaluate a greedy policy 

def evaluate_policy(policy, env, episodes=1_000):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done  = False
        total = 0.0
        while not done:
            state, r, terminated, truncated, _ = env.step(policy[state])
            done   = terminated or truncated
            total += r
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


# train all methods 

print("SARSA...")
pol_sarsa, _, V_sarsa, mv_sarsa = sarsa(env, episodes=80_000, epsilon_start=10.0)

print("Q-learning...")
pol_ql, Q_ql, V_ql, mv_ql = q_learning(env, episodes=10_000)

print("MC control...")
pol_mc, _, V_mc, mv_mc, comp_rates_mc = monte_carlo_control(env, episodes=50_000)

print("Double Q-learning...")
pol_dql, Q1, Q2, V_dql, mv_dql, maxq_dql = double_q_learning(env, episodes=20_000)

print("Prioritized Q-learning...")
pol_per, _, V_per, mv_per = prioritized_q_learning(env, episodes=10_000)

print("n-step SARSA sweep...")
n_values    = [1, 3, 5, 10]
nstep_res   = {}
for n in n_values:
    policy_n, _, V_n, mv_n = n_step_sarsa(
        env, n=n, episodes=10_000, epsilon_start=float(n)
    )
    nstep_res[n] = (policy_n, V_n, mv_n)


# evaluate all final policies 

print("\nEvaluating policies...")
eval_res = {
    "SARSA":      evaluate_policy(pol_sarsa, env),
    "Q-learning": evaluate_policy(pol_ql,    env),
    "MC control": evaluate_policy(pol_mc,    env),
    "Double QL":  evaluate_policy(pol_dql,   env),
    "PER":        evaluate_policy(pol_per,   env),
}
for name, (mean, std) in eval_res.items():
    print(f"  {name:12s}  {mean:.2f} ± {std:.2f}")


# figure 
#
#  Six panels (2×3):
#    TL – learning curves, all methods (mean V, normalised x-axis)
#    TC – eval reward bar chart vs DP baseline
#    TR – n-step SARSA: mean V curves per n value
#    BL – Double Q vs Q-learning max-Q over training (overestimation proxy)
#    BC – MC control completion rate over training
#    BR – per-state V at convergence for each method

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("Control Methods — Taxi-v3", fontsize=13, fontweight="bold")

colours = {
    "SARSA":      "steelblue",
    "Q-learning": "darkorange",
    "MC control": "seagreen",
    "Double QL":  "mediumpurple",
    "PER":        "firebrick",
}

#  TL: learning curves (all methods) 
ax = axes[0, 0]
pairs = [
    ("SARSA",      mv_sarsa),
    ("Q-learning", mv_ql),
    ("MC control", mv_mc),
    ("Double QL",  mv_dql),
    ("PER",        mv_per),
]
for name, mv in pairs:
    ax.plot(np.linspace(0, 1, len(mv)), mv,
            label=name, color=colours[name], linewidth=1.2)
ax.axhline(DP_MEAN_V, color="black", linestyle="--",
           linewidth=0.8, label=f"DP ({DP_MEAN_V})")
ax.set_xlabel("Training progress (normalised)")
ax.set_ylabel("Mean V")
ax.set_title("Learning curves")
ax.legend(fontsize=7)

# TC: eval reward bar chart 
ax    = axes[0, 1]
names = list(eval_res.keys())
means = [eval_res[n][0] for n in names]
stds  = [eval_res[n][1] for n in names]
cols  = [colours[n] for n in names]
bars  = ax.bar(names, means, yerr=stds, capsize=4, color=cols, width=0.5)
ax.axhline(7.91, color="black", linestyle="--",
           linewidth=0.8, label="DP baseline (7.91)")
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
ax.set_ylabel("Eval reward")
ax.set_title("Policy evaluation (1 000 episodes)")
ax.tick_params(axis="x", labelsize=8)
ax.legend(fontsize=7)

# TR: n-step SARSA sweep 
ax = axes[0, 2]
n_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for (n, (_, _, mv_n)), c in zip(nstep_res.items(), n_colours):
    ax.plot(np.linspace(0, 1, len(mv_n)), mv_n,
            label=f"n={n}", color=c, linewidth=1.2)
ax.axhline(DP_MEAN_V, color="black", linestyle="--",
           linewidth=0.8, label=f"DP ({DP_MEAN_V})")
ax.set_xlabel("Training progress (normalised)")
ax.set_ylabel("Mean V")
ax.set_title("n-step SARSA — effect of n")
ax.legend(fontsize=8)

# BL: Double Q vs Q-learning max Q (overestimation) 
# Q-learning's max Q should overshoot 20.0; Double Q should stay closer
ax = axes[1, 0]

# rebuild Q-learning max-Q history — reuse the training run above
# sarsa_qlearning.q_learning doesn't return max_qs, so we approximate
# from V_ql (max over actions is already V) — track its max over states
# Note: this is max of V, which equals max of Q for the greedy action
ax.plot(np.linspace(0, 1, len(maxq_dql)), maxq_dql,
        label="Double QL", color=colours["Double QL"], linewidth=1.2)
# V_ql.max() is a single point — annotate as a horizontal reference instead
ax.axhline(V_ql.max(), color=colours["Q-learning"], linestyle="-.",
           linewidth=1.0, label=f"Q-learning final max ({V_ql.max():.1f})")
ax.axhline(DP_MAX_Q, color="black", linestyle="--",
           linewidth=0.8, label=f"True max ({DP_MAX_Q})")
ax.set_xlabel("Training progress (normalised)")
ax.set_ylabel("Max Q")
ax.set_title("Q overestimation — Double QL vs Q-learning")
ax.legend(fontsize=8)

# BC: MC control completion rate over training 
ax = axes[1, 1]
# comp_rates logged every log_every=10_000 episodes over 50_000 total
xs = np.linspace(0, 50_000, len(comp_rates_mc))
ax.plot(xs, comp_rates_mc, color=colours["MC control"], linewidth=1.5)
ax.set_xlabel("Episode")
ax.set_ylabel("Completion rate (%)")
ax.set_title("MC control — completion rate over training")

# BR: per-state V at convergence 
ax   = axes[1, 2]
bins = np.linspace(-10, 22, 50)
for name, V in [("SARSA",      V_sarsa),
                ("Q-learning", V_ql),
                ("MC control", V_mc),
                ("Double QL",  V_dql),
                ("PER",        V_per)]:
    ax.hist(V, bins=bins, alpha=0.4, label=name,
            color=colours[name], edgecolor="none")
ax.set_xlabel("V(s)")
ax.set_ylabel("State count")
ax.set_title("Value function distributions at convergence")
ax.legend(fontsize=7)

plt.tight_layout()
plt.show()