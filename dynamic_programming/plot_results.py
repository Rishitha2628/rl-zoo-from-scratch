import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from value_iteration import value_iteration
from policy_iteration import policy_iteration


# run both algorithms 

env = gym.make("Taxi-v3")
np.random.seed(42)

print("Running value iteration...")
policy_vi, V_vi, deltas_vi = value_iteration(env)

print("\nRunning policy iteration...")
policy_pi, V_pi, deltas_pi = policy_iteration(env)


# figure 
#
#  Four panels:
#    TL – VI delta per sweep (how fast VI contracts)
#    TR – PI final-eval delta per step (how tight the last eval round gets)
#    BL – value function histogram overlay (do both reach the same V?)
#    BR – per-state |V_vi − V_pi| (where do they disagree?)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Dynamic Programming — Taxi-v3", fontsize=13, fontweight="bold")

# TL: VI convergence 
ax = axes[0, 0]
ax.semilogy(range(1, len(deltas_vi) + 1), deltas_vi,
            color="steelblue", linewidth=1.5)
ax.set_xlabel("Sweep")
ax.set_ylabel("Max |ΔV|  (log)")
ax.set_title("Value iteration — convergence")

# TR: PI final-eval convergence 
# deltas_pi is from the last evaluation round only — earlier rounds converge faster
ax = axes[0, 1]
ax.semilogy(range(1, len(deltas_pi) + 1), deltas_pi,
            color="darkorange", linewidth=1.5)
ax.set_xlabel("Eval step (final iteration)")
ax.set_ylabel("Max |ΔV|  (log)")
ax.set_title("Policy iteration — final eval convergence")

# BL: value function distributions 
ax = axes[1, 0]
ax.hist(V_vi, bins=40, alpha=0.6, color="steelblue",  label="VI",
        edgecolor="white", linewidth=0.3)
ax.hist(V_pi, bins=40, alpha=0.6, color="darkorange", label="PI",
        edgecolor="white", linewidth=0.3)
ax.set_xlabel("V(s)")
ax.set_ylabel("State count")
ax.set_title("Value function distributions")
ax.legend(fontsize=9)

# BR: per-state disagreement 
diff = np.abs(V_vi - V_pi)
ax   = axes[1, 1]
ax.hist(diff, bins=40, color="slategray",
        edgecolor="white", linewidth=0.3)
ax.set_xlabel("|V_VI(s) − V_PI(s)|")
ax.set_ylabel("State count")
ax.set_title("Per-state value disagreement")

# summary stats in a text box
policy_match = np.mean(policy_vi == policy_pi) * 100
ax.text(0.97, 0.95,
        f"Policy match: {policy_match:.1f}%\nMean |ΔV|: {diff.mean():.4f}",
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.tight_layout()
plt.show()
