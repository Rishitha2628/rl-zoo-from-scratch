# rl-zoo-from-scratch

A clean, readable implementation of core reinforcement learning algorithms in Python using [Gymnasium](https://gymnasium.farama.org/). Built as a learning reference — every file is self-contained, every design choice is commented with *why*, not just *what*.

---

## Structure

```
rl-zoo-from-scratch/
├── assets/
│   ├── prediction_results.png
│   ├── policy_gradient_results.png
│   ├── lunarlander_ppo.mov
│   └── sac_halfcheetah.mov
├── dynamic_programming/
│   ├── dp.py                   # Value iteration, Policy iteration
│   └── plot_results.py
├── prediction/
│   ├── mc_td_prediction.py     # Monte Carlo, TD(0)
│   ├── td_lambda.py            # TD(λ) with eligibility traces
│   └── plot_results.py
├── control/
│   ├── sarsa_qlearning.py      # SARSA, Q-learning
│   ├── mc_control.py           # Monte Carlo control
│   ├── n_step_sarsa.py         # n-step SARSA
│   ├── double_q_learning.py    # Double Q-learning
│   ├── prioritised_q_learning.py  # PER
│   └── plot_results.py
├── policy_gradient/
│   ├── reinforce.py            # REINFORCE w/ baseline (LunarLander-v3)
│   ├── actor_critic.py         # Online Actor-Critic (LunarLander-v3)
│   ├── ppo.py                  # PPO + GAE (LunarLander-v3)
│   └── plot_results.py
└── continuous_control/
    ├── ddpg.py                 # DDPG (HalfCheetah-v4)
    ├── sac.py                  # SAC + auto-α (HalfCheetah-v4)
    └── plot_results.py
```

---

## Environments

| Folder | Environment | Type |
|---|---|---|
| `dynamic_programming/` | `Taxi-v3` | Tabular, model-based |
| `prediction/` | `Taxi-v3` | Tabular, model-free |
| `control/` | `Taxi-v3` | Tabular, model-free |
| `policy_gradient/` | `LunarLander-v3` | Discrete actions, neural net |
| `continuous_control/` | `HalfCheetah-v4` | Continuous actions, neural net |

---

## Algorithms

### Dynamic Programming
| Algorithm | Key idea |
|---|---|
| Value Iteration | Collapse eval + improvement into one max sweep |
| Policy Iteration | Alternate full eval and greedy improvement |

### Prediction
| Algorithm | Key idea |
|---|---|
| Monte Carlo | Full-episode returns, first-visit only |
| TD(0) | Bootstrap from next state after every step |
| TD(λ) | Eligibility traces blend TD and MC credit assignment |

### Control
| Algorithm | Key idea |
|---|---|
| SARSA | On-policy TD — updates use the action actually taken |
| Q-learning | Off-policy TD — updates always use the greedy action |
| MC Control | First-visit MC with optimistic initialisation |
| n-step SARSA | Interpolates between TD(0) and MC via n |
| Double Q-learning | Two tables decouple selection from evaluation, removing maximisation bias |
| Prioritised Q-learning | Sample high-TD-error transitions more often; IS weights correct the bias |

### Policy Gradient
| Algorithm | Key idea |
|---|---|
| REINFORCE | MC policy gradient; baseline reduces variance without changing the expected gradient |
| Actor-Critic | Online TD advantage; updates every step instead of waiting for episode end |
| PPO | Clipped surrogate objective keeps updates inside a trust region; GAE for advantages |

### Continuous Control
| Algorithm | Key idea |
|---|---|
| DDPG | Deterministic policy + Gaussian exploration noise; target networks stabilise training |
| SAC | Stochastic policy + entropy bonus; twin critics suppress Q overestimation; auto-α removes the hardest hyperparameter |

---

## Installation

```bash
git clone https://github.com/your-username/rl-zoo-from-scratch.git
cd rl-zoo-from-scratch
pip install -r requirements.txt
```

For `HalfCheetah-v4` you also need MuJoCo:
```bash
pip install mujoco
```

---

## Usage

Each algorithm file is self-contained and runnable. The `run_agent()` function at the bottom of every file trains the agent and (where applicable) saves the model and opens a render window.

```python
# uncomment run_agent() in the __main__ block
# e.g. in control/sarsa_qlearning.py:
if __name__ == "__main__":
    run_agent()
```

To reproduce all plots for a folder:
```bash
cd control/
python plot_results.py
```

To load a saved model and watch it run:
```python
from policy_gradient.ppo import load_model, watch
model = load_model("ppo_lunarlander.pt")
watch(model, episodes=5)
```

---

## Design Conventions

- No global `env`, `n_states`, or `n_actions` — all derived from the `env` parameter inside each function
- `terminated` and `truncated` handled separately everywhere — truncation bootstraps, termination zeros the value target
- Every training function returns a history list for plotting
- `run_agent()` is always commented out by default
- `plot_results.py` in each folder imports from sibling files and produces a single comparison figure

---

## Results

### Prediction — Taxi-v3

Evaluated under a random policy. The 0% completion rate is expected — a random agent almost never solves Taxi-v3, but the value estimates are still valid and comparable against the DP ground truth (V = −52.815). TD methods halve the per-state estimation error compared to Monte Carlo (≈20 vs ≈38), and increasing λ toward 1.0 further reduces error by blending in more multi-step returns.

![Prediction results](assets/prediction_results.png)

| Algorithm | Mean \|V_estimated − V_true\| |
|---|---|
| TD(λ=0.9) | 19.852 |
| TD(0) | 19.968 |
| Monte Carlo | 38.156 |

---

### Control — Taxi-v3

| Algorithm | Eval reward |
|---|---|
| Q-learning | 7.96 ± 2.57 |
| Double Q-learning | — |
| PER | — |
| SARSA | — |
| MC Control | — |

---

### Policy Gradient — LunarLander-v3

Solved threshold = mean reward ≥ 200 over 100 episodes. All methods trained for 3000 episodes.

PPO is the only method that approaches the solve threshold. REINFORCE with baseline outperforms plain REINFORCE as expected — subtracting a state-dependent baseline reduces gradient variance without biasing the update.

Actor-Critic underperforms REINFORCE+b despite using a learned critic. This is a known failure mode of online TD(0) Actor-Critic: the critic and actor chase each other early in training — the actor shifts the policy before the critic has reliable value estimates, producing noisy advantages that destabilise both networks. Without experience replay or GAE, the critic never gets a stable regression target. PPO addresses this through clipped updates, minibatch replay, and Generalised Advantage Estimation.

![Policy gradient results](assets/policy_gradient_results.png)

| Algorithm | Eval reward |
|---|---|
| PPO | 188.2 |
| REINFORCE w/ baseline | −570.2 |
| Actor-Critic | −584.5 |
| REINFORCE | −785.3 |

**PPO agent — LunarLander-v3:**

<video src="assets/lunarlander_ppo.mov" controls width="640"></video>

---

### Continuous Control — HalfCheetah-v4

| Algorithm | Eval reward |
|---|---|
| SAC | — |
| DDPG | — |

**SAC agent — HalfCheetah-v4:**

<video src="assets/sac_halfcheetah.mov" controls width="640"></video>

---

## References

- Sutton & Barto — [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- Schulman et al. — [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- Haarnoja et al. — [Soft Actor-Critic](https://arxiv.org/abs/1812.05905)
- Lillicrap et al. — [DDPG](https://arxiv.org/abs/1509.02971)
- Schaul et al. — [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
