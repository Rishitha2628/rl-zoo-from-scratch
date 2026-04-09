# rl-fundamentals

A clean, readable implementation of core reinforcement learning algorithms in Python using [Gymnasium](https://gymnasium.farama.org/). Built as a learning reference — every file is self-contained, every design choice is commented with *why*, not just *what*.

---

## Structure

```
rl-fundamentals/
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
│   ├── reinforce.py            # REINFORCE w/ baseline (LunarLander-v2)
│   ├── actor_critic.py         # Online Actor-Critic (LunarLander-v2)
│   ├── ppo.py                  # PPO + GAE (LunarLander-v2)
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
| `policy_gradient/` | `LunarLander-v2` | Discrete actions, neural net |
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
git clone https://github.com/your-username/rl-fundamentals.git
cd rl-fundamentals
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

Comparison plots are saved as `.png` files inside each folder after running `plot_results.py`.

**DP — Taxi-v3**
Ground truth: `V(328) = 1.62`, `mean V = 2.47`, `max Q = 20.0`

**Control — Taxi-v3**
| Algorithm | Eval reward |
|---|---|
| Q-learning | 7.96 ± 2.57 |
| Double Q-learning | — |
| PER | — |
| SARSA | — |
| MC Control | — |

**Policy Gradient — LunarLander-v2** *(solved = mean reward ≥ 200)*
| Algorithm | Eval reward |
|---|---|
| PPO | — |
| Actor-Critic | — |
| REINFORCE w/ baseline | — |
| REINFORCE | — |

**Continuous Control — HalfCheetah-v4**
| Algorithm | Eval reward |
|---|---|
| SAC | — |
| DDPG | — |

*Fill in after running `plot_results.py` in each folder.*

---

## References

- Sutton & Barto — [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- Schulman et al. — [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- Haarnoja et al. — [Soft Actor-Critic](https://arxiv.org/abs/1812.05905)
- Lillicrap et al. — [DDPG](https://arxiv.org/abs/1509.02971)
- Schaul et al. — [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
