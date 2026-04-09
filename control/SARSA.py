import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym


# environment 

env = gym.make("Taxi-v3")

N_STATES  = env.observation_space.n   # 500
N_ACTIONS = env.action_space.n        # 6

# epsilon greedy 

def epsilon_greedy(
    Q:       np.ndarray,
    state:   int,
    epsilon: float,
    env:     gym.Env,
) -> int:
    """
    With probability epsilon, explore randomly.
    Otherwise exploit the current best action.

    epsilon decays over time in both SARSA and Q-learning
    so early episodes explore widely, later ones lock in the good policy.
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])


# sarsa 

def sarsa(
    env:            gym.Env,
    episodes:       int   = 80000,
    gamma:          float = 0.9,
    alpha:          float = 0.1,
    epsilon_start:  float = 10.0,
    log_every:      int   = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    On-policy TD control. Picks the next action BEFORE the update,
    so the Q update uses the action actually taken — not the greedy best.

    This makes SARSA more conservative than Q-learning: it accounts for
    the fact that it'll sometimes take exploratory (bad) actions.

    epsilon_start=10.0 is intentionally high — on-policy methods need
    a lot more exploration to cover the state space, so we decay slowly.
    That's also why we run 80k episodes vs 10k for Q-learning.
    """
    Q       = np.zeros((N_STATES, N_ACTIONS))
    mean_vs = []

    for ep in range(1, episodes + 1):
        state, _  = env.reset()
        epsilon   = epsilon_start / ep
        action    = epsilon_greedy(Q, state, epsilon, env)
        done      = False

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon, env)

            # update uses next_action, not max — that's the on-policy part
            td_target        = reward + gamma * Q[next_state][next_action] * (not done)
            Q[state][action] += alpha * (td_target - Q[state][action])

            state  = next_state
            action = next_action

        if ep % log_every == 0:
            mean_v = np.max(Q, axis=1).mean()
            mean_vs.append(mean_v)
            print(f"[SARSA] ep {ep:6d} | mean V: {mean_v:.3f} | ε: {epsilon:.5f}")

    policy = np.argmax(Q, axis=1)
    V      = np.max(Q, axis=1)
    return policy, Q, V, mean_vs



# evaluation 

def evaluate_policy(
    policy:   np.ndarray,
    env:      gym.Env,
    episodes: int = 1000,
) -> tuple[float, float]:
    """Run the policy greedy (no exploration) and return mean ± std reward."""
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done     = False
        total    = 0.0

        while not done:
            action             = policy[state]
            state, r, done, _, _ = env.step(action)
            total             += r

        rewards.append(total)

    return float(np.mean(rewards)), float(np.std(rewards))


# rollout 

def run_agent(policy: np.ndarray, episodes: int = 3) -> None:
    env_render = gym.make("Taxi-v3", render_mode="human")

    for ep in range(episodes):
        state, _ = env_render.reset()
        done     = False
        total_r  = 0
        steps    = 0

        print(f"\n--- episode {ep + 1} ---")

        while not done:
            action              = policy[state]
            state, r, done, _, _ = env_render.step(action)
            total_r            += r
            steps              += 1
            time.sleep(0.25)

        print(f"solved in {steps} steps | total reward: {total_r}")

    env_render.close()

# main 

if __name__ == "__main__":

    np.random.seed(42)

    print("\n" + "=" * 50)
    print("SARSA (on-policy)")
    print("=" * 50)
    policy_sarsa, Q_sarsa, V_sarsa, means_sarsa = sarsa(env)

    # uncomment to watch the agent in the rendered environment
    run_agent(policy_sarsa)