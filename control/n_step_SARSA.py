import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import time


# epsilon-greedy 

def epsilon_greedy(Q, state, epsilon, env):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])


# n-step SARSA 

def n_step_sarsa(env, n=5, episodes=10_000, gamma=0.9,
                 alpha=0.1, epsilon_start=1.0, log_every=200):
    """
    On-policy TD control with n-step returns.
    Larger n → lower bias, higher variance; needs more episodes to converge.
    Returns policy, Q, V, and a mean-V history for plotting.
    """
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q         = np.zeros((n_states, n_actions))
    mean_vs   = []

    for ep in range(1, episodes + 1):
        epsilon = epsilon_start / ep          # decay faster for small n_start
        state, _ = env.reset()
        action    = epsilon_greedy(Q, state, epsilon, env)

        # sliding window of (s, a, r); position 0 is the state being updated
        buffer = deque()
        buffer.append((state, action, 0))

        t = 0
        T = float('inf')                      # set when terminal is reached

        while True:
            if t < T:
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_action = epsilon_greedy(Q, next_state, epsilon, env)
                buffer.append((next_state, next_action, reward))

                if terminated or truncated:
                    T = t + 1

                action = next_action
                state  = next_state

            # tau is the step whose Q-value we'r──e updating this iteration
            tau = t - n + 1

            if tau >= 0:
                # accumulate discounted rewards over the n-step window
                G = sum(
                    (gamma ** i) * r
                    for i, (_, _, r) in enumerate(list(buffer)[1 : n + 1])
                )

                # bootstrap from the state n steps ahead unless terminal
                if tau + n < T:
                    sn, an, _ = list(buffer)[n]
                    G += (gamma ** n) * Q[sn][an]

                s_tau, a_tau, _ = list(buffer)[0]
                Q[s_tau][a_tau] += alpha * (G - Q[s_tau][a_tau])

                buffer.popleft()              # advance the window

            t += 1

            if tau >= T - 1:
                break

        if ep % log_every == 0:
            mean_vs.append(np.max(Q, axis=1).mean())

    policy = np.argmax(Q, axis=1)
    V      = np.max(Q, axis=1)
    return policy, Q, V, mean_vs


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


# entry point 


if __name__ == "__main__":

    env = gym.make("Taxi-v3")
    np.random.seed(42)

    n_values = [1, 3, 5, 10]
    results  = {}

    for n in n_values:
        # scale epsilon_start with n so exploration matches the longer horizon
        policy, Q, V, mean_vs = n_step_sarsa(
            env,
            n=n,
            episodes=10_000,
            epsilon_start=float(n),
        )
        results[n] = (policy, Q, V, mean_vs)
        print(f"n={n:2d}  V(328): {V[328]:.2f}  mean V: {V.mean():.2f}")


    run_agent(policy)
    pass