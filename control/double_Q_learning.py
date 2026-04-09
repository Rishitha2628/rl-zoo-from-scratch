import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time


# epsilon-greedy for double Q 

def epsilon_greedy(Q1, Q2, state, epsilon, env):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    # sum both tables — unbiased estimate of the action value
    return np.argmax(Q1[state] + Q2[state])



# double Q-learning 

def double_q_learning(env, episodes=20_000, gamma=0.9,
                      alpha=0.1, epsilon_start=1.0, log_every=200):
    """
    Two independent Q-tables; each update randomly assigns one to select the
    action and the other to evaluate it. Decoupling selection from evaluation
    removes the maximisation bias that inflates Q-values in standard Q-learning.
    """
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    mean_vs = []
    max_qs  = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        epsilon  = epsilon_start / ep
        done     = False

        while not done:
            action = epsilon_greedy(Q1, Q2, state, epsilon, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if np.random.random() < 0.5:
                # Q1 selects, Q2 evaluates — breaks the upward bias
                best_action = np.argmax(Q1[next_state])
                td_target   = reward + gamma * Q2[next_state][best_action] * (not terminated)
                Q1[state][action] += alpha * (td_target - Q1[state][action])
            else:
                best_action = np.argmax(Q2[next_state])
                td_target   = reward + gamma * Q1[next_state][best_action] * (not terminated)
                Q2[state][action] += alpha * (td_target - Q2[state][action])

            state = next_state

        if ep % log_every == 0:
            V = np.max(Q1 + Q2, axis=1) / 2
            mean_vs.append(V.mean())
            max_qs.append(np.max((Q1 + Q2) / 2))

    policy = np.argmax(Q1 + Q2, axis=1)
    V      = np.max(Q1 + Q2, axis=1) / 2
    return policy, Q1, Q2, V, mean_vs, max_qs

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



if __name__ == "__main__":

    env = gym.make("Taxi-v3")
    np.random.seed(42)

    res_dql = double_q_learning(env)
    pol_dql, Q1, Q2, V_dql, _, _ = res_dql

    print(f"{'':18}  {'Double QL':>10}  {'Q-learning':>10}  {'DP':>6}")

    run_agent(res_dql)
    pass