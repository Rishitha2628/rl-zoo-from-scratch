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
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])


# monte carlo control 

def monte_carlo_control(
    env:            gym.Env,
    episodes:       int   = 50000,
    gamma:          float = 0.9,
    alpha:          float = 0.1,
    epsilon_start:  float = 10.0,
    max_steps:      int   = 500,
    log_every:      int   = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    """
    First-visit MC control with epsilon-greedy exploration.

    Q is initialised to 10 instead of 0 — optimistic initialisation.
    Every action looks promising at the start, so the agent is forced to try
    everything before it can trust any Q value. Essentially free exploration
    without needing a separate exploration bonus. Works well when you know
    a rough upper bound on the true Q values.

    Unlike TD methods, the Q update only happens at the END of each episode
    after computing the full return G. This means slow early learning on Taxi
    since many random episodes never even reach the goal.

    max_steps caps wandering episodes — without it, a bad policy can loop
    for thousands of steps and skew the return estimates badly.
    """
    # optimistic init — every (s, a) looks like it's worth 10 to start
    Q = np.ones((N_STATES, N_ACTIONS)) * 10.0

    completed  = 0
    mean_vs    = []
    comp_rates = []   # completion rate over time — interesting to watch this grow

    for ep in range(1, episodes + 1):
        epsilon  = epsilon_start / ep
        state, _ = env.reset()
        done     = False
        episode  = []
        steps    = 0

        while not done and steps < max_steps:
            action                          = epsilon_greedy(Q, state, epsilon, env)
            next_state, reward, done, _, _  = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1

        if done:
            completed += 1

        # walk back through the episode computing discounted returns
        G       = 0.0
        visited = set()

        for s, a, reward in reversed(episode):
            G = reward + gamma * G

            if (s, a) not in visited:
                visited.add((s, a))
                Q[s][a] += alpha * (G - Q[s][a])

        if ep % log_every == 0:
            mean_v    = np.max(Q, axis=1).mean()
            comp_rate = completed / ep * 100
            mean_vs.append(mean_v)
            comp_rates.append(comp_rate)
            print(
                f"[MC] ep {ep:6d} | mean V: {mean_v:.3f} | "
                f"ε: {epsilon:.5f} | completed: {comp_rate:.2f}%"
            )

    policy = np.argmax(Q, axis=1)
    V      = np.max(Q, axis=1)
    return policy, Q, V, mean_vs, comp_rates


# evaluation 

def evaluate_policy(
    policy:   np.ndarray,
    env:      gym.Env,
    episodes: int = 1000,
) -> tuple[float, float]:
    """Greedy rollouts — no exploration. Returns mean ± std reward."""
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done     = False
        total    = 0.0

        while not done:
            action              = policy[state]
            state, r, done, _, _ = env.step(action)
            total              += r

        rewards.append(total)

    return float(np.mean(rewards)), float(np.std(rewards))


# main 

if __name__ == "__main__":

    np.random.seed(42)

    print("Running value iteration for reference...")

    print("\n" + "=" * 50)
    print("Monte Carlo Control")
    print("=" * 50)
    policy_mc, Q_mc, V_mc, mean_vs, comp_rates = monte_carlo_control(env)

    print(f"\nMC result    | mean V: {V_mc.mean():.3f} | V(328): {V_mc[328]:.3f}")



    