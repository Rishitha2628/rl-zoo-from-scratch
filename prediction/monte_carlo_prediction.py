import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym


# environment 

env = gym.make("Taxi-v3")

N_STATES  = env.observation_space.n   # 500
N_ACTIONS = env.action_space.n        # 6


# monte carlo prediction 

def monte_carlo_prediction(
    policy:    np.ndarray,
    env:       gym.Env,
    episodes:  int   = 50000,
    gamma:     float = 0.9,
    alpha:     float = 0.1,
    max_steps: int   = 5000,
    log_every: int   = 2000,
) -> tuple[np.ndarray, list[float]]:
    """
    First-visit MC prediction with incremental mean updates.

    Runs full episodes, then walks backwards to compute returns.
    Only the first visit to each state in an episode counts — revisits are skipped.
    alpha replaces 1/N(s) for stability when episodes are long and noisy.

    max_steps caps episodes so a bad random policy doesn't wander forever.
    """
    V        = np.zeros(N_STATES)
    mean_vs  = []   # track mean V over time to plot learning progress

    for ep in range(episodes):
        state, _ = env.reset()
        episode  = []
        done     = False
        steps    = 0

        while not done and steps < max_steps:
            action                           = policy[state]
            next_state, reward, done, _, _   = env.step(action)
            episode.append((state, reward))
            state = next_state
            steps += 1

        # walk backwards through the episode accumulating discounted return
        G       = 0.0
        visited = set()

        for s, reward in reversed(episode):
            G = reward + gamma * G

            if s not in visited:
                visited.add(s)
                V[s] += alpha * (G - V[s])   # incremental update toward G

        if (ep + 1) % log_every == 0:
            mean_vs.append(V.mean())
            print(f"[MC] ep {ep+1:6d} | mean V: {V.mean():.3f}")

    return V, mean_vs



# main 

if __name__ == "__main__":

    np.random.seed(42)
    random_policy = np.random.randint(0, N_ACTIONS, size=N_STATES)

    print("\n" + "=" * 50)
    print("Monte Carlo Prediction")
    print("=" * 50)
    V_mc, means_mc = monte_carlo_prediction(random_policy, env)
