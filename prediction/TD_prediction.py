import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym


# environment 

env = gym.make("Taxi-v3")

N_STATES  = env.observation_space.n   # 500
N_ACTIONS = env.action_space.n        # 6



# td(0) prediction 

def td_prediction(
    policy:    np.ndarray,
    env:       gym.Env,
    episodes:  int   = 10000,
    gamma:     float = 0.9,
    alpha:     float = 0.1,
    max_steps: int   = 5000,
    log_every: int   = 500,
) -> tuple[np.ndarray, list[float], int]:
    """
    TD(0) prediction — bootstraps off the next state's value estimate
    instead of waiting for the episode to end.

    The key difference from MC: V[state] gets updated after every single step,
    not at the end of the episode. Much faster to propagate information
    backwards through state space, especially in long episodes like Taxi.

    Returns V, mean-V history for plotting, and how many episodes completed
    (reached the goal, not just hit max_steps) — useful sanity check.
    """
    V          = np.zeros(N_STATES)
    mean_vs    = []
    completed  = 0

    for ep in range(episodes):
        state, _ = env.reset()
        done     = False
        steps    = 0

        while not done and steps < max_steps:
            action                             = policy[state]
            next_state, reward, done, _, _     = env.step(action)

            # bootstrap: use current estimate of V[next_state] instead of full return
            td_target   = reward + gamma * V[next_state] * (not done)
            V[state]   += alpha * (td_target - V[state])

            state  = next_state
            steps += 1

        if done:
            completed += 1

        if (ep + 1) % log_every == 0:
            mean_vs.append(V.mean())
            print(f"[TD] ep {ep+1:6d} | mean V: {V.mean():.3f}")

    print(f"[TD] completed {completed}/{episodes} episodes ({100*completed/episodes:.1f}%)")
    return V, mean_vs, completed



# main 

if __name__ == "__main__":

    np.random.seed(42)
    random_policy = np.random.randint(0, N_ACTIONS, size=N_STATES)


    print("\n" + "=" * 50)
    print("TD(0) Prediction")
    print("=" * 50)
    V_td, means_td, _ = td_prediction(random_policy, env)
