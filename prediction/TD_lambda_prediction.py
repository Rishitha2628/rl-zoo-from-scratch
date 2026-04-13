import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym


# environment 

env = gym.make("Taxi-v3")

N_STATES  = env.observation_space.n   # 500
N_ACTIONS = env.action_space.n        # 6


# td(λ) prediction 

def td_lambda_prediction(
    policy:    np.ndarray,
    env:       gym.Env,
    episodes:  int   = 10000,
    gamma:     float = 0.9,
    alpha:     float = 0.1,
    lam:       float = 0.9,
    log_every: int   = 2000,
) -> tuple[np.ndarray, list[float]]:
    """
    TD(λ) with accumulating eligibility traces.

    The trace e[s] tracks how recently and how often each state was visited.
    After every step, ALL states get a V update — but states visited long ago
    have near-zero traces so their update is basically nothing.

    λ controls the credit assignment horizon:
      λ=0  → pure TD(0), only the current state gets updated
      λ=1  → approaches MC, credit spreads all the way back through the episode
      λ∈(0,1) → somewhere in between, usually 0.9 works well in practice

    """
    V       = np.zeros(N_STATES)
    mean_vs = []

    for ep in range(episodes):
        state, _ = env.reset()
        done     = False
        e        = np.zeros(N_STATES)   # reset traces every episode

        max_steps = 5000
        step = 0

        while not done and step < max_steps:
            step += 1
            action                          = policy[state]
            next_state, reward, done, _, _  = env.step(action)

            td_error  = reward + gamma * V[next_state] * (not done) - V[state]
            e[state] += 1                     # bump trace for the state we just left

            V  += alpha * td_error * e        # all states updated, scaled by trace
            e  *= gamma * lam                 # decay all traces

            state = next_state

        if (ep + 1) % log_every == 0:
            mean_vs.append(V.mean())
            print(f"[TD(λ={lam})] ep {ep+1:6d} | mean V: {V.mean():.3f}")

    return V, mean_vs


# state visitation analysis 

def get_visitation_counts(
    policy:   np.ndarray,
    env:      gym.Env,
    episodes: int = 1000,
) -> np.ndarray:
    """
    Roll out the policy and count how many times each state gets visited.
    Useful to sanity-check which parts of the state space TD(λ) actually sees —
    states that never get visited will have V[s] stuck at 0.
    """
    counts = np.zeros(N_STATES)

    for _ in range(episodes):
        state, _ = env.reset()
        done     = False

        while not done:
            counts[state] += 1
            action         = policy[state]
            state, _, done, _, _ = env.step(action)

    return counts


# main 

if __name__ == "__main__":

    np.random.seed(42)
    random_policy = np.random.randint(0, N_ACTIONS, size=N_STATES)

    print("\n" + "=" * 50)
    print("TD(λ=0.9) Prediction")
    print("=" * 50)
    V_tdl, means_tdl = td_lambda_prediction(random_policy, env, lam=0.9)

    print(f"\nTD(λ) result  | mean V: {V_tdl.mean():.3f} | V(328): {V_tdl[328]:.3f}")

    # sweep across lambda values — shows how λ affects how far credit travels back
    print("\n" + "=" * 50)
    print("Lambda sweep (5k episodes each)")
    print("=" * 50)
    lambda_vals  = [0.0, 0.3, 0.5, 0.9, 1.0]
    lambda_means = {}

    for lam in lambda_vals:
        V_test, _ = td_lambda_prediction(random_policy, env, episodes=5000, lam=lam, log_every=999999)
        lambda_means[lam] = V_test.mean()
        print(f"λ={lam:.1f} | mean V: {V_test.mean():.3f}")

    # how much of the state space does the optimal policy actually visit?
    print("\nRunning state visitation analysis...")
    visit_counts = get_visitation_counts(random_policy, env)
    print(f"Never visited : {np.sum(visit_counts == 0)} / {N_STATES} states")

