import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym


# environment 

env = gym.make("Taxi-v3")
env.reset()

N_STATES  = env.observation_space.n   # 500
N_ACTIONS = env.action_space.n        # 6

ACTION_NAMES = ["south", "north", "east", "west", "pickup", "dropoff"]


def inspect_state(state: int) -> None:

    # check the transition table for a particular state

    print(f"\nTransitions from state {state}:\n")
    for action, name in enumerate(ACTION_NAMES):
        transitions = env.unwrapped.P[state][action]
        print(f"  {name:8s}: {transitions}")


# policy evaluation

def policy_evaluation(
    policy: np.ndarray,
    env: gym.Env,
    gamma: float = 0.9,
    theta: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    
    # Iteratively evaluates a fixed policy until the value function stops changing.
    
    V = np.zeros(N_STATES)
    deltas = []

    while True:
        delta = 0.0

        for s in range(N_STATES):
            a = policy[s]
            new_v = 0.0

            for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                new_v += prob * (reward + gamma * V[next_s] * (not done))

            delta = max(delta, abs(new_v - V[s]))
            V[s]  = new_v

        deltas.append(delta)

        if delta < theta:
            break

    return V, deltas


# policy improvement 

def policy_improvement(
    V: np.ndarray,
    env: gym.Env,
    gamma: float = 0.9,
) -> np.ndarray:
    """
    One-step lookahead over V to get a greedy policy.
    Called after every policy evaluation round.
    """
    policy = np.zeros(N_STATES, dtype=int)

    for s in range(N_STATES):
        q_values = np.zeros(N_ACTIONS)

        for a in range(N_ACTIONS):
            for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_s] * (not done))

        policy[s] = np.argmax(q_values)

    return policy


# policy iteration 

def policy_iteration(
    env: gym.Env,
    gamma: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Alternate between policy evaluation and greedy improvement until the
    policy stops changing. Usually converges in few iterations.

    Returns the optimal policy, its value function, and the delta history
    from the final evaluation round.
    """
    np.random.seed(42)
    policy = np.random.randint(0, N_ACTIONS, size=N_STATES)

    iteration = 0
    last_deltas = []

    while True:
        iteration += 1

        V, last_deltas  = policy_evaluation(policy, env, gamma)
        new_policy = policy_improvement(V, env, gamma)

        if np.array_equal(new_policy, policy):
            print(f"[PI] converged after {iteration} iterations")
            break

        policy = new_policy
        print(f"[PI] iteration {iteration:2d} | mean V: {V.mean():.3f}")

    return policy, V, last_deltas



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
            action          = policy[state]
            state, r, done, truncated, _ = env_render.step(action)
            total_r        += r
            steps          += 1
            time.sleep(0.25)

        print(f"solved in {steps} steps | total reward: {total_r}")

    env_render.close()


# main 

if __name__ == "__main__":

    print("Policy Iteration")
    print("=" * 50)
    policy_pi, V_pi, deltas_pi = policy_iteration(env)
    
    # to watch the agent run in the rendered environment
    run_agent(policy_pi)