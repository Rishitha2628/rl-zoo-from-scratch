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


# value iteration 

def value_iteration(
    env: gym.Env,
    gamma: float = 0.9,
    theta: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Collapse eval + improvement into one sweep — just take the max over actions
    instead of following a fixed policy. Skips the outer policy loop entirely.

    Policy is extracted at the end with a single greedy pass over the converged V.
    Returns the optimal policy, value function, and per-sweep delta history.
    """
    V = np.zeros(N_STATES)
    deltas = []
    sweep  = 0

    while True:
        delta = 0.0
        sweep += 1

        for s in range(N_STATES):
            q_values = np.zeros(N_ACTIONS)

            for a in range(N_ACTIONS):
                for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                    q_values[a] += prob * (reward + gamma * V[next_s] * (not done))

            best = np.max(q_values)
            delta = max(delta, abs(best - V[s]))
            V[s] = best

        deltas.append(delta)
        print(f"[VI] sweep {sweep:3d} | delta: {delta:.8f}")

        if delta < theta:
            break

    print(f"[VI] converged in {sweep} sweeps")

    # single greedy pass to recover the policy from V
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        q_values = np.zeros(N_ACTIONS)
        for a in range(N_ACTIONS):
            for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_s] * (not done))
        policy[s] = np.argmax(q_values)

    return policy, V, deltas


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

    print("Value Iteration")
    print("=" * 50)
    policy_vi, V_vi, deltas_vi = value_iteration(env)

    # to watch the agent run in the rendered environment
    run_agent(policy_vi)
