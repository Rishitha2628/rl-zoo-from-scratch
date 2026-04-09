import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time


# prioritized replay buffer 

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10_000, alpha=0.6):
        self.capacity   = capacity
        self.alpha      = alpha          # how strongly priority skews sampling
        self.buffer     = []
        self.priorities = []
        self.position   = 0

    def add(self, transition, td_error):
        # small epsilon avoids zero-priority — every transition stays in contention
        priority = (abs(td_error) + 0.01) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position]     = transition
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs      = priorities / priorities.sum()
        indices    = np.random.choice(len(self.buffer), batch_size,
                                      p=probs, replace=False)

        # IS weights undo the bias introduced by non-uniform sampling
        N       = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()           # normalise so the largest weight is 1

        return [self.buffer[i] for i in indices], indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = (abs(td_error) + 0.01) ** self.alpha

    def __len__(self):
        return len(self.buffer)


# prioritized Q-learning 

def prioritized_q_learning(env, episodes=10_000, gamma=0.9,
                            alpha=0.1, epsilon_start=1.0,
                            batch_size=32, beta_start=0.4,
                            log_every=200):
    """
    Q-learning with prioritized experience replay (PER).
    High-TD-error transitions are sampled more often; IS weights correct the bias.
    beta anneals toward 1.0 so the IS correction becomes exact at convergence.
    """
    n_states  = env.observation_space.n
    n_actions = env.action_space.n          # noqa: F841
    Q         = np.zeros((n_states, n_actions))
    buffer    = PrioritizedReplayBuffer()
    beta      = beta_start
    mean_vs   = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        epsilon  = epsilon_start / ep
        done     = False
        step     = 0

        while not done and step < 1_000:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            td_error = (reward + gamma * np.max(Q[next_state]) * (not terminated)
                        - Q[state][action])
            # store `terminated`, not `done` — truncation shouldn't zero the target
            buffer.add((state, action, reward, next_state, terminated), td_error)

            if len(buffer) >= batch_size:
                # anneal toward full IS correction as Q converges
                beta = min(1.0, beta + 0.0001)

                transitions, indices, weights = buffer.sample(batch_size, beta)
                td_errors = []
                for j, (s, a, r, s_, term) in enumerate(transitions):
                    td_target  = r + gamma * np.max(Q[s_]) * (not term)
                    td_error_j = td_target - Q[s][a]
                    Q[s][a]   += alpha * weights[j] * td_error_j
                    td_errors.append(td_error_j)

                buffer.update_priorities(indices, td_errors)

            state  = next_state
            step  += 1

        if ep % log_every == 0:
            mean_vs.append(np.max(Q, axis=1).mean())

    policy = np.argmax(Q, axis=1)
    V      = np.max(Q, axis=1)
    return policy, Q, V, mean_vs



# entry point 

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

    res_per = prioritized_q_learning(env)
    pol_per, Q_per, V_per, _ = res_per
    run_agent(pol_per)
    pass