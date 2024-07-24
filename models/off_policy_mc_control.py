import numpy as np
from tqdm import tqdm
from typing import Type, Dict, List, Tuple

def off_policy_monte_carlo_control(env_class: Type,
                                   gamma: float = 0.9,
                                   num_episodes: int = 10000,
                                   epsilon: float = 0.1,
                                   max_steps: int = 10) -> Dict[Tuple[int, int], int]:
    Pi = {}
    Q = {}
    Returns = {}
    C = {}

    def behavior_policy(state):
        return np.random.choice(env_class().available_actions()) if np.random.rand() < epsilon else Pi.get(state, np.random.choice(env_class().available_actions()))

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        env = env_class().from_random_state()
        state = env.reset()
        action = behavior_policy(state)
        trajectory = [(state, action, 1)]
        done = False
        steps_count = 0

        while not done and steps_count < max_steps:
            next_state, reward, done = env.step(action)
            next_state_tuple = tuple(next_state)
            if not done:
                next_action = behavior_policy(next_state)
            else:
                next_action = None
            trajectory.append((next_state_tuple, next_action, reward))
            state, action = next_state, next_action
            steps_count += 1

        G = 0
        W = 1
        visited_pairs = set()
        for (state, action, reward) in reversed(trajectory):
            G = gamma * G + reward
            state_tuple = tuple(state)
            if (state_tuple, action) not in visited_pairs:
                visited_pairs.add((state_tuple, action))
                if (state_tuple, action) not in Returns:
                    Returns[(state_tuple, action)] = []
                    C[(state_tuple, action)] = 0
                C[(state_tuple, action)] += W
                Returns[(state_tuple, action)].append(W * G)
                Q[(state_tuple, action)] = np.mean(Returns[(state_tuple, action)])
                Pi[state_tuple] = max(env_class().available_actions(), key=lambda a: Q.get((state_tuple, a), 0))
            if action is not None:
                W *= 1 / (epsilon if action == behavior_policy(state) else 1 - epsilon)

    return Pi
