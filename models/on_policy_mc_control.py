import numpy as np
from tqdm import tqdm
from typing import Type, Dict, List, Tuple

def on_policy_first_visit_monte_carlo_control(env_class: Type,
                                              gamma: float = 0.9,
                                              num_episodes: int = 10000,
                                              max_steps: int = 10) -> Dict[Tuple[int, int], int]:
    Pi = {}
    Q = {}
    Returns = {}

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        env = env_class().from_random_state()
        state = env.reset()
        done = False
        trajectory = []
        steps_count = 0

        while not done and steps_count < max_steps:
            state_tuple = tuple(state)
            if state_tuple not in Pi:
                Pi[state_tuple] = np.random.choice(env.available_actions())

            action = Pi[state_tuple]
            next_state, reward, done = env.step(action)
            next_state_tuple = tuple(next_state)
            trajectory.append((state_tuple, action, reward))
            state = next_state
            steps_count += 1

        G = 0
        visited_pairs = set()
        for (state, action, reward) in reversed(trajectory):
            G = gamma * G + reward
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                if (state, action) not in Returns:
                    Returns[(state, action)] = []
                Returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(Returns[(state, action)])
                best_action = max(env.available_actions(), key=lambda a: Q.get((state, a), 0))
                Pi[state] = best_action

    return Pi
