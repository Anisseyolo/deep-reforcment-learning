import numpy as np
from tqdm import tqdm
from typing import Dict, Type


def naive_monte_carlo_with_exploring_starts(env_class: Type,
                                            gamma: float = 0.999,
                                            nb_iter: int = 500,
                                            max_steps: int = 10) -> Dict[int, int]:
    Pi = {}
    Q = {}
    Returns = {}

    for it in tqdm(range(nb_iter)):
        env = env_class().from_random_state()

        is_first_action = True
        trajectory = []
        steps_count = 0

        while not env.is_game_over() and steps_count < max_steps:
            s = env.state_id()
            aa = env.available_actions()

            if s not in Pi:
                Pi[s] = np.random.choice(aa)

            if is_first_action:
                a = np.random.choice(aa)
                is_first_action = False
            else:
                a = Pi[s]

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            trajectory.append((s, a, r, aa))
            steps_count += 1

            env.display()
            # print(f"Step: {steps_count}, Action: {a}, Reward: {r}, State: {env.state_id()}")

        G = 0
        for (t, (s, a, r, aa)) in reversed(list(enumerate(trajectory))):
            G = gamma * G + r

            if all(map(lambda triplet: triplet[0] != s or triplet[1] != a, trajectory[:t])):
                if (s, a) not in Returns:
                    Returns[(s, a)] = []
                Returns[(s, a)].append(G)
                Q[(s, a)] = np.mean(Returns[(s, a)])

                best_a = None
                best_a_score = -float('inf')

                for action in aa:
                    if (s, action) not in Q:
                        Q[(s, action)] = np.random.random()
                    if Q[(s, action)] > best_a_score:
                        best_a = action
                        best_a_score = Q[(s, action)]

                Pi[s] = best_a

    return Pi
