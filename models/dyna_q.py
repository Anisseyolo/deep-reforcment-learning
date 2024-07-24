import numpy as np
from typing import Type, Dict
from collections import defaultdict
from tqdm import tqdm

def dyna_q(env_class: Type,
           gamma: float = 0.99,
           alpha: float = 0.1,
           epsilon: float = 0.1,
           planning_steps: int = 5,
           nb_iter: int = 1000,
           max_steps: int = 100) -> Dict[int, int]:

    Q = defaultdict(lambda: np.zeros(len(env_class().available_actions())))
    model = defaultdict(list)

    for _ in tqdm(range(nb_iter), desc="Dyna-Q Training Progress"):
        env = env_class()
        env.reset()
        for _ in range(max_steps):
            s = env.state_id()
            aa = env.available_actions()

            if np.random.rand() < epsilon:
                a = np.random.choice(aa)
            else:
                a = np.argmax(Q[s])

            prev_score = env.score()
            env.step(a)
            s_prime = env.state_id()
            r = env.score() - prev_score

            Q[s][a] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s][a])

            model[(s, a)] = (r, s_prime)

            for _ in range(planning_steps):
                s_rand, a_rand = list(model.keys())[np.random.choice(len(model))]
                r, s_prime = model[(s_rand, a_rand)]
                Q[s_rand][a_rand] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s_rand][a_rand])

            if env.is_game_over():
                break

    Pi = {s: np.argmax(q_values) for s, q_values in Q.items()}
    return Pi
