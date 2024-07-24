import numpy as np


def q_learning(env_class, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    env = env_class()
    n_actions = len(env.available_actions())
    state_space_size = env.max_position + 1

    Q = np.zeros((state_space_size, n_actions))

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(env.available_actions())
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done = env.step(action)
            best_next_action = np.argmax(Q[next_state, :])

            target = reward + gamma * Q[next_state, best_next_action]

            Q[state, action] = Q[state, action] + alpha * (target - Q[state, action])

            state = next_state

    return Q
