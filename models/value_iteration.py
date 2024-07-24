import numpy as np

def value_iteration(env, gamma=0.9, theta=1e-6):
    width, height = env.width, env.height
    n_actions = len(env.available_actions())

    V = np.zeros((width, height))
    policy = np.zeros((width, height), dtype=int)

    def get_state_index(state):
        return state[1], state[0]

    while True:
        delta = 0
        for x in range(width):
            for y in range(height):
                state = (x, y)
                action_values = np.zeros(n_actions)
                for a in range(n_actions):
                    next_state, reward, done = env.step(a)
                    next_x, next_y = next_state
                    action_values[a] = reward + gamma * V[next_x, next_y]
                best_action_value = np.max(action_values)
                delta = max(delta, abs(V[x, y] - best_action_value))
                V[x, y] = best_action_value
                policy[x, y] = np.argmax(action_values)
        if delta < theta:
            break

    return policy, V