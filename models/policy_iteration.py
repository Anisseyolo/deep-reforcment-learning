import numpy as np


def policy_iteration(env, gamma=0.9, theta=1e-6):
    width, height = env.width, env.height
    n_actions = len(env.available_actions())

    policy = np.zeros((width, height), dtype=int)
    V = np.zeros((width, height))

    def get_state_index(state):
        return state[1], state[0]

    def policy_evaluation(policy, V, gamma, theta):
        while True:
            delta = 0
            for x in range(width):
                for y in range(height):
                    state = (x, y)
                    a = policy[x, y]
                    next_state, reward, done = env.step(a)
                    next_x, next_y = next_state
                    next_state_index = get_state_index(next_state)
                    v = V[x, y]
                    if done:
                        V[x, y] = reward
                    else:
                        V[x, y] = reward + gamma * V[next_x, next_y]
                    delta = max(delta, abs(v - V[x, y]))
            if delta < theta:
                break

    def policy_improvement(V, policy, gamma):
        policy_stable = True
        for x in range(width):
            for y in range(height):
                old_action = policy[x, y]
                state = (x, y)
                action_values = np.zeros(n_actions)
                for a in range(n_actions):
                    next_state, reward, done = env.step(a)
                    next_x, next_y = next_state
                    action_values[a] = reward + gamma * V[next_x, next_y]
                best_action = np.argmax(action_values)
                policy[x, y] = best_action
                if old_action != best_action:
                    policy_stable = False
        return policy_stable

    while True:
        policy_evaluation(policy, V, gamma, theta)
        policy_stable = policy_improvement(V, policy, gamma)
        if policy_stable:
            break

    return policy, V