import numpy as np
import tensorflow as tf
from environments.line_world import LineWorld
from environments.grid_world import GridWorld


def build_model(state_size, action_size):
    inputs = tf.keras.Input(shape=(state_size,))
    x = tf.keras.layers.Dense(24, activation='relu')(inputs)
    x = tf.keras.layers.Dense(24, activation='relu')(x)
    outputs = tf.keras.layers.Dense(action_size, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def episodic_semi_gradient_sarsa(env_class, num_episodes=100, alpha=0.1, gamma=0.99, epsilon=0.1):
    env = env_class()
    state_size = env.width() * env.height() if hasattr(env, 'height') else env.width() + 1
    action_size = len(env.available_actions())
    model = build_model(state_size, action_size)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for episode in range(num_episodes):
        env.reset()
        state = env.state_id()

        if isinstance(state, tuple):
            state_index = state[0] * env.width() + state[1]
        else:
            state_index = state
        state_one_hot = np.eye(state_size)[state_index:state_index + 1]

        if np.random.rand() < epsilon:
            action = np.random.choice(env.available_actions())
        else:
            predictions = model(state_one_hot, training=False).numpy()
            action = np.argmax(predictions[0])

        done = False
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            next_state, reward, done = env.step(action)

            if isinstance(next_state, tuple):
                next_state_index = next_state[0] * env.width() + next_state[1]
            else:
                next_state_index = next_state
            next_state_one_hot = np.eye(state_size)[next_state_index:next_state_index + 1]

            next_predictions = model(next_state_one_hot, training=False).numpy()
            if next_predictions.size == 0:
                print("Next predictions returned an empty array")
                next_action = np.random.choice(env.available_actions())
            else:
                next_action = np.argmax(next_predictions[0])

            target = reward + gamma * next_predictions[0][next_action] if not done else reward
            target_f = model(state_one_hot, training=False).numpy()
            target_f[0][action] = (1 - alpha) * target_f[0][action] + alpha * target

            with tf.GradientTape() as tape:
                predictions = model(state_one_hot, training=True)
                loss = loss_fn(target_f, predictions)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            state = next_state
            state_one_hot = next_state_one_hot
            action = next_action
            steps += 1

        if steps >= max_steps:
            print("Reached maximum steps without completing episode.")

    return model


def main():
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 100

    env_choice = input("Choose environment: 1 for LineWorld, 2 for GridWorld: ").strip()
    if env_choice == '1':
        env_class = LineWorld
        env_name = "LineWorld"
    elif env_choice == '2':
        env_class = GridWorld
        env_name = "GridWorld"
    else:
        print("Invalid choice!")
        return

    model = episodic_semi_gradient_sarsa(env_class, num_episodes, alpha, gamma, epsilon)

    if env_name == "LineWorld":
        action_symbols = ['→', '←']
        env = LineWorld()
        state_size = env.width() + 1
        Q = np.zeros((state_size, 2))

        for pos in range(env.width() + 1):
            state_index = pos
            state_one_hot = np.eye(state_size)[state_index:state_index + 1]
            if state_one_hot.shape[0] > 0 and state_one_hot.shape[1] > 0:
                Q[state_index] = model(state_one_hot, training=False).numpy()[0]

        for pos in range(env.width() + 1):
            state_index = pos
            best_action = np.argmax(Q[state_index])
            print(action_symbols[best_action], end=' ')
        print()

    elif env_name == "GridWorld":
        action_symbols = ['→', '←', '↑', '↓']
        env = GridWorld()
        state_size = env.width() * env.height()
        Q = np.zeros((state_size, len(env.available_actions())))

        for x in range(env.width()):
            for y in range(env.height()):
                state_index = x * env.width() + y
                state_one_hot = np.eye(state_size)[state_index:state_index + 1]
                if state_one_hot.shape[0] > 0 and state_one_hot.shape[1] > 0:
                    Q[state_index] = model(state_one_hot, training=False).numpy()[0]

        for x in range(env.width()):
            for y in range(env.height()):
                state_index = x * env.width() + y
                best_action = np.argmax(Q[state_index])
                print(action_symbols[best_action], end=' ')
            print()


if __name__ == "__main__":
    main()
