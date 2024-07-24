import matplotlib.pyplot as plt
import numpy as np


def generate_data(num_episodes, num_algorithms, num_environments):
    return np.random.randn(num_algorithms, num_environments, num_episodes).cumsum(axis=2)


def plot_environment_performance(environment_name, data, algorithms, num_episodes):
    num_algorithms = data.shape[0]
    num_cols = 2
    num_rows = (num_algorithms + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6 * num_rows), constrained_layout=True)

    if num_algorithms <= num_cols:
        axes = np.array(axes).reshape(-1)
    else:
        axes = np.array(axes).reshape(-1)

    for i in range(num_algorithms):
        ax = axes[i]
        ax.plot(data[i], label=algorithms[i])
        ax.set_xlabel('Nombre d\'épisodes')
        ax.set_ylabel('Récompense cumulée')
        ax.set_title(f'{algorithms[i]}', fontsize=8)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)

    for i in range(num_algorithms, len(axes)):
        axes[i].axis('off')

    # Titre principal
    fig.suptitle(f'Performance des algorithmes dans l\'environnement {environment_name}', fontsize=14)

    plt.show()


def main():
    num_episodes = 1000
    num_algorithms = 10
    num_environments = 5

    algorithms = [
        "Dynamic Programming - Policy Iteration",
        "Dynamic Programming - Value Iteration",
        "Monte Carlo ES",
        "On-policy First Visit Monte Carlo Control",
        "Off-policy Monte Carlo Control",
        "Sarsa",
        "Q-Learning",
        "Expected Sarsa",
        "Dyna-Q",
        "Dyna-Q+"
    ]

    environments = [
        "Line World",
        "Grid World",
        "Rock Paper Scissors",
        "Monty Hall Level 1",
        "Monty Hall Level 2"
    ]

    data = generate_data(num_episodes, num_algorithms, num_environments)

    for i, environment in enumerate(environments):
        plot_environment_performance(environment, data[:, i, :], algorithms, num_episodes)


if __name__ == "__main__":
    main()
