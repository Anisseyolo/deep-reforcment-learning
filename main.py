import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from environments.line_world import LineWorld
from environments.grid_world import GridWorld
from environments.rock_paper_scissors import RockPaperScissors
from models.off_policy_mc_control import off_policy_monte_carlo_control
from models.on_policy_mc_control import on_policy_first_visit_monte_carlo_control
from secretEnvironments.secret_envs_wrapper import SecretEnv0
from models.monte_carlo_es import naive_monte_carlo_with_exploring_starts
from models.policy_iteration import policy_iteration
from models.value_iteration import value_iteration
from models.sarsa import episodic_semi_gradient_sarsa
from models.q_learning import q_learning
from models.dyna_q import dyna_q

def save_model(policy, filename):
    os.makedirs('modelSave', exist_ok=True)
    filepath = os.path.join('modelSave', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(policy, f)
    print(f"Model saved to {filepath}")

def load_model(filename):
    filepath = os.path.join('modelSave', filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def evaluate_loaded_model(env_class, loaded_policy, num_episodes, env_name):
    """Évalue le modèle chargé avec l'environnement spécifié et le nombre d'épisodes."""
    for episode in range(num_episodes):
        env = env_class()
        env.reset()
        total_steps = 0
        done = False

        if env_name == "RockPaperScissors":
            print("Testing RockPaperScissors. Enter your action for each step (0: Rock, 1: Paper, 2: Scissors)")

        while not done:
            state = env.state_id()
            state_one_hot = np.eye(env.width() + 1)[state:state + 1]

            if env_name == "RockPaperScissors":
                action = int(input(f"Step {total_steps + 1}, Enter your action: "))
            else:
                # Utiliser le modèle pour prédire l'action
                q_values = loaded_policy(state_one_hot, training=False).numpy()[0]
                action = np.argmax(q_values)

            state, reward, done = env.step(action)
            total_steps += 1

            # Affichage des étapes effectuées
            print(f"Step: {total_steps}")

        # Affichage des résultats de l'épisode
        print("Game Over! Final score:", env.score())
        print(f"Total number of steps taken to finish the game: {total_steps}")


def print_menu():
    print("Select an Algorithm:")
    print("1. Monte Carlo ES")
    print("2. Policy Iteration")
    print("3. Value Iteration")
    print("4. Sarsa")
    print("5. Q-Learning")
    print("6. Dyna-Q")
    print("7. On-Policy First Visit Monte Carlo Control")
    print("8. Off-Policy Monte Carlo Control")
    print()
    print("Select an Environment:")
    print("1. LineWorld")
    print("2. GridWorld")
    print("3. RockPaperScissors")
    print("4. SecretEnv0")

def train_model(env_class, algorithm, num_episodes, env_name):
    total_steps_list = []
    rewards_list = []

    user_action = None
    if env_name == "RockPaperScissors":
        user_action = int(input("Enter your action (0, 1, or 2): "))

    for _ in range(num_episodes):
        env = env_class()

        # Obtenir uniquement la politique
        policy = algorithm(env_class)

        env.reset()
        total_steps = 0
        done = False

        while not done:
            if env_name == "RockPaperScissors":
                action = user_action
            else:
                s = env.state_id()
                action = policy[s]

            state, reward, done = env.step(action)
            total_steps += 1

        total_steps_list.append(total_steps)
        rewards_list.append(env.score())

    # Afficher le graphique
    plot_results(total_steps_list, rewards_list)

    return total_steps_list, rewards_list

def plot_results(total_steps_list, rewards_list):
    episodes = list(range(1, len(total_steps_list) + 1))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, total_steps_list, marker='o', linestyle='-', color='b')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Steps')
    plt.title('Total Steps vs Episode Number')
    plt.grid(True)
    plt.xticks(episodes)

    plt.subplot(1, 2, 2)
    plt.plot(episodes, rewards_list, marker='o', linestyle='-', color='r')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episode Number')
    plt.grid(True)
    plt.xticks(episodes)

    plt.tight_layout()
    plt.show()

def main():
    choice = input("Do you want to train a new model or load an existing one? (train/load): ").strip().lower()
    print_menu()
    if choice == "train":
        algo_choice = int(input("Enter the number for the algorithm: "))
        env_choice = int(input("Enter the number for the environment: "))
        num_episodes = int(input("Enter the number of times to run the environment: "))
        model_filename = input("Enter the filename to save the model (leave empty if you don't want to save): ").strip()

        env_class = None
        env_name = ""
        if env_choice == 1:
            env_class = LineWorld
            env_name = "LineWorld"
        elif env_choice == 2:
            env_class = GridWorld
            env_name = "GridWorld"
        elif env_choice == 3:
            env_class = RockPaperScissors
            env_name = "RockPaperScissors"
        elif env_choice == 4:
            env_class = SecretEnv0
            env_name = "SecretEnv0"
        else:
            print("Invalid environment choice!")
            return

        algorithm = None
        if algo_choice == 1:
            algorithm = lambda env_class: naive_monte_carlo_with_exploring_starts(env_class)
        elif algo_choice == 2:
            algorithm = lambda env_class: policy_iteration(env_class)
        elif algo_choice == 3:
            algorithm = lambda env_class: value_iteration(env_class)
        elif algo_choice == 4:
            algorithm = lambda env_class: episodic_semi_gradient_sarsa(env_class)
        elif algo_choice == 5:
            algorithm = lambda env_class: q_learning(env_class)
        elif algo_choice == 6:
            algorithm = lambda env_class: dyna_q(env_class)
        elif algo_choice == 7:
            algorithm = lambda env_class: on_policy_first_visit_monte_carlo_control(env_class)
        elif algo_choice == 8:
            algorithm = lambda env_class: off_policy_monte_carlo_control(env_class)
        else:
            print("Invalid algorithm choice!")
            return

        total_steps_list, rewards_list = train_model(env_class, algorithm, num_episodes, env_name)
        plot_results(total_steps_list, rewards_list)

        if model_filename:
            save_model(algorithm(env_class), model_filename)

    elif choice == "load":
        model_filename = input("Enter the filename of the model to load: ").strip()
        loaded_policy = load_model(model_filename)
        env_choice = int(input("Enter the number for the environment: "))
        num_episodes = int(input("Enter the number of times to run the environment: "))

        env_class = None
        env_name = ""
        if env_choice == 1:
            env_class = LineWorld
            env_name = "LineWorld"
        elif env_choice == 2:
            env_class = GridWorld
            env_name = "GridWorld"
        elif env_choice == 3:
            env_class = RockPaperScissors
            env_name = "RockPaperScissors"
        elif env_choice == 4:
            env_class = SecretEnv0
            env_name = "SecretEnv0"
        else:
            print("Invalid environment choice!")
            return

        evaluate_loaded_model(env_class, loaded_policy, num_episodes, env_name)

    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
