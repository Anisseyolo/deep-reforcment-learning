import numpy as np

class RockPaperScissors:
    def __init__(self):
        self.state = 0
        self.reward = 0
        self.reset()
        self.round = 10
        self.actions = ['Pierre', 'Feuille', 'Ciseaux']
        self.initial_action = None

    def reset(self):
        self.done = False
        self.current_round = 0
        self.state = None
        self.reward = 0
        self.history = []
        self.initial_action = None
        return self.state_desc()

    def state_desc(self):
        return np.array([])

    def step(self, action):
        if self.initial_action is None:
            self.initial_action = action

        opponent_action = np.random.choice([0, 1, 2])

        if (self.initial_action == 0 and opponent_action == 2) or \
           (self.initial_action == 1 and opponent_action == 0) or \
           (self.initial_action == 2 and opponent_action == 1):
            reward = 1
        elif self.initial_action == opponent_action:
            reward = 0
        else:
            reward = -1

        self.reward += reward
        self.current_round += 1
        self.history.append((self.current_round, self.initial_action, opponent_action, reward))

        if self.current_round >= self.round:
            self.done = True

        return self.state_desc(), reward, self.done

    def available_actions(self):
        return [0, 1, 2]

    def num_actions(self):
        return 3

    def from_random_state(self):
        self.state = None
        self.done = False
        self.reward = 0
        self.current_round = 0
        self.history = []
        return self

    def is_game_over(self):
        return self.done

    def state_id(self):
        return self.state

    def score(self):
        return self.reward

    def num_states(self):
        return self.round

    def display(self):
        #print(f"Round {self.current_round}/{self.round}")
        for round_num, action, opponent_action, reward in self.history:
            action_name = self.actions[action]
            opponent_action_name = self.actions[opponent_action]
            result = "Win" if reward == 1 else "Draw" if reward == 0 else "Lose"
            #print(f"Round {round_num}: You chose {action_name}, Opponent chose {opponent_action_name}. Result: {result}")
        #print(f"Total Score: {self.score()}")
