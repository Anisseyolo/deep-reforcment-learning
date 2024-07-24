import numpy as np

class LineWorld:
    def __init__(self):
        self.state = 0
        self.done = False
        self.max_position = 10

    def reset(self):
        self.state = 0
        self.done = False
        return self.state

    def width(self):
        return self.max_position

    def height(self):
        return 1

    def available_actions(self):
        return [0, 1]

    def step(self, action):
        if action == 0:
            self.state += 1
        elif action == 1:
            self.state -= 1

        self.state = max(0, min(self.state, self.max_position))

        reward = 0
        if self.state == self.max_position:
            reward = 1
            self.done = True
        elif self.state == 0:
            reward = -1

        if self.state == self.max_position or self.state == 0:
            self.done = True

        return (self.state, reward, self.done)

    def state_id(self):
        return self.state

    def score(self):
        return self.state

    def from_random_state(self):
        self.state = np.random.randint(0, self.max_position + 1)
        self.done = False
        return self

    def is_game_over(self):
        return self.done

    def display(self):
        """Affiche l'état actuel de l'environnement."""
        line = ['-' for _ in range(self.max_position + 1)]
        line[self.state] = '*'