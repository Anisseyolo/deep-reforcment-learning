import numpy as np
from typing import List, Tuple


class GridWorld:
    def __init__(self, width: int = 5, height: int = 5, start_pos: Tuple[int, int] = (0, 0),
                 goal_pos: Tuple[int, int] = (4, 4)):
        self._width = width
        self._height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.agent_pos = start_pos
        self.max_position = width * height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def from_random_state(self) -> 'GridWorld':
        env = GridWorld(self._width, self._height, self.start_pos, self.goal_pos)
        env.agent_pos = (np.random.randint(self._width), np.random.randint(self._height))
        return env

    def available_actions(self) -> List[int]:
        return [0, 1, 2, 3]

    def is_game_over(self) -> bool:
        return self.agent_pos == self.goal_pos

    def state_id(self) -> int:
        return self.agent_pos[0] * self._width + self.agent_pos[1]

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        if self.is_game_over():
            raise Exception("Game is over, can't take more actions.")

        x, y = self.agent_pos
        if action == 0:
            y = min(self._height - 1, y + 1)
        elif action == 1:
            x = min(self._width - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            x = max(0, x - 1)

        self.agent_pos = (x, y)

        reward = self.score()
        done = self.is_game_over()
        return self.agent_pos, reward, done

    def score(self) -> float:
        return 1.0 if self.agent_pos == self.goal_pos else 0.0

    def display(self):
        for y in range(self._height - 1, -1, -1):
            for x in range(self._width):
                if (x, y) == self.agent_pos:
                    print('A', end=' ')
                elif (x, y) == self.goal_pos:
                    print('G', end=' ')
                else:
                    print('.', end=' ')
            print()

    def reset(self):
        self.agent_pos = self.start_pos

    def num_states(self):
        return self.max_position
