"""snake-dqn: the snake game, deterministic for a seed. site/src/lib/explainers/snake-dqn/rules.ts is the same game.

The board has SIZE x SIZE cells, numbered row by row from the top left (cell = y * SIZE + x). The snake starts with
length 3 in the middle: head on cell 40 (row 4, column 4), heading up, body on cells 49 and 58.
Actions are relative to the heading: 0 straight on, 1 turn left, 2 turn right. Headings: 0 up, 1 right, 2 down, 3 left.
One move: the head moves one cell.
  - off the board, or onto the body, is a crash: reward R_CRASH, game over. The tail cell is free, because the tail
    moves on in the same move (unless the snake eats, but the food is never on the snake);
  - onto the food: reward R_FOOD, the snake grows by one, and the next food appears (no empty cell left: the board is
    full, game over);
  - otherwise reward R_STEP (0 on the page; train.py --variant hurry uses -0.1).
Food: the k-th food (k = 0, 1, ...) goes to empty cell number Rng(seed).fork(f"food{k}").int(number of empty cells),
counting the empty cells row by row. ak.Rng is bit-identical to the TS rng, so a seed gives the same game in both.
The game itself has no move limit; the code that plays it (train.py, evaluate.py, the page) stops a game after
HUNGER moves without food.

The network's input (encode): 3 planes of SIZE x SIZE numbers, float32:
  0 head   1 on the head's cell
  1 body   on the i-th body cell behind the head (i = 1 .. L-1, L = length): (L - i) / L, so the cell right behind the
           head is brightest and the tail, the next cell to become free, is dimmest
  2 food   1 on the food's cell
"""
from __future__ import annotations

import numpy as np

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

SIZE = 9
CELLS = SIZE * SIZE
DY = (-1, 0, 1, 0)
DX = (0, 1, 0, -1)
TURN = (0, 3, 1)  # straight on, left (counterclockwise), right (clockwise)
START = (40, 49, 58)
START_DIR = 0
HUNGER = 100
REWARDS = {"main": (1.0, -1.0, 0.0), "hurry": (1.0, -1.0, -0.1)}
ACTIONS = ("straight on", "turn left", "turn right")


class Game:
    __slots__ = ("seed", "snake", "dir", "food", "score", "moves", "hunger", "over", "rew")

    def __init__(self, seed: int, rewards=REWARDS["main"]):
        self.seed = int(seed)
        self.snake = list(START)
        self.dir = START_DIR
        self.score = 0
        self.moves = 0
        self.hunger = 0
        self.over = ""  # '', 'wall', 'self', 'full'
        self.rew = rewards
        self.food = -1
        self._place()

    def copy(self) -> "Game":
        g = Game.__new__(Game)
        g.seed, g.snake, g.dir, g.food, g.score = self.seed, list(self.snake), self.dir, self.food, self.score
        g.moves, g.hunger, g.over, g.rew = self.moves, self.hunger, self.over, self.rew
        return g

    def _place(self) -> None:
        taken = set(self.snake)
        empty = [c for c in range(CELLS) if c not in taken]
        if not empty:
            self.food = -1
            self.over = "full"
            return
        self.food = empty[ak.Rng(self.seed).fork(f"food{self.score}").int(len(empty))]

    def next_cell(self, a: int) -> int:
        """The cell the head moves to with action a (-1: off the board)."""
        d = (self.dir + TURN[a]) % 4
        y, x = divmod(self.snake[0], SIZE)
        ny, nx = y + DY[d], x + DX[d]
        return ny * SIZE + nx if 0 <= ny < SIZE and 0 <= nx < SIZE else -1

    def safe(self, a: int) -> bool:
        c = self.next_cell(a)
        return c >= 0 and (c == self.food or c not in self.snake[:-1])

    def step(self, a: int) -> float:
        assert not self.over, "the game is over"
        c = self.next_cell(a)
        self.dir = (self.dir + TURN[a]) % 4
        self.moves += 1
        self.hunger += 1
        if c < 0:
            self.over = "wall"
            return self.rew[1]
        eat = c == self.food
        if not eat and c in self.snake[:-1]:
            self.over = "self"
            return self.rew[1]
        self.snake.insert(0, c)
        if eat:
            self.score += 1
            self.hunger = 0
            self._place()
            return self.rew[0]
        self.snake.pop()
        return self.rew[2]

    def encode(self) -> np.ndarray:
        x = np.zeros((3, SIZE, SIZE), dtype=np.float32)
        n = len(self.snake)
        x[0].flat[self.snake[0]] = 1.0
        for i in range(1, n):
            x[1].flat[self.snake[i]] = np.float32((n - i) / n)
        if self.food >= 0:
            x[2].flat[self.food] = 1.0
        return x


def replay(seed: int, moves) -> Game:
    """The game after a list of moves (moves after the game ended are ignored)."""
    g = Game(seed)
    for a in moves:
        if g.over:
            break
        g.step(int(a))
    return g


if __name__ == "__main__":
    g = Game(1)
    print("food", g.food, "snake", g.snake)
    rng = np.random.default_rng(0)
    n = 0
    while not g.over and g.hunger < HUNGER:
        safe = [a for a in range(3) if g.safe(a)] or [0]
        g.step(int(rng.choice(safe)))
        n += 1
    print("random safe player:", n, "moves, score", g.score, "end", g.over or "hungry")
