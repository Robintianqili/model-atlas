"""snake-dqn: the Q-network (kind A).

It reads the board as 3 planes of 9 x 9 numbers (game.py encode: head, body, food) and gives one number per action,
Q(s, a) for a = straight on, turn left, turn right: how much reward the snake can still expect if it takes that action
now and plays well afterwards (rewards later on count less, by gamma per move).
  conv1  3 -> 16 channels, 3 x 3, padding 1, ReLU
  conv2  16 -> 32 channels, 3 x 3, padding 1, ReLU
  fc1    32 * 9 * 9 = 2,592 -> 64, ReLU
  out    64 -> 3
171,235 parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, c1: int = 16, c2: int = 32, hidden: int = 64, size: int = 9):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc1 = nn.Linear(c2 * size * size, hidden)
        self.out = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.fc1(h.flatten(1)))
        return self.out(h)


if __name__ == "__main__":
    m = QNet()
    print(m, sum(p.numel() for p in m.parameters()), "params")
