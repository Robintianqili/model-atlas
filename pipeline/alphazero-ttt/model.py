"""alphazero-ttt: the policy/value network (kind A).

It reads the board from the side of the player to move (9 numbers: +1 mine, -1 theirs, 0 empty) and a mask (0 for an
empty cell, -inf for a taken one), and gives two answers:
  prior = softmax(pol(h) + mask)   a chance for each of the 9 cells; taken cells get exactly 0
  value = tanh(val(h))             how the game will end for the player to move, from -1 (loss) to +1 (win)
with h = relu(fc2(relu(fc1(board)))). 9 -> 96 -> 96 -> (9, 1): 11,242 parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN = 96


class PVNet(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(9, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pol = nn.Linear(hidden, 9)
        self.val = nn.Linear(hidden, 1)

    def trunk(self, board: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(board))))

    def forward(self, board: torch.Tensor, mask: torch.Tensor):
        h = self.trunk(board)
        prior = torch.softmax(self.pol(h) + mask, dim=-1)
        value = torch.tanh(self.val(h))
        return prior, value

    def logits(self, board: torch.Tensor, mask: torch.Tensor):
        """Training: masked log-priors and the value."""
        h = self.trunk(board)
        return torch.log_softmax(self.pol(h) + mask, dim=-1), torch.tanh(self.val(h))


class Packed(nn.Module):
    """export_model calls module(x) with one argument: x = (board, mask)."""

    def __init__(self, net: PVNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x[0], x[1])


if __name__ == "__main__":
    m = PVNet()
    print(m, sum(p.numel() for p in m.parameters()), "params")
