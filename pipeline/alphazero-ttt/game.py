"""alphazero-ttt: tic-tac-toe rules, the board as the network reads it, and an exact solver.

A board is a tuple of 9 ints, cells 0..8 row by row: 1 = X, -1 = O, 0 = empty. X moves first, so the player to move is
X when the number of marks is even. The network reads the board from the side of the player to move (`view`):
+1 for the mover's marks, -1 for the opponent's, 0 for empty; the mask is 0 for an empty cell and -inf for a taken one.
"""
from __future__ import annotations

from functools import lru_cache

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
EMPTY = (0,) * 9


def board_of(moves) -> tuple:
    b = [0] * 9
    for i, c in enumerate(moves):
        b[c] = 1 if i % 2 == 0 else -1
    return tuple(b)


def to_move(b) -> int:
    """1 (X) or -1 (O)."""
    return 1 if sum(1 for v in b if v != 0) % 2 == 0 else -1


def winner(b) -> int:
    """1 or -1 when that player has three in a row, else 0."""
    for x, y, z in LINES:
        if b[x] != 0 and b[x] == b[y] == b[z]:
            return b[x]
    return 0


def terminal(b) -> bool:
    return winner(b) != 0 or all(v != 0 for v in b)


def legal(b) -> list:
    return [i for i in range(9) if b[i] == 0]


def play(b, cell) -> tuple:
    if b[cell] != 0:
        raise ValueError(f"cell {cell} is taken")
    nb = list(b)
    nb[cell] = to_move(b)
    return tuple(nb)


def result_for_mover(b) -> float:
    """At a finished board: the result for the player to move there (-1 lost, 0 draw)."""
    w = winner(b)
    return 0.0 if w == 0 else (1.0 if w == to_move(b) else -1.0)


def view(b) -> list:
    s = to_move(b)
    return [float(v * s) + 0.0 for v in b]


def mask(b) -> list:
    """0 for an empty cell, -inf for a taken one; all 0 on a full board (the numbers stay finite), as the page's adapter."""
    full = all(v != 0 for v in b)
    return [0.0 if v == 0 or full else float("-inf") for v in b]


@lru_cache(maxsize=None)
def solve(b) -> int:
    """The game's value for the player to move with perfect play from both sides: 1 win, 0 draw, -1 loss."""
    if terminal(b):
        return int(result_for_mover(b))
    return max(-solve(play(b, c)) for c in legal(b))


def move_values(b) -> dict:
    """The value of each legal move for the player making it (perfect play afterwards)."""
    return {c: -solve(play(b, c)) for c in legal(b)}


def best_moves(b) -> list:
    mv = move_values(b)
    top = max(mv.values())
    return [c for c, v in mv.items() if v == top]


def all_positions() -> list:
    """Every board reachable from the empty board (terminal ones included), in a fixed order."""
    seen, order, stack = {EMPTY}, [EMPTY], [EMPTY]
    while stack:
        b = stack.pop()
        if terminal(b):
            continue
        for c in legal(b):
            nb = play(b, c)
            if nb not in seen:
                seen.add(nb)
                order.append(nb)
                stack.append(nb)
    return sorted(order, key=lambda x: (sum(1 for v in x if v != 0), x))


def count_games(b=EMPTY) -> int:
    """Lines of play (complete games) from b."""
    if terminal(b):
        return 1
    return sum(count_games(play(b, c)) for c in legal(b))


def count_nodes(b=EMPTY) -> int:
    """Positions in the full game tree from b (every line counted separately, b included)."""
    if terminal(b):
        return 1
    return 1 + sum(count_nodes(play(b, c)) for c in legal(b))


SYMS = []  # the 8 symmetries of the square as cell permutations: new[i] = old[perm[i]]
for _k in range(4):
    for _f in (False, True):
        perm = []
        for i in range(9):
            r, c = divmod(i, 3)
            if _f:
                c = 2 - c
            for _ in range(_k):
                r, c = c, 2 - r
            perm.append(r * 3 + c)
        SYMS.append(tuple(perm))

if __name__ == "__main__":
    ps = all_positions()
    nt = [p for p in ps if not terminal(p)]
    print("positions", len(ps), "non-terminal", len(nt), "games", count_games(), "tree nodes", count_nodes(), "value", solve(EMPTY))
    print("syms distinct", len(set(SYMS)))
