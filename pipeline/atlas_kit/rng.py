"""ak.Rng: mulberry32, bit-identical to site/src/lib/atlas/core/rng.ts for next(), int() and fork() (design §4.1).

    next()       a float in [0, 1)
    int(n)       floor(next() * n)
    fork(label)  Rng((seed ^ fnv1a32(label)) >>> 0), independent of how much was consumed; fnv1a32 runs over the
                 UTF-8 bytes of the label (the same as TextEncoder in TS; batch labels are ASCII: 'init', 'batch')
    normal()     Box-Muller with u1 = 1 - next(), u2 = next(), z = sqrt(-2 ln u1) * cos(2 pi u2), no caching.
                 NOT cross-engine exact (libm may differ in the last ulp): processes that draw normals set
                 RecordOpts.recordDraws, and Python references replay the recorded draws instead.
    get_state()  the 32-bit counter as an unsigned int; set_state(s) restores it (camelCase aliases match TS)

Test vectors (core.ts): mulberry32(42) -> 0.6011037519, 0.4482905590, 0.8524657935, 0.6697340414;
mulberry32(1) -> 0.6270739406, 0.0027357212, 0.5274470400.
"""
from __future__ import annotations

import math

_M32 = 0xFFFFFFFF


def _u32(x: int) -> int:
    return x & _M32


def _imul(a: int, b: int) -> int:
    """Math.imul as an unsigned 32-bit result (the bit pattern is what matters)."""
    return (a * b) & _M32


def fnv1a32(label: str) -> int:
    """32-bit FNV-1a over the UTF-8 bytes of `label` (offset basis 0x811c9dc5, prime 0x01000193)."""
    h = 0x811C9DC5
    for byte in label.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & _M32
    return h


def _seed_u32(seed) -> int:
    """JS `seed >>> 0`: truncate toward zero, then wrap to an unsigned 32-bit integer."""
    if isinstance(seed, float):
        if not math.isfinite(seed):
            return 0
        seed = int(seed)  # truncates toward zero, like ToUint32
    return int(seed) & _M32


class Rng:
    """mulberry32 with fork(); the Python twin of the TS Rng interface."""

    __slots__ = ("seed", "_a")

    def __init__(self, seed: int):
        self.seed = _seed_u32(seed)
        self._a = self.seed

    # ---- the cross-engine exact part
    def next(self) -> float:
        a = _u32(self._a + 0x6D2B79F5)
        self._a = a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = _u32(t + _imul(t ^ (t >> 7), 61 | t)) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0

    def int(self, n: int) -> int:
        return math.floor(self.next() * n)

    def fork(self, label: str) -> "Rng":
        return Rng(self.seed ^ fnv1a32(label))

    # ---- not cross-engine exact (see the module docstring)
    def normal(self) -> float:
        u1 = 1.0 - self.next()
        u2 = self.next()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ---- state
    def get_state(self) -> int:
        return self._a

    def set_state(self, s: int) -> None:
        self._a = _seed_u32(s)

    getState = get_state  # TS names
    setState = set_state

    def __repr__(self) -> str:
        return f"Rng(seed={self.seed}, state={self._a})"


def mulberry32(seed: int):
    """A plain next() closure, like the three legacy TS copies."""
    r = Rng(seed)
    return r.next
