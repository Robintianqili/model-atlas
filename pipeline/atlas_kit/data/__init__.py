"""Data sets that need no new download (design §7): Quick, Draw! and TinyStories (already in pipeline/data/, which is
git-ignored and lives in the main tree only), and networkx's built-in karate club. Synthetic data is generated in TS
(site/src/lib/atlas/data/*.ts, the source of truth) and handed to Python by `atlas run <slug> <source> --save`.

These helpers never download anything.
"""
from .karate import karate
from .quickdraw import quickdraw
from .tinystories import tinystories_chars

__all__ = ["karate", "quickdraw", "tinystories_chars"]
