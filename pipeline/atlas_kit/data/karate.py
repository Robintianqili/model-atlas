"""ak.data.karate(): Zachary's karate club from networkx (no download), as an UNWEIGHTED graph.

networkx's karate_club_graph() carries `weight` attributes on its edges (interaction counts); the GCN page uses the
plain 0/1 adjacency, so A = nx.to_numpy_array(g, weight=None). Labels come from each node's 'club' attribute.
Cite: W. W. Zachary, "An information flow model for conflict and fission in small groups",
Journal of Anthropological Research 33(4), 452-473 (1977).
"""
from __future__ import annotations

from ..errors import fail

CITATION = ('W. W. Zachary, "An information flow model for conflict and fission in small groups", '
            "Journal of Anthropological Research 33(4), 452-473 (1977)")
CLUBS = ("Mr. Hi", "Officer")


def karate() -> dict:
    """{A: float32 [34, 34] 0/1 symmetric, no self-loops; labels: int64 [34] (0 = Mr. Hi, 1 = Officer);
    clubs: ('Mr. Hi', 'Officer'); edges: [(i, j)] with i < j; n: 34; citation}."""
    try:
        import networkx as nx
        import numpy as np
    except ImportError as e:
        fail("AK002", f"ak.data.karate needs networkx and numpy ({e})", fix="pip install -r pipeline/requirements.txt")
    g = nx.karate_club_graph()
    nodes = sorted(g.nodes())
    A = nx.to_numpy_array(g, nodelist=nodes, weight=None).astype(np.float32)
    labels = np.array([CLUBS.index(g.nodes[v]["club"]) for v in nodes], dtype=np.int64)
    edges = sorted((min(u, v), max(u, v)) for u, v in g.edges())
    assert A.shape == (34, 34) and (A == A.T).all() and set(np.unique(A)) <= {0.0, 1.0} and not A.diagonal().any()
    return {"A": A, "labels": labels, "clubs": CLUBS, "edges": edges, "n": len(nodes), "citation": CITATION}
