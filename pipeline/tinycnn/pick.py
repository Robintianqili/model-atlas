import numpy as np, itertools, random
from screen_cands import CANDS
L = np.load("screen_logits.npy"); y = np.load("screen_y.npy")
def score(S):
    S = list(S); idx = np.isin(y, S)
    sub = L[idx][:, S]; yy = np.searchsorted(np.array(sorted(S)), y[idx])
    order = np.argsort(S); sub = sub[:, order]
    acc = (sub.argmax(1) == yy).reshape(len(S), -1).mean(1)
    return acc.mean(), acc.min()
name = {c: i for i, c in enumerate(CANDS)}
def show(lbl, S):
    m, mn = score([name[c] for c in S]); print(f"{lbl}: mean={m:.4f} min={mn:.4f} {S}")
show("favorites", ["cat","fish","house","tree","car","star","umbrella","clock","sun","apple"])
show("fav2", ["cat","fish","house","tree","car","star","umbrella","clock","bicycle","apple"])
show("fav3", ["fish","house","tree","car","star","umbrella","clock","bicycle","sun","apple"])
excl = {name["ladder"], name["envelope"]}
pool = [i for i in range(len(CANDS)) if i not in excl]
random.seed(0); best = None
for r in range(30):
    S = random.sample(pool, 10)
    cur = score(S)[0]
    improved = True
    while improved:
        improved = False
        for i in range(10):
            for c in pool:
                if c in S: continue
                T = S.copy(); T[i] = c; s = score(T)[0]
                if s > cur + 1e-9: S, cur, improved = T, s, True
    if best is None or cur > best[0]: best = (cur, sorted(S))
print("best", best[0], [CANDS[i] for i in best[1]], score(best[1]))
# held-out pairs (2-way restricted accuracy)
pairs = []
for a, b in itertools.combinations(range(len(CANDS)), 2):
    pairs.append((score([a, b])[0], CANDS[a], CANDS[b]))
pairs.sort()
print("hardest pairs", pairs[:8]); print("ladder/envelope", score([name["ladder"], name["envelope"]]))
