"""snake-dqn: train the Q-network with DQN (Mnih et al. 2015): a replay buffer and a target network.

    npm run atlas -- py pipeline/snake-dqn/train.py                    # the page's network (runs/main/), GPU if any
    npm run atlas -- py pipeline/snake-dqn/train.py --variant hurry    # the same with -0.1 for every move (runs/hurry/)

ENVS games run side by side. Each move picks an action epsilon-greedily from the online network's Q-values (epsilon
falls linearly from 1 to EPS_END over the first EPS_FRAC of the moves). Every move (s, a, r, s', done) goes into a
replay buffer of BUFFER moves (the oldest are overwritten). Once WARMUP moves are stored, each round of ENVS moves is
followed by UPDATES Adam steps, each on BATCH moves drawn at random from the buffer:
    y = r + GAMMA * max_a' Q_target(s', a')      (y = r when the move ended the game: a crash or a full board)
    loss = mean over the batch of (y - Q(s, a))^2
The target network is a copy of the online network, refreshed every SYNC updates. A game that goes HUNGER moves without
food is stopped; that last move is not an ending for the target (the board itself is not over), so y still looks ahead.
Every EVAL_EVERY moves the greedy policy (epsilon 0) plays EVAL_SEEDS fixed seeds; the best checkpoint is kept (with the
target network of that moment). Every PROBE_EVERY updates both networks' Q-values on one fixed board are logged.
Writes runs/<variant>/{best.pt, log.json}.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import game as G  # noqa: E402
from model import QNet  # noqa: E402

GAMMA, BUFFER, BATCH, LR, SYNC, WARMUP = 0.95, 100_000, 256, 5e-4, 500, 20_000
ENVS, UPDATES, MOVES, EPS_END, EPS_FRAC = 64, 2, 1_500_000, 0.02, 0.4
EVAL_EVERY, EVAL_SEEDS, EVAL_CAP = 50_000, range(100_000, 100_200), 3_000
PROBE_EVERY, PROBE_SEED, LOG_EVERY = 25, 1, 10_000


def play_greedy(net, seeds, dev, rewards=G.REWARDS["main"], cap=EVAL_CAP, dtype=torch.float32):
    """The greedy policy on every seed, side by side. Returns the finished games (stopped after HUNGER moves without
    food or `cap` moves)."""
    games = [G.Game(s, rewards) for s in seeds]
    alive = list(range(len(games)))
    while alive:
        x = torch.from_numpy(np.stack([games[i].encode() for i in alive])).to(dev, dtype)
        with torch.no_grad():
            acts = net(x).argmax(1).cpu().numpy()
        nxt = []
        for i, act in zip(alive, acts):
            g = games[i]
            g.step(int(act))
            if not g.over and g.hunger < G.HUNGER and g.moves < cap:
                nxt.append(i)
        alive = nxt
    return games


def new_seed(rng) -> int:
    return int(rng.integers(0, 2**31 - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main", choices=sorted(G.REWARDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--moves", type=int, default=MOVES)
    args = ap.parse_args()
    rewards = G.REWARDS[args.variant]
    out = HERE / "runs" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    net = QNet().to(dev)
    tgt = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bs = torch.zeros((BUFFER, 3, G.SIZE, G.SIZE), device=dev)
    bs2 = torch.zeros_like(bs)
    ba = torch.zeros(BUFFER, dtype=torch.long, device=dev)
    br = torch.zeros(BUFFER, device=dev)
    bd = torch.zeros(BUFFER, device=dev)
    probe = torch.from_numpy(G.Game(PROBE_SEED).encode()[None]).to(dev)
    envs = [G.Game(new_seed(rng), rewards) for _ in range(ENVS)]
    states = np.stack([g.encode() for g in envs])
    moves = updates = ptr = size = 0
    log, probes, evals = [], [], []
    ep_scores, ep_ends = [], {"wall": 0, "self": 0, "full": 0, "hungry": 0}
    loss_sum, loss_n = torch.zeros((), device=dev), 0
    best, t0 = -1.0, time.time()
    while moves < args.moves:
        eps = max(EPS_END, 1.0 - (1.0 - EPS_END) * moves / (EPS_FRAC * args.moves))
        with torch.no_grad():
            greedy = net(torch.from_numpy(states).to(dev)).argmax(1).cpu().numpy()
        explore = rng.random(ENVS) < eps
        acts = np.where(explore, rng.integers(0, 3, ENVS), greedy)
        nxt = np.empty_like(states)
        rs = np.zeros(ENVS, np.float32)
        ds = np.zeros(ENVS, np.float32)
        for i, g in enumerate(envs):
            rs[i] = g.step(int(acts[i]))
            ds[i] = 1.0 if g.over else 0.0
            nxt[i] = g.encode()
            if g.over or g.hunger >= G.HUNGER:
                ep_scores.append(g.score)
                ep_ends[g.over or "hungry"] += 1
                envs[i] = G.Game(new_seed(rng), rewards)
        idx = torch.from_numpy((ptr + np.arange(ENVS)) % BUFFER).to(dev)
        bs[idx] = torch.from_numpy(states).to(dev)
        bs2[idx] = torch.from_numpy(nxt).to(dev)
        ba[idx] = torch.from_numpy(acts.astype(np.int64)).to(dev)
        br[idx] = torch.from_numpy(rs).to(dev)
        bd[idx] = torch.from_numpy(ds).to(dev)
        ptr = (ptr + ENVS) % BUFFER
        size = min(size + ENVS, BUFFER)
        states = np.stack([g.encode() for g in envs])
        prev = moves
        moves += ENVS
        if size >= WARMUP:
            for _ in range(UPDATES):
                j = torch.randint(0, size, (BATCH,), device=dev)
                with torch.no_grad():
                    y = br[j] + GAMMA * (1.0 - bd[j]) * tgt(bs2[j]).max(1).values
                q = net(bs[j]).gather(1, ba[j][:, None]).squeeze(1)
                loss = F.mse_loss(q, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                loss_sum += loss.detach()
                loss_n += 1
                updates += 1
                if updates % SYNC == 0:
                    tgt.load_state_dict(net.state_dict())
                if updates % PROBE_EVERY == 0:
                    with torch.no_grad():
                        probes.append([updates] + net(probe)[0].tolist() + tgt(probe)[0].tolist())
        if moves // LOG_EVERY != prev // LOG_EVERY:
            log.append({"moves": moves, "updates": updates, "eps": round(eps, 4),
                        "score": float(np.mean(ep_scores)) if ep_scores else None, "episodes": len(ep_scores),
                        "loss": float(loss_sum / loss_n) if loss_n else None, "ends": dict(ep_ends)})
            ep_scores, loss_sum, loss_n = [], torch.zeros((), device=dev), 0
            ep_ends = {k: 0 for k in ep_ends}
        if moves // EVAL_EVERY != prev // EVAL_EVERY:
            net.eval()
            games = play_greedy(net, EVAL_SEEDS, dev, rewards)
            net.train()
            sc = float(np.mean([g.score for g in games]))
            evals.append({"moves": moves, "updates": updates, "score": sc, "max": max(g.score for g in games)})
            print(f"{args.variant} moves {moves:>8} updates {updates:>7} eps {eps:.3f} greedy mean {sc:6.2f} "
                  f"max {evals[-1]['max']:3d} train {log[-1]['score'] if log else None} ({time.time() - t0:.0f} s)", flush=True)
            if sc > best:
                best = sc
                torch.save({"online": {k: v.cpu() for k, v in net.state_dict().items()},
                            "target": {k: v.cpu() for k, v in tgt.state_dict().items()},
                            "moves": moves, "updates": updates, "sinceSync": updates % SYNC, "eval": sc}, out / "best.pt")
            (out / "log.json").write_text(json.dumps({
                "variant": args.variant, "rewards": rewards, "seed": args.seed, "gamma": GAMMA, "buffer": BUFFER,
                "batch": BATCH, "lr": LR, "sync": SYNC, "warmup": WARMUP, "envs": ENVS, "updatesPerRound": UPDATES,
                "moves": args.moves, "epsEnd": EPS_END, "epsFrac": EPS_FRAC, "hunger": G.HUNGER,
                "evalSeeds": [EVAL_SEEDS.start, EVAL_SEEDS.stop], "probeSeed": PROBE_SEED, "best": best, "log": log,
                "evals": evals, "probes": probes, "seconds": round(time.time() - t0)}), encoding="utf-8")
    print("done", args.variant, "best", best, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
