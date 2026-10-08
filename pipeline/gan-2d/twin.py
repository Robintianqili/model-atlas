"""gan-2d: the PyTorch float64 twin of trainer 'gan' (kind B parity, design §4.11).

`npm run atlas -- parity gan-2d` runs the TS trainer (site/src/lib/explainers/gan-2d/trainer.ts) in fine detail on the
page's start data, writes its initial weights and batches to site/.atlas/gan-2d/train_fixture.gan.<optimizer>.json, then
runs `python -m atlas_kit.train_reference` with this twin: losses, step-1 gradients and final params must agree over 50
steps (plain SGD <= 1e-10 relative, Adam <= 1e-7 relative).

The model is two MLPs with the TS state_dict names: gen = Sequential(Linear(2, h), Tanh, Linear(h, h), Tanh,
Linear(h, 2)) and disc = Sequential(Linear(2, h), Tanh, Linear(h, h), Tanh, Linear(h, 1)).
Each training step updates ONE of them; the batch says whose turn it is (turn 0: D, turn 1: G):
    D step: loss_D = BCE(D(x), 1) + BCE(D(G(z).detach()), 0)
    G step: loss_G = BCE(D(G(z)), 1)            (= -mean log D(G(z)), the non-saturating generator loss)
BCE is written in the TS form max(l, 0) - l*y + log(1 + exp(-|l|)) (atlas/autograd/loss.ts bceWithLogits), mean over
the batch. Two optimizers (one per net, D at cfg lr, G at cfg lrG); step() steps the one whose turn it is.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i + 2 < len(sizes):
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Gan(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.gen = mlp([2, hidden, hidden, 2])
        self.disc = mlp([2, hidden, hidden, 1])
        self.turn = 0


def build(cfg: dict) -> nn.Module:
    """The TS model for this run's options: hidden width (default 16)."""
    return Gan(int(cfg.get("hidden", 16))).double()


def bce(logits: torch.Tensor, y: float) -> torch.Tensor:
    return (torch.relu(logits) - logits * y + torch.log(torch.exp(-logits.abs()) + 1)).mean()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'x': [n, 2] real points, 'z': [B, 2] noise, 'turn': [1]} float64."""
    turn = int(batch["turn"].reshape(-1)[0].item())
    model.turn = turn
    x, z = batch["x"], batch["z"]
    if x.shape[0] == 0:
        w = (model.gen if turn else model.disc)[0].weight
        return w.sum() * 0, {}
    if turn == 0:
        fake = model.gen(z).detach()
        return bce(model.disc(x), 1.0) + bce(model.disc(fake), 0.0), {}
    return bce(model.disc(model.gen(z)), 1.0), {}


class Alternating:
    """Two optimizers; step() updates the net whose turn the last loss() call set."""

    def __init__(self, model: Gan, d: torch.optim.Optimizer, g: torch.optim.Optimizer):
        self.model, self.d, self.g = model, d, g

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.d.zero_grad(set_to_none=set_to_none)
        self.g.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        (self.g if self.model.turn == 1 else self.d).step()


def optimizer(model: Gan, cfg: dict) -> Alternating:
    kind = str(cfg.get("optimizer", "adam"))
    lr = float(cfg["lr"])
    lr_g = float(cfg.get("lrG") or lr)

    def make(params, rate: float) -> torch.optim.Optimizer:
        if kind == "sgd":
            return torch.optim.SGD(params, lr=rate, foreach=False)
        if kind == "adam":
            betas = tuple(cfg.get("betas") or (0.9, 0.999))
            return torch.optim.Adam(params, lr=rate, betas=betas, eps=float(cfg.get("eps", 1e-8)), foreach=False)
        raise ValueError(f"gan-2d twin: optimizer {kind!r} is not used by the page")

    return Alternating(model, make(model.disc.parameters(), lr), make(model.gen.parameters(), lr_g))
