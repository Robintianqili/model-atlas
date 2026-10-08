"""The Python op mirrors against ops.json, the torch goldens and the TS ops (design §8.2 W1-E2 acceptance 2).

    cd pipeline && python -m unittest atlas_kit.tests.test_mirrors -v      (ATLAS_SITE = the site or lane to check)
    python -m atlas_kit.tests.test_mirrors                                   (prints the table only)

1. Metadata: every mirror's OP, VERSION, ATTRS (type, default, required), ALIASES (and alias params), WEIGHTS, SUBS,
   CATEGORY and LABEL equal its entry in site/src/lib/atlas/schema/ops.json, and every ops.json op has a mirror.
2. Goldens: each mirror's reference(), run in float64 on the float32 golden inputs, equals the golden (PyTorch float64)
   within |a - b| <= 1e-6 + 1e-6 |b| for every value and every sub-step.
3. TS ops: the same reference equals what the TS op computes on the same inputs (scripts/atlas/tools/op-outputs.ts,
   run with node), within the same bar. The float32 path that SpecBuilder uses is reported too (it does not gate:
   float32 torch kernels sum in float32, the TS ops in float64).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

import atlas_kit as ak  # noqa: F401  (TF32 off)
from atlas_kit import ops as M
from atlas_kit.paths import atlas_site

ATOL = RTOL = 1e-6


def _site() -> Path:
    return Path(os.environ["ATLAS_SITE"]).resolve() if os.environ.get("ATLAS_SITE") else atlas_site()


def _dec(t: dict, dtype=None) -> torch.Tensor:
    raw = base64.b64decode(t["b64"])
    np_dt = {"float32": "<f4", "float64": "<f8", "int32": "<i4"}[t["dtype"]]
    a = np.frombuffer(raw, dtype=np_dt).reshape(t["shape"]).copy()
    x = torch.from_numpy(a)
    if dtype is not None and x.dtype.is_floating_point:
        x = x.to(dtype)
    return x


def _ratio(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(max |a-b| / (atol + rtol |b|), max |a-b|) with -inf == -inf; NaN or unmatched infinities give inf."""
    a = a.detach().double().reshape(-1)
    b = b.detach().double().reshape(-1)
    if a.shape != b.shape:
        return float("inf"), float("inf")
    if a.numel() == 0:
        return 0.0, 0.0
    both = torch.isinf(a) & torch.isinf(b) & (torch.sign(a) == torch.sign(b))
    d = torch.where(both, torch.zeros_like(a), (a - b).abs())
    if torch.isnan(d).any():
        return float("inf"), float("inf")
    bb = torch.where(torch.isinf(b), torch.zeros_like(b), b.abs())
    return float((d / (ATOL + RTOL * bb)).max()), float(d.max())


def _run(mirror, case: dict, dtype) -> tuple[torch.Tensor, dict]:
    _, alias_params, _, _ = M.resolve(case["op"])
    params = {**alias_params, **case["params"]}
    xs = [_dec(x, dtype) for x in case["inputs"]]
    ws = {k: _dec(v, dtype) for k, v in case["weights"].items()}
    with torch.no_grad():
        out = mirror.reference(xs, params, ws)
    subs: dict = {}
    if isinstance(out, tuple):
        out, subs = out
    return out, subs


def ts_outputs(site: Path) -> dict | None:
    """What every TS op computes on its golden inputs (None when node is not available)."""
    node = shutil.which("node")
    tool = site / "scripts" / "atlas" / "tools" / "op-outputs.ts"
    if not node or not tool.is_file():
        return None
    r = subprocess.run([node, str(tool)], cwd=site, capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "ATLAS_SITE": str(site)}, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"op-outputs.ts failed:\n{r.stderr}")
    return json.loads(r.stdout)["ops"]


def compare_all(site: Path) -> list[dict]:
    """One row per op: cases, worst ratios (f64 mirror vs golden, f64 mirror vs TS, f32 mirror vs golden)."""
    gdir = site / "tests" / "atlas" / "ops"
    ts = ts_outputs(site)
    rows = []
    for name, mirror in M.MIRRORS.items():
        f = gdir / f"{name}.json"
        row = {"op": name, "cases": 0, "golden": None, "ts": None, "f32": None, "f32_abs": None, "ts_cases": 0}
        if not f.is_file():
            rows.append(row)
            continue
        g = json.loads(f.read_text(encoding="utf-8"))
        tsc = {c["id"]: c for c in (ts or {}).get(name, [])}
        wg = wt = w32 = a32 = 0.0
        for case in g["cases"]:
            row["cases"] += 1
            out, subs = _run(mirror, case, torch.float64)
            pairs = [(out, case["out"], tsc.get(case["id"], {}).get("out"))]
            for k, v in (case.get("subs") or {}).items():
                pairs.append((subs.get(k), v, (tsc.get(case["id"], {}).get("subs") or {}).get(k)))
            out32, subs32 = _run(mirror, case, torch.float32)
            p32 = [(out32, case["out"])] + [(subs32.get(k), v) for k, v in (case.get("subs") or {}).items()]
            for mine, gold, tso in pairs:
                if mine is None:
                    wg = float("inf")
                    continue
                wg = max(wg, _ratio(mine, _dec(gold))[0])
                if tso is not None:
                    wt = max(wt, _ratio(mine, _dec(tso))[0])
            if case["id"] in tsc:
                row["ts_cases"] += 1
            for mine, gold in p32:
                r, a = _ratio(mine, _dec(gold)) if mine is not None else (float("inf"), float("inf"))
                w32, a32 = max(w32, r), max(a32, a)
        row.update(golden=wg, ts=wt if tsc else None, f32=w32, f32_abs=a32)
        rows.append(row)
    return rows


def table(rows: list[dict]) -> str:
    fmt = lambda v: "-" if v is None else f"{v:.3g}"  # noqa: E731
    out = [f"{'op':<20} {'cases':>5}  {'f64 vs golden':>13}  {'f64 vs TS':>9}  {'f32 vs golden':>13}  {'f32 max|d|':>10}"]
    for r in rows:
        out.append(f"{r['op']:<20} {r['cases']:>5}  {fmt(r['golden']):>13}  {fmt(r['ts']):>9}  {fmt(r['f32']):>13}  {fmt(r['f32_abs']):>10}")
    out.append("(ratios are max |a-b| / (1e-6 + 1e-6|b|); <= 1 passes)")
    return "\n".join(out)


class MirrorTest(unittest.TestCase):
    rows: list[dict] = []

    @classmethod
    def setUpClass(cls) -> None:
        M.clear_cache()
        cls.site = _site()
        cls.rows = compare_all(cls.site)
        print("\n" + table(cls.rows), file=sys.stderr)

    def test_manifest(self) -> None:
        self.assertIsNotNone(M.load_manifest(self.site), "schema/ops.json not found")
        self.assertEqual(M.compare_with_manifest(self.site), [])
        self.assertEqual(len(M.MIRRORS), 20)

    def test_goldens(self) -> None:
        for r in self.rows:
            with self.subTest(op=r["op"]):
                self.assertEqual(r["cases"], 8, f"{r['op']}: golden file missing or incomplete")
                self.assertLessEqual(r["golden"], 1.0, f"{r['op']}: mirror vs torch golden")

    def test_ts_ops(self) -> None:
        if not shutil.which("node"):
            self.skipTest("node not on PATH")
        for r in self.rows:
            with self.subTest(op=r["op"]):
                self.assertEqual(r["ts_cases"], 8, f"{r['op']}: TS outputs missing")
                self.assertLessEqual(r["ts"], 1.0, f"{r['op']}: mirror vs TS op")

    def test_lstm_builder_ak104(self) -> None:
        """SpecBuilder.lstm records the gate subs and proves them against nn.LSTM; a doctored module gives AK104."""
        torch.manual_seed(3)
        m = torch.nn.LSTM(3, 4)

        def describe(b, c):
            x = b.input("x", c.input, adapter="series", kind="series", axes="t,d", shape=["T", 3],
                        dims={"T": (1, 8, ("steps", "步"))}, layer=False)
            b.output(b.lstm("lstm1", x, m))

        cases = [ak.Case("t1", "1 step", torch.randn(1, 3)), ak.Case("t8", "8 steps", torch.randn(8, 3))]
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            rep = ak.export_model(describe, cases, slug="lstm-demo", name=("LSTM demo", "LSTM 演示"), out=d,
                                  module=m, module_input=lambda c: c.input)
            spec = json.loads(Path(d, "spec.json").read_text(encoding="utf-8"))
            lay = spec["layers"][0]
            self.assertEqual(lay["op"], "lstm")
            self.assertEqual(lay["sub"], ["i", "f", "g", "o", "c", "h"])
            self.assertEqual(lay["outShape"], ["T", 4])
            self.assertEqual(lay["params"], {"hidden": 4, "inputSize": 3})
            self.assertLessEqual(rep.module_max_abs_diff, 1e-6)

        class Doctored(torch.nn.LSTM):
            def forward(self, x, hx=None):  # a different gate order than the reference loop
                y, s = super().forward(x, hx)
                return y * 0.5, s

        bad = Doctored(3, 4)

        def describe_bad(b, c):
            x = b.input("x", c.input, adapter="series", kind="series", axes="t,d", layer=False)
            b.output(b.lstm("lstm1", x, bad))

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ak.AtlasError) as e:
                ak.export_model(describe_bad, [ak.Case("a", "a", torch.randn(5, 3))], slug="x", name=("x", "x"), out=d)
            self.assertEqual(e.exception.code, "AK104")


if __name__ == "__main__":
    print(table(compare_all(_site())))
