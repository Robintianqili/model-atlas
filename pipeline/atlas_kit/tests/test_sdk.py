"""atlas_kit self-tests (stdlib unittest; no pytest needed).

    cd pipeline && python -m unittest atlas_kit.tests.test_sdk -v
or  npm run atlas -- doctor   (runs a short import check only)

Writes only into temporary folders. The CUDA test is skipped on machines without a GPU.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import atlas_kit as ak
from atlas_kit import numerics
from atlas_kit.refs import read_reference
from atlas_kit.tape import decode, read_tape

ROOT = Path(__file__).resolve().parents[3]


def mlp(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 1)

        def forward(self, x):
            return torch.sigmoid(self.fc2(torch.tanh(self.fc1(x))))

    return M()


def describe_for(m: nn.Module):
    def describe(b: "ak.SpecBuilder", c: "ak.Case") -> None:
        x = b.input("input", c.input, adapter="points2d", kind="points2d", axes="f")
        h = b.tanh("act1", b.linear("fc1", x, m.fc1, label=("Hidden layer", "隐藏层")))
        b.output(b.sigmoid("out", b.linear("fc2", h, m.fc2, label=("Output score", "输出分数"))))
    return describe


CASES = [ak.Case("zero", "(0, 0)", torch.tensor([0.0, 0.0])), ak.Case("one", "(1, 0)", torch.tensor([1.0, 0.0]), edge=True),
         ak.Case("big", "(3, -2)", torch.tensor([3.0, -2.0]))]


class TempDir(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="atlas_kit_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- acceptance 1: Rng
class RngTest(unittest.TestCase):
    def test_doc_vectors(self):
        r = ak.Rng(42)
        self.assertEqual([f"{r.next():.10f}" for _ in range(4)], ["0.6011037519", "0.4482905590", "0.8524657935", "0.6697340414"])
        r = ak.Rng(1)
        self.assertEqual([f"{r.next():.10f}" for _ in range(3)], ["0.6270739406", "0.0027357212", "0.5274470400"])

    def test_int_and_fork_vectors(self):
        # Exact values (float repr), shared with site/scripts/atlas/tests/rng.test.ts, which recomputes them in TS.
        r = ak.Rng(42)
        self.assertEqual([r.int(10) for _ in range(8)], [6, 4, 8, 6, 1, 5, 2, 6])
        self.assertEqual(ak.fnv1a32("init"), 0x16B1D373)
        f = ak.Rng(42).fork("init")
        self.assertEqual(f.seed, (42 ^ 0x16B1D373) & 0xFFFFFFFF)
        self.assertEqual(f.seed, 380752729)
        self.assertEqual(f.next(), 0.3591537452302873)
        g = ak.Rng(7).fork("batch")
        self.assertEqual([g.int(100) for _ in range(5)], [27, 91, 61, 27, 53])

    def test_fork_independent_of_consumption(self):
        a, b = ak.Rng(123), ak.Rng(123)
        for _ in range(17):
            b.next()
        self.assertEqual(a.fork("x").next(), b.fork("x").next())
        self.assertEqual(a.fork("x").fork("y").seed, (123 ^ ak.fnv1a32("x") ^ ak.fnv1a32("y")) & 0xFFFFFFFF)

    def test_state_roundtrip_and_seed_wrap(self):
        r = ak.Rng(5)
        r.next()
        s = r.get_state()
        x = r.next()
        r.set_state(s)
        self.assertEqual(r.next(), x)
        self.assertEqual(ak.Rng(-1).seed, 0xFFFFFFFF)          # JS: -1 >>> 0
        self.assertEqual(ak.Rng(2**32 + 3).seed, 3)
        self.assertEqual(ak.Rng(7.9).seed, 7)                   # ToUint32 truncates
        self.assertTrue(all(0.0 <= ak.Rng(s).next() < 1.0 for s in (0, 1, 2**31, 2**32 - 1)))

    def test_normal_box_muller(self):
        r = ak.Rng(9)
        q = ak.Rng(9)
        u1 = 1 - q.next()
        u2 = q.next()
        self.assertAlmostEqual(r.normal(), math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2), places=15)


# ---------------------------------------------------------------------------------------------- acceptance 4: paths
class PathsTest(TempDir):
    def run_py(self, code: str, cwd: Path, env: dict | None = None) -> str:
        e = {k: v for k, v in os.environ.items() if k not in ("ATLAS_SITE", "ATLAS_ROOT")}
        e["PYTHONPATH"] = str(ROOT / "pipeline")
        e.update(env or {})
        out = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=e, capture_output=True, text=True, timeout=120)
        if out.returncode:
            raise AssertionError(out.stderr)
        return out.stdout.strip()

    CODE = "from atlas_kit.paths import atlas_site, atlas_root, lane_name; print(atlas_site().as_posix()); print(atlas_root().as_posix()); print(lane_name())"

    def test_main_tree(self):
        for cwd in (ROOT, ROOT / "pipeline", ROOT / "site" / "src" / "lib", ROOT / "notes"):
            site, root, lane = self.run_py(self.CODE, cwd).splitlines()
            self.assertEqual(Path(site), (ROOT / "site").resolve(), cwd)
            self.assertEqual(Path(root), ROOT.resolve(), cwd)
            self.assertEqual(lane, "None")

    def test_lane(self):
        lanes = [p for p in (ROOT / "site" / ".work").glob("*/site") if (p / "svelte.config.js").is_file()]
        if not lanes:
            self.skipTest("no .work lane on this machine")
        lane = lanes[0]
        site, root, name = self.run_py(self.CODE, lane / "src").splitlines()
        self.assertEqual(Path(site), lane.resolve())
        self.assertEqual(Path(root), ROOT.resolve())
        self.assertEqual(name, lane.parent.name)
        # the CLI passes both explicitly
        site, root, name = self.run_py(self.CODE, self.tmp, {"ATLAS_SITE": str(lane), "ATLAS_ROOT": str(ROOT)}).splitlines()
        self.assertEqual((Path(site), Path(root)), (lane.resolve(), ROOT.resolve()))

    def test_synthetic_lane_layout(self):
        # <tmp>/repo/{CONTRACTS.md, pipeline/, site/.work/x/site/...}: site/.work/x/site/.. has no pipeline/
        repo = self.tmp / "repo"
        lane = repo / "site" / ".work" / "x" / "site"
        for d in (repo / "pipeline", repo / "site" / "src" / "lib", lane / "src" / "lib"):
            d.mkdir(parents=True)
        (repo / "CONTRACTS.md").write_text("x")
        for s in (repo / "site", lane):
            (s / "svelte.config.js").write_text("")
            (s / "package.json").write_text("{}")
        site, root, name = self.run_py(self.CODE, lane / "src" / "lib").splitlines()
        self.assertEqual((Path(site), Path(root), name), (lane.resolve(), repo.resolve(), "x"))
        site, root, name = self.run_py(self.CODE, repo / "pipeline").splitlines()
        self.assertEqual((Path(site), Path(root), name), ((repo / "site").resolve(), repo.resolve(), "None"))


# ---------------------------------------------------------------------------------------------- export
class ExportTest(TempDir):
    def test_roundtrip(self):
        m = mlp()
        rep = ak.export_model(describe_for(m), CASES, slug="demo", name=("Demo MLP", "演示 MLP"), out=str(self.tmp),
                              module=m, module_input=lambda c: c.input.unsqueeze(0), facts={"testAcc": {"value": 0.97, "unit": "share"}},
                              presets={"points": [[0, 0], [1, 1]]}, extras={"classes": ["a", "b"]})
        self.assertEqual(ak.validate(self.tmp), [])
        self.assertEqual(rep.params, 2 * 4 + 4 + 4 + 1)
        self.assertEqual(rep.module_max_abs_diff, 0.0)
        spec = json.loads((self.tmp / "spec.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["atlas"]["spec"], "1.1")
        self.assertEqual(spec["parity"]["max_abs_diff"], None)
        self.assertEqual(spec["parity"]["checked"], None)
        self.assertEqual([l["id"] for l in spec["layers"]], ["input", "fc1", "act1", "fc2", "out"])
        self.assertEqual(spec["outputs"], ["out"])
        self.assertEqual(spec["classes"], ["a", "b"])
        raw = (self.tmp / "spec.json").read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r\n", raw)
        w = np.fromfile(self.tmp / "weights.bin", dtype="<f4")
        t = {e["name"]: e for e in spec["tensors"]}
        e = t["fc1.weight"]
        np.testing.assert_array_equal(w[e["offset"]:e["offset"] + e["length"]].reshape(e["shape"]), m.fc1.weight.detach().numpy())
        ref = read_reference(json.loads((self.tmp / "reference.json").read_text(encoding="utf-8")))
        self.assertEqual([c["id"] for c in ref], ["zero", "one", "big"])
        self.assertTrue(ref[1]["edge"])
        with torch.no_grad():
            y = m(CASES[2].input.unsqueeze(0))[0].numpy()
        np.testing.assert_array_equal(ref[2]["tensors"]["out"], y)
        facts = json.loads((self.tmp / "facts.json").read_text(encoding="utf-8"))
        self.assertEqual(facts["format"], "atlas-facts/1")
        self.assertEqual(facts["facts"]["testAcc"]["value"], 0.97)
        self.assertIn("measured", facts["facts"]["testAcc"])

    def test_tied_weights_and_symbolic_dims(self):
        torch.manual_seed(1)
        emb = nn.Embedding(11, 6)
        def describe(b, c):
            ids = b.input("input", c.input, adapter="text-char", kind="text", axes="t", shape=["T"],
                          dims={"T": (1, 16, ("characters", "字符"))})
            x = b.op("embedding", "tok_emb", [ids], weights={"weight": emb.weight},
                     ref=lambda xs, p, w: w["weight"][xs[0]], label=("Embedding", "嵌入"))
            h = b.relu("act", x)
            b.output(b.linear("head", h, tied_to="tok_emb.weight", label=("Head", "输出头")))
        cases = [ak.Case("t1", "1 char", torch.tensor([3]), edge=True), ak.Case("t5", "5 chars", torch.tensor([1, 2, 3, 4, 5])),
                 ak.Case("t16", "16 chars", torch.arange(16) % 11, edge=True)]
        rep = ak.export_model(describe, cases, slug="tied", name=("Tied", "共享"), out=str(self.tmp))
        spec = json.loads((self.tmp / "spec.json").read_text(encoding="utf-8"))
        self.assertEqual([t["name"] for t in spec["tensors"]], ["tok_emb.weight"])  # stored once
        self.assertEqual(rep.params, 66)
        head = spec["layers"][-1]
        self.assertEqual(head["w"], {"weight": "tok_emb.weight"})
        self.assertEqual(head["outShape"], ["T", 11])
        self.assertEqual(spec["dims"]["T"], {"min": 1, "max": 16, "label": {"en": "characters", "zh": "字符"}})
        self.assertEqual(spec["inputs"][0]["dtype"], "i32")
        self.assertEqual([l["id"] for l in spec["layers"]], ["tok_emb", "act", "head"])  # i32 input: no input layer
        self.assertEqual(ak.validate(self.tmp), [])
        ref = read_reference(json.loads((self.tmp / "reference.json").read_text(encoding="utf-8")))
        self.assertEqual(ref[1]["inputs"]["input"].dtype, np.dtype("<i4"))

    def test_sub_steps(self):
        def describe(b, c):
            x = b.input("input", c.input, adapter="points2d", kind="points2d", axes="f")
            b.output(b.op("double_then_square", "f", [x], ref=lambda xs, p, w: ((2 * xs[0]) ** 2, {"double": 2 * xs[0]}),
                          label=("Local op", "本地算子")))
        ak.export_model(describe, CASES, slug="subs", name=("Subs", "子步骤"), out=str(self.tmp))
        spec = json.loads((self.tmp / "spec.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["layers"][-1]["sub"], ["double"])
        ref = json.loads((self.tmp / "reference.json").read_text(encoding="utf-8"))
        self.assertEqual(ref["cases"][0]["order"], ["input", "f", "f#double"])
        self.assertEqual(ak.validate(self.tmp), [])

    def assertCode(self, code: str, fn, *a, **kw):
        with self.assertRaises(ak.AtlasError) as cm:
            fn(*a, **kw)
        self.assertEqual(cm.exception.code, code, str(cm.exception))
        return cm.exception

    # acceptance 2: CUDA refused
    def test_ak102_cuda_module_and_tensor(self):
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device")
        m = mlp().cuda()
        e = self.assertCode("AK102", ak.export_model, describe_for(m), CASES, slug="demo", name=("D", "D"), out=str(self.tmp))
        self.assertIn("cuda", e.message)
        e = self.assertCode("AK102", ak.export_model, describe_for(mlp()), CASES, slug="demo", name=("D", "D"), out=str(self.tmp),
                            module=m, module_input=lambda c: c.input)
        self.assertIn("module", e.message)
        cuda_cases = [ak.Case("g", "gpu", torch.tensor([0.0, 1.0], device="cuda"))]
        e = self.assertCode("AK102", ak.export_model, describe_for(mlp()), cuda_cases, slug="demo", name=("D", "D"), out=str(self.tmp))
        self.assertIn("input", e.message)
        self.assertFalse(any(self.tmp.iterdir()), "nothing may be written when export fails")

    def test_ak102_tf32_switched_on(self):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            self.assertCode("AK102", ak.export_model, describe_for(mlp()), CASES, slug="demo", name=("D", "D"), out=str(self.tmp))
        finally:
            numerics.enforce()
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_ak101_graph_changes(self):
        m = mlp()
        def describe(b, c):
            x = b.input("input", c.input, adapter="points2d", kind="points2d", axes="f")
            h = b.linear("fc1", x, m.fc1)
            if c.id == "one":
                h = b.relu("extra", h)
            b.output(h)
        self.assertCode("AK101", ak.export_model, describe, CASES, slug="demo", name=("D", "D"), out=str(self.tmp))

    def test_ak101_weights_change(self):
        def describe(b, c):
            x = b.input("input", c.input, adapter="points2d", kind="points2d", axes="f")
            b.output(b.linear("fc1", x, mlp(seed=len(c.id)).fc1))  # a new model per case
        self.assertCode("AK101", ak.export_model, describe, CASES, slug="demo", name=("D", "D"), out=str(self.tmp))

    def test_ak103_describe_differs_from_module(self):
        m = mlp()
        def describe(b, c):  # forgets the final sigmoid
            x = b.input("input", c.input, adapter="points2d", kind="points2d", axes="f")
            b.output(b.linear("fc2", b.tanh("act1", b.linear("fc1", x, m.fc1)), m.fc2))
        self.assertCode("AK103", ak.export_model, describe, CASES, slug="demo", name=("D", "D"), out=str(self.tmp),
                        module=m, module_input=lambda c: c.input.unsqueeze(0))

    def test_ak105_budget(self):
        e = self.assertCode("AK105", ak.export_model, describe_for(mlp()), CASES, slug="demo", name=("D", "D"), out=str(self.tmp),
                            file_mb=0.001)
        self.assertIn("MB", e.message)
        self.assertFalse(any(self.tmp.iterdir()))

    def test_ak106_attrs(self):
        m = mlp()
        def describe(b, c):
            x = b.input("input", c.input, adapter="points2d", kind="points2d", axes="f")
            b.output(b.op("relu", "r", [x], params={"slope": 0.1}))
        self.assertCode("AK106", ak.export_model, describe, CASES, slug="demo", name=("D", "D"), out=str(self.tmp))
        def describe2(b, c):
            x = b.input("input", c.input, adapter="points2d", kind="points2d", axes="f")
            b.output(b.linear("fc1", x, m.fc1, accumulate=3))
        self.assertCode("AK106", ak.export_model, describe2, CASES, slug="demo", name=("D", "D"), out=str(self.tmp))

    def test_bad_ids(self):
        def describe(b, c):
            x = b.input("input", c.input, adapter="points2d", kind="points2d", axes="f")
            b.relu("a#b", x)
        self.assertCode("AK301", ak.export_model, describe, CASES, slug="demo", name=("D", "D"), out=str(self.tmp))

    def test_varying_dim_without_declaration(self):
        def describe(b, c):
            b.input("input", c.input, adapter="series", kind="series", axes="t")
        cases = [ak.Case("a", "a", torch.zeros(3)), ak.Case("b", "b", torch.zeros(4))]
        self.assertCode("AK101", ak.export_model, describe, cases, slug="demo", name=("D", "D"), out=str(self.tmp))


# ---------------------------------------------------------------------------------------------- tapes
class TapeTest(TempDir):
    SCHEMA = {"Q": {"kind": "tensor", "label": ("Q table", "Q 表"), "track": "changes", "tol": 1e-12},
              "pos": {"kind": "int", "label": ("Position", "位置"), "tol": "exact"},
              "V": {"kind": "tensor", "label": ("Values", "价值"), "derived": True}}
    PHASES = {"act": {"label": ("Act", "行动"), "kind": "act"}, "update": {"label": ("Update", "更新"), "kind": "update"}}

    def test_write_read(self):
        w = ak.TapeWriter("q", self.SCHEMA, seed=7, opt={"alpha": 0.5}, key_every=4, phases=self.PHASES)
        Q = np.zeros((3, 2), dtype=np.float64)
        rng = ak.Rng(7)
        for step in range(10):
            a = rng.int(2)
            w.frame("act", {"pos": step % 3, **({"Q": Q} if step == 0 else {})}, step=step, metrics={"a": a})
            old = Q[step % 3, a]
            Q[step % 3, a] += 0.5 * (1 - old)
            w.frame("update", {"Q": Q}, step=step, changes=[{"field": "Q", "index": [step % 3, a], "from": float(old), "to": float(Q[step % 3, a])}],
                    metrics={"r": 1.0, "loss": math.nan}, draws=[0.25])
        j, b = w.write(self.tmp / "tapes" / "q")
        self.assertEqual(j.name, "q.tape.json")
        h, blob = read_tape(self.tmp / "tapes" / "q")
        self.assertEqual(h["format"], "atlas-tape/1")
        self.assertEqual(len(h["frames"]), 20)
        self.assertEqual(h["bin"]["bytes"], len(blob))
        keys = [i for i, f in enumerate(h["frames"]) if f.get("key")]
        self.assertEqual(keys, [0, 4, 8, 12, 16])
        self.assertNotIn("Q", h["frames"][1]["state"])
        # replay: keyframe value + changes == the running Q at frame 19
        last_key = h["frames"][16]
        q = np.array(decode(last_key["state"]["Q"], blob))
        for f in h["frames"][17:]:
            for c in f.get("changes", []):
                q[tuple(c["index"])] = c["to"]
        np.testing.assert_array_equal(q, Q)
        self.assertTrue(math.isnan(decode(h["frames"][1]["metrics"], blob)["loss"]))
        self.assertEqual(h["schema"]["Q"]["label"], {"en": "Q table", "zh": "Q 表"})
        self.assertNotIn(b"\r\n", j.read_bytes())

    def test_unreported_mutation(self):
        w = ak.TapeWriter("q", self.SCHEMA, seed=1, opt={}, phases=self.PHASES)
        Q = np.zeros((2, 2))
        w.frame("act", {"Q": Q, "pos": 0})
        Q2 = Q.copy()
        Q2[0, 0] = 1
        Q2[1, 1] = 5  # not reported below
        with self.assertRaises(ak.AtlasError) as cm:
            w.frame("update", {"Q": Q2}, changes=[{"field": "Q", "index": [0, 0], "to": 1.0}])
        self.assertEqual(cm.exception.code, "AK301")

    def test_derived_not_stored(self):
        w = ak.TapeWriter("q", self.SCHEMA, seed=1, opt={}, phases=self.PHASES)
        with self.assertRaises(ak.AtlasError):
            w.frame("act", {"Q": np.zeros(2), "V": np.zeros(2)})


# ---------------------------------------------------------------------------------------------- data
class DataTest(unittest.TestCase):
    def test_karate_unweighted(self):
        k = ak.data.karate()
        self.assertEqual(k["A"].shape, (34, 34))
        self.assertEqual(int(k["A"].sum()) // 2, 78)
        self.assertEqual(len(k["edges"]), 78)
        self.assertEqual(set(np.unique(k["A"])), {0.0, 1.0})
        self.assertEqual(int(k["labels"].sum()), 17)  # 17 members follow the officer

    def test_tinystories(self):
        p = ak.root_path("pipeline", "data", "tinystories", "train_head_100MB.raw.txt")
        if not p.is_file():
            self.skipTest("TinyStories not downloaded")
        t = ak.data.tinystories_chars(n_chars=5000)
        self.assertLessEqual(len(t), 5000)
        self.assertGreater(len(t), 1000)
        self.assertNotIn("<|endoftext|>", t)

    def test_quickdraw(self):
        if not ak.root_path("pipeline", "data", "quickdraw", "apple.npy").is_file():
            self.skipTest("Quick, Draw! not downloaded")
        X, y, idx = ak.data.quickdraw(["apple", "sun"], "test")
        self.assertEqual(X.shape, (3000, 28, 28))
        self.assertEqual(X.dtype, np.uint8)
        self.assertEqual(sorted(set(y.tolist())), [0, 1])
        with self.assertRaises(ak.AtlasError) as cm:
            ak.data.quickdraw(["not a class"], "test")
        self.assertEqual(cm.exception.code, "AK002")


class ValidateLegacyTest(unittest.TestCase):
    def test_shipped_folders(self):
        for slug in ("tinycnn", "tinygpt"):
            d = ak.site_path("static", "models", slug)
            if d.is_dir():
                self.assertEqual(ak.validate(d), [], slug)


if __name__ == "__main__":
    unittest.main()
