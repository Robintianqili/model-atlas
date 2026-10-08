"""W1-E2 acceptance 1: re-describing the shipped TinyCNN with the SDK gives the same layer ids, ops, outShape and params,
and a byte-identical weights.bin. Written to a temporary folder (the CLI form writes $ATLAS_SITE/.atlas/tmp/tinycnn/);
never to static/.

    cd pipeline && python -m unittest atlas_kit.tests.test_redescribe -v
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from atlas_kit.paths import atlas_root, atlas_site
from atlas_kit.tests.redescribe_tinycnn import redescribe

SHIPPED = atlas_site() / "static" / "models" / "tinycnn"


@unittest.skipUnless((atlas_root() / "pipeline" / "tinycnn" / "model_final.pt").is_file() and (SHIPPED / "spec.json").is_file(),
                     "TinyCNN checkpoint or shipped assets not found")
class RedescribeTinyCNN(unittest.TestCase):
    def test_same_graph_and_byte_identical_weights(self) -> None:
        before = hashlib.sha256((SHIPPED / "spec.json").read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            r = redescribe(Path(d) / "tinycnn")
        self.assertEqual(r["layer_diffs"], [])
        self.assertTrue(r["same_ids_ops_outShape_params"])
        self.assertTrue(r["tensors_table_equal"])
        self.assertTrue(r["weights_byte_identical"], f"{r['weights_sha256']} vs {r['shipped_weights_sha256']}")
        self.assertEqual(r["layers"], 14)
        self.assertEqual(r["params"], 6722)
        self.assertEqual(r["module_max_abs_diff"], 0.0)
        self.assertEqual(hashlib.sha256((SHIPPED / "spec.json").read_bytes()).hexdigest(), before, "static/ is never written")


if __name__ == "__main__":
    unittest.main()
