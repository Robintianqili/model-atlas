"""Dump test samples for the node checks (pipeline/parity/tinycnn.drawing.ts).
out/base_test_sample.bin: 150 test drawings per base class (uint8 784 each), labels in out/base_test_sample.json."""
import json, os
import numpy as np
from common import load_split
from classes import BASE
X, y, idx = load_split(BASE, "test")
rng = np.random.default_rng(3)
sel = np.concatenate([rng.choice(np.where(y == k)[0], 150, replace=False) for k in range(len(BASE))])
os.makedirs("out", exist_ok=True)
X[sel].astype(np.uint8).tofile("out/base_test_sample.bin")
json.dump({"classes": BASE, "labels": y[sel].tolist(), "fileIndex": idx[sel].tolist()}, open("out/base_test_sample.json", "w"))
print(len(sel))
