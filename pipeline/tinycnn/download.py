"""Download the first N drawings per class of Quick, Draw! numpy bitmaps via HTTP Range.

Source: https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<class>.npy
License: CC BY 4.0, "Quick, Draw! dataset by Google Creative Lab".
Saves pipeline/data/quickdraw/<class>.npy as uint8 [N, 784].
"""
import ast, os, sys, struct, urllib.request, urllib.parse
import numpy as np

N = 15000
BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "quickdraw")

def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    return data

def download(cls):
    path = os.path.join(OUT, cls.replace(" ", "_") + ".npy")
    if os.path.exists(path):
        a = np.load(path)
        if a.shape == (N, 784):
            return "cached"
    url = BASE + urllib.parse.quote(cls) + ".npy"
    head = fetch_range(url, 0, 255)
    assert head[:6] == b"\x93NUMPY", head[:10]
    major = head[6]
    if major == 1:
        hlen = struct.unpack("<H", head[8:10])[0]; hstart = 10
    else:
        hlen = struct.unpack("<I", head[8:12])[0]; hstart = 12
    header = head[hstart:hstart + hlen].decode("latin1")
    d = ast.literal_eval(header)
    assert d["descr"] in ("|u1", "<u1", "u1"), d
    assert not d["fortran_order"]
    total, w = d["shape"]
    assert w == 784
    n = min(N, total)
    off = hstart + hlen
    body = fetch_range(url, off, off + n * 784 - 1)
    assert len(body) == n * 784, (len(body), n * 784)
    a = np.frombuffer(body, dtype=np.uint8).reshape(n, 784)
    np.save(path, a)
    return f"ok total={total}"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for c in sys.argv[1:]:
        try:
            print(c, download(c), flush=True)
        except Exception as e:
            print(c, "FAILED", repr(e), flush=True)
