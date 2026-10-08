"""Screen candidate classes: train the tiny model on all candidates, then score 10-class subsets
by restricting the logits to the subset (proxy for a 10-way model's accuracy)."""
import sys, json, itertools, time
import numpy as np, torch, torch.nn.functional as F
from common import *

CANDS = ["cat","fish","house","tree","car","star","umbrella","clock","bicycle","sun","apple","key",
         "ladder","envelope","smiley face","cup","light bulb","moon","lightning","cloud","flower","t-shirt",
         "eyeglasses","scissors","airplane","butterfly","door","mountain","ice cream","donut"]
dev = "cuda"
torch.manual_seed(0)
Xtr, ytr, _ = load_split(CANDS, "train"); Xva, yva, _ = load_split(CANDS, "val")
Xtr = torch.tensor(Xtr, device=dev).unsqueeze(1); ytr = torch.tensor(ytr, device=dev)
Xva = torch.tensor(Xva, device=dev).unsqueeze(1).float() / 255; yva = torch.tensor(yva, device=dev)
m = TinyVGG((16, 16, 32, 32), len(CANDS)).to(dev)
opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-4)
E = 8; bs = 512; steps = E * (len(Xtr) // bs)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
gen = torch.Generator(device=dev); gen.manual_seed(0)
t0 = time.time()
for e in range(E):
    perm = torch.randperm(len(Xtr), device=dev)
    for i in range(len(Xtr) // bs):
        b = perm[i * bs:(i + 1) * bs]
        x = augment(Xtr[b].float() / 255, gen)
        loss = F.cross_entropy(m(x), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    print(e, loss.item(), time.time() - t0, flush=True)
m.eval()
with torch.no_grad():
    L = torch.cat([m(Xva[i:i + 4096]) for i in range(0, len(Xva), 4096)]).cpu().numpy()
yv = yva.cpu().numpy()
print("val acc 30-way", (L.argmax(1) == yv).mean())
C = len(CANDS)
conf = np.zeros((C, C))
for i in range(C):
    for j in range(C):
        if i == j: continue
        rows = yv == i
        conf[i, j] = (L[rows, j] > L[rows, i]).mean()  # pairwise: how often j beats i on i's images
np.save("screen_logits.npy", L); np.save("screen_y.npy", yv)
pc = (L.argmax(1) == yv).reshape(C, -1).mean(1)
for i in np.argsort(pc): print(f"{CANDS[i]:12s} acc30={pc[i]:.3f}  worst-pair {CANDS[np.argmax(conf[i])]} {conf[i].max():.3f}")
