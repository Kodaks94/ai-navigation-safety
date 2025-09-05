import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from transformer_policy import TransformerPolicy, PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART  = ROOT / "artifacts"

# A* utilities (copied minimal)
import heapq
MOVES8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
def astar(start, goal, hazard, mpa, obstacles, w_dist=1.0, w_hazard=1.5, w_mpa=8.0, diag=True):
    H, W = hazard.shape
    MOVES = MOVES8 if diag else [(1,0),(-1,0),(0,1),(0,-1)]
    def in_bounds(y,x): return 0 <= y < H and 0 <= x < W
    def cost(y,x): return w_dist + w_hazard*float(hazard[y,x]) + (w_mpa if mpa[y,x] else 0.0)
    def heuristic(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
    if obstacles[start] or obstacles[goal]: return None
    pq=[]; heapq.heappush(pq,(0.0,start)); came={start:None}; g={start:0.0}
    while pq:
        f, cur = heapq.heappop(pq)
        if cur == goal: break
        cy,cx=cur
        for dy,dx in MOVES:
            ny,nx=cy+dy,cx+dx
            if not in_bounds(ny,nx) or obstacles[ny,nx]: continue
            sc = cost(ny,nx)
            ng = g[cur] + sc
            if (ny,nx) not in g or ng < g[(ny,nx)]:
                g[(ny,nx)] = ng; came[(ny,nx)] = cur
                heapq.heappush(pq, (ng+heuristic((ny,nx),goal), (ny,nx)))
    if goal not in came: return None
    path=[]; cur=goal
    while cur is not None:
        path.append(cur); cur = came[cur]
    return path[::-1]

# Dataset: local patch tokens + action index (0..7)
class ImitationDS(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        tokens, action = self.samples[i]
        return torch.tensor(tokens, dtype=torch.float32), torch.tensor(action, dtype=torch.long)

def extract_patch_tokens(y, x, hazard, mpa, obstacles, goal, patch=9):
    H, W = hazard.shape
    r = patch//2
    # features per cell: [hazard, mpa, obstacle, rel_y, rel_x, goal_dy, goal_dx, dist_to_goal]
    feats = []
    gy, gx = goal
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                h = float(hazard[ny,nx])
                m = 1.0 if mpa[ny,nx] else 0.0
                o = 1.0 if obstacles[ny,nx] else 0.0
            else:
                h, m, o = 1.0, 0.0, 1.0  # out of bounds = blocked & high hazard
            rel_y = dy / max(1, r)
            rel_x = dx / max(1, r)
            goal_dy = (gy - y) / max(1, H)
            goal_dx = (gx - x) / max(1, W)
            dist_goal = math.hypot(gy - y, gx - x) / math.hypot(H, W)
            feats.append([h, m, o, rel_y, rel_x, goal_dy, goal_dx, dist_goal])
    return np.array(feats, dtype=np.float32)

def build_dataset(hazard, mpa, obstacles, start_goal_pairs, w=(1.0,1.5,8.0), patch=9, diag=True):
    samples = []
    H, W = hazard.shape
    for (start, goal) in start_goal_pairs:
        path = astar(start, goal, hazard, mpa, obstacles, w_dist=w[0], w_hazard=w[1], w_mpa=w[2], diag=diag)
        if not path or len(path) < 2:
            continue
        for i in range(len(path)-1):
            y, x = path[i]
            y2, x2 = path[i+1]
            # map move to action index in MOVES8
            move = (y2-y, x2-x)
            action_idx = MOVES8.index(move)
            tokens = extract_patch_tokens(y, x, hazard, mpa, obstacles, goal, patch=patch)
            samples.append((tokens, action_idx))
    return samples

def main():
    hazard = np.load(DATA / "hazard.npy")
    cfg = json.loads((DATA / "map.json").read_text())
    H, W = cfg["height"], cfg["width"]
    obstacles = np.zeros((H, W), dtype=bool)
    for (y, x) in cfg["obstacles"]:
        obstacles[y, x] = True
    mpa = np.zeros((H, W), dtype=bool)
    for m in cfg["mpas"]:
        mpa[m["y1"]:m["y2"], m["x1"]:m["x2"]] = True

    # Generate diverse start/goal pairs
    rng = np.random.default_rng(7)
    pairs = []
    for _ in range(500):
        sy, sx = int(rng.integers(1, H-1)), int(rng.integers(1, W//4))
        gy, gx = int(rng.integers(1, H-1)), int(rng.integers(3*W//4, W-1))
        pairs.append(((sy, sx), (gy, gx)))

    samples = build_dataset(hazard, mpa, obstacles, pairs, w=(1.0,1.5,8.0), patch=9, diag=True)
    rng = np.random.default_rng(42)
    rng.shuffle(samples)
    n = len(samples); tr = int(n*0.9)
    train_s, val_s = samples[:tr], samples[tr:]

    tr_ds = ImitationDS(train_s); va_ds = ImitationDS(val_s)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PolicyConfig()
    model = TransformerPolicy(in_dim=8, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best = (1e9, None)
    for epoch in range(8):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        model.eval(); val_loss=0.0; nval=0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item() * len(yb)
                nval += len(yb)
        v = val_loss / nval
        if v < best[0]:
            best = (v, model.state_dict())

    if best[1] is not None:
        model.load_state_dict(best[1])

    # Save
    torch.save(model.state_dict(), ART / "transformer_policy.pt")
    import joblib
    joblib.dump({"patch": cfg.patch, "actions": cfg.actions}, ART / "transformer_policy_meta.joblib")
    # quick metric: accuracy on val
    correct=0; total=0
    with torch.no_grad():
        for xb, yb in va_loader:
            xb=xb.to(device); yb=yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred==yb).sum().item()
            total += len(yb)
    acc = correct/total if total else 0.0
    import pandas as pd
    pd.DataFrame([{"metric":"val_acc","value":acc}]).to_csv(ART / "transformer_policy_report.csv", index=False)

if __name__ == "__main__":
    main()
