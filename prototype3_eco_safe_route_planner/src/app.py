import json
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import heapq
import pandas as pd

from rl_agent import RLConfig, train_qlearning, greedy_path, GridEnv
from transformer_policy import TransformerPolicy, PolicyConfig
import joblib, torch
from rag_utils import retrieve, answer_llm, ensure_index

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="Eco-Safe Route Planner (A* + RL + Transformer)", layout="wide")
st.title("Prototype 3: Eco-Safe Route Planner (A* + Reinforcement Learning + Transformer Imitation)")

st.markdown("""
Multi-objective path planning that balances **distance**, **hazard**, and **ecological impact** using:
- **A\\*** optimisation with a hand-crafted cost,
- **Reinforcement Learning (Q-learning)** that learns a policy from rewards, and
- **Transformer Policy (Imitation)** that learns to copy an A* expert from local observations.
Use the sliders to adjust weights and compare routes.
""")

# Load map + layers
hazard = np.load(DATA / "hazard.npy")
cfg = json.loads((DATA / "map.json").read_text())
H, W = cfg["height"], cfg["width"]
start = tuple(cfg["start"])  # (y,x)
goal  = tuple(cfg["goal"])

# Build masks
obstacles = np.zeros((H, W), dtype=bool)
for (y, x) in cfg["obstacles"]:
    obstacles[y, x] = True

mpa_mask = np.zeros((H, W), dtype=bool)
for m in cfg["mpas"]:
    mpa_mask[m["y1"]:m["y2"], m["x1"]:m["x2"]] = True

# Sidebar weights
st.sidebar.header("Objective Weights")
w_dist = st.sidebar.slider("Distance weight", 0.0, 2.0, 1.0, 0.1)
w_hazard = st.sidebar.slider("Hazard weight", 0.0, 5.0, 1.5, 0.1)
w_mpa = st.sidebar.slider("MPA penalty weight", 0.0, 20.0, 8.0, 0.5)
diag = st.sidebar.checkbox("Allow diagonal moves", value=True)

# Tabs for planners
tab1, tab2, tab3 = st.tabs(["A* Optimisation", "RL (Q-learning)", "Transformer Policy (Imitation)"])

# ---------- A* implementation ----------
if diag:
    MOVES = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
else:
    MOVES = [(1,0),(-1,0),(0,1),(0,-1)]

def in_bounds(y,x): return 0 <= y < H and 0 <= x < W

def cost_fn(y,x):
    base = w_dist
    hz = w_hazard * hazard[y,x]
    eco = w_mpa * (1.0 if mpa_mask[y,x] else 0.0)
    return base + hz + eco

def heuristic(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def astar(start, goal):
    if obstacles[start] or obstacles[goal]:
        return None, np.inf
    pq = []
    heapq.heappush(pq, (0.0, start))
    came = {start: None}
    g = {start: 0.0}
    while pq:
        f, cur = heapq.heappop(pq)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path, g[path[-1]]
        cy, cx = cur
        for dy, dx in MOVES:
            ny, nx = cy+dy, cx+dx
            if not in_bounds(ny,nx) or obstacles[ny,nx]:
                continue
            step_cost = cost_fn(ny, nx)
            ng = g[cur] + step_cost
            if (ny, nx) not in g or ng < g[(ny,nx)]:
                g[(ny,nx)] = ng
                came[(ny,nx)] = cur
                hf = ng + heuristic((ny,nx), goal)
                heapq.heappush(pq, (hf, (ny,nx)))
    return None, np.inf

with tab1:
    st.subheader("A* Route Comparison")
    def astar_with_weights(wd, wh, wm):
        global w_dist, w_hazard, w_mpa
        ow_d, ow_h, ow_m = w_dist, w_hazard, w_mpa
        w_dist, w_hazard, w_mpa = wd, wh, wm
        path, cost = astar(start, goal)
        w_dist, w_hazard, w_mpa = ow_d, ow_h, ow_m
        return path, cost

    shortest_path, shortest_cost = astar_with_weights(1.0, 0.0, 0.0)
    safer_path,    safer_cost    = astar_with_weights(1.0, 1.5, 0.0)
    eco_path,      eco_cost      = astar_with_weights(1.0, 1.5, 8.0)
    custom_path,   custom_cost   = astar(start, goal)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(hazard, cmap="inferno", origin="upper", alpha=0.6)
    oy, ox = np.where(obstacles); ax.scatter(ox, oy, s=6, c="black", label="Obstacles")
    mpy, mpx = np.where(mpa_mask); ax.scatter(mpx, mpy, s=6, c="deepskyblue", alpha=0.2, label="MPA")

    def draw_path(path, label, lw, color):
        if path is None: return
        y = [p[0] for p in path]; x=[p[1] for p in path]
        ax.plot(x, y, linewidth=lw, label=label, color=color)

    draw_path(shortest_path, f"Shortest (cost {shortest_cost:.1f})", 2.5, "white")
    draw_path(safer_path,    f"Safer (cost {safer_cost:.1f})",     2.5, "lime")
    draw_path(eco_path,      f"Eco-safe (cost {eco_cost:.1f})",    2.5, "cyan")
    draw_path(custom_path,   f"Custom (cost {custom_cost:.1f})",   3.0, "yellow")

    ax.scatter([start[1]],[start[0]], c="blue", s=60, marker="o")
    ax.scatter([goal[1]],[goal[0]], c="red", s=60, marker="*")
    ax.set_title("A* routes (heat = hazard; cyan dots = MPA)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right")
    st.pyplot(fig)

    def hazard_sum(p):
        return float(np.sum([hazard[y,x] for (y,x) in p])) if p else None
    def summarize(name, p, cost):
        if p is None: return {"Route": name, "Length": None, "EcoCells": None, "HazardSum": None, "Cost": None}
        return {"Route": name, "Length": len(p), "EcoCells": sum(1 for (y,x) in p if mpa_mask[y,x]), "HazardSum": round(hazard_sum(p),1), "Cost": round(cost,1)}
    rows = [
        summarize("Shortest", shortest_path, shortest_cost),
        summarize("Safer", safer_path, safer_cost),
        summarize("Eco-safe", eco_path, eco_cost),
        summarize("Custom", custom_path, custom_cost),
    ]
    st.dataframe(pd.DataFrame(rows))

# ---------- RL (Q-learning) ----------
with tab2:
    st.subheader("Reinforcement Learning (Q-learning)")
    st.markdown("The agent learns a policy by receiving **rewards**: negative per-step cost and a **positive reward** on reaching the goal.")

    # Controls
    alpha = st.slider("Learning rate (alpha)", 0.01, 0.5, 0.1, 0.01)
    gamma = st.slider("Discount (gamma)", 0.80, 0.999, 0.95, 0.01)
    epsilon = st.slider("Exploration (epsilon)", 0.0, 1.0, 0.2, 0.05)
    episodes = st.slider("Episodes", 100, 5000, 800, 100)
    max_steps = st.slider("Max steps per episode", 100, 3000, 600, 50)
    goal_reward = st.slider("Goal reward", 1.0, 50.0, 10.0, 1.0)

    cfg_rl = RLConfig(
        alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes, max_steps=max_steps,
        diag=diag, w_dist=w_dist, w_hazard=w_hazard, w_mpa=w_mpa, goal_reward=goal_reward
    )

    # Try to load cached Q-table that matches current settings
    meta_path = ART / "rl_meta.json"
    q_path = ART / "q_table.npy"
    cached_ok = False
    if meta_path.exists() and q_path.exists():
        meta = json.loads(meta_path.read_text())
        def approx_eq(a,b,tol=1e-6): return abs(a-b) <= tol
        keys = ["alpha","gamma","epsilon","episodes","max_steps","diag","w_dist","w_hazard","w_mpa","goal_reward","H","W"]
        cached_ok = all(k in meta for k in keys) and \
                    approx_eq(meta["alpha"], alpha) and approx_eq(meta["gamma"], gamma) and \
                    approx_eq(meta["epsilon"], epsilon) and meta["episodes"] == episodes and \
                    meta["max_steps"] == max_steps and bool(meta["diag"]) == bool(diag) and \
                    approx_eq(meta["w_dist"], w_dist) and approx_eq(meta["w_hazard"], w_hazard) and \
                    approx_eq(meta["w_mpa"], w_mpa) and approx_eq(meta["goal_reward"], goal_reward) and \
                    meta["H"] == H and meta["W"] == W

    if cached_ok:
        q = np.load(q_path)
        st.info("Loaded cached Q-table for current settings.")
    else:
        if st.button("Train RL agent now"):
            with st.spinner("Training RL agent..."):
                q, env = train_qlearning(hazard, mpa_mask, obstacles, start, goal, cfg_rl)
                np.save(q_path, q)
                meta = {
                    "alpha": alpha, "gamma": gamma, "epsilon": epsilon,
                    "episodes": episodes, "max_steps": max_steps, "diag": diag,
                    "w_dist": w_dist, "w_hazard": w_hazard, "w_mpa": w_mpa,
                    "goal_reward": goal_reward, "H": H, "W": W
                }
                meta_path.write_text(json.dumps(meta, indent=2))
                st.success("Training complete and Q-table cached.")
        else:
            st.warning("No cached Q-table for these settings. Click **Train RL agent now** to train.")

    # If any Q-table exists, plot its greedy path (so training in-session shows immediately)
    if (ART / "q_table.npy").exists():
        env = GridEnv(hazard, mpa_mask, obstacles, start, goal, cfg_rl)
        q = np.load(q_path)
        path = greedy_path(q, env)

        # Plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.imshow(hazard, cmap="inferno", origin="upper", alpha=0.6)
        oy, ox = np.where(obstacles); ax2.scatter(ox, oy, s=6, c="black", label="Obstacles")
        mpy, mpx = np.where(mpa_mask); ax2.scatter(mpx, mpy, s=6, c="deepskyblue", alpha=0.2, label="MPA")
        if path:
            y = [p[0] for p in path]; x=[p[1] for p in path]
            ax2.plot(x, y, linewidth=3.0, color="yellow", label="RL (greedy)")
        ax2.scatter([start[1]],[start[0]], c="blue", s=60, marker="o")
        ax2.scatter([goal[1]],[goal[0]], c="red", s=60, marker="*")
        ax2.set_title("RL route (greedy from learned Q-table)")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

        def hazard_sum(p):
            return float(np.sum([hazard[y,x] for (y,x) in p])) if p else None

        rows = [{
            "Route": "RL (greedy)",
            "Length": len(path) if path else None,
            "EcoCells": sum(1 for (y,x) in path if mpa_mask[y,x]) if path else None,
            "HazardSum": round(hazard_sum(path), 1) if path else None
        }]
        st.dataframe(pd.DataFrame(rows))

# ---------- RAG Copilot (LLM + Retrieval) ----------
tab_rag = st.tabs(["RAG Copilot (LLM)"])[0]
with tab_rag:
    st.subheader("RAG Copilot — Ask the map (LLM + Retrieval)")
    st.markdown(
        "Ask natural-language questions. I retrieve the most relevant KB snippets and, "
        "if available, call an LLM to answer with citations."
    )

    # Make/refresh the index (OpenAI+FAISS if possible; else TF-IDF).
    try:
        backend = ensure_index()
        st.caption(f"Retriever backend: {backend.upper()}")
    except Exception as e:
        st.error(f"Index build failed: {e}")
        backend = None

    top_k = st.sidebar.slider("RAG: top_k chunks", 2, 8, 4, 1)
    query = st.text_input("Question", value="How do MPAs affect route planning and safety trade-offs?")

    if st.button("Search & (optionally) Ask LLM") and backend is not None:
        hits = retrieve(query, k=top_k)
        if not hits:
            st.warning("No results retrieved.")
        else:
            st.markdown("**Retrieved context:**")
            for h in hits:
                st.write(f"- `{h['meta']['source']}#{h['meta']['chunk_id']}` (score={h['score']:.3f})")
                st.code(h["text"][:600])

            # LLM answer (if OPENAI_API_KEY set); otherwise shows fallback text
            ans = answer_llm(query, hits)
            st.markdown("### Answer")
            st.write(ans)



# ---------- Transformer Policy (Imitation) ----------
with tab3:
    st.subheader("Transformer Policy (Behavioral Cloning of A*)")
    st.markdown(
        "A small Transformer consumes a **local patch** around the agent and predicts the **next move** "
        "(8 directions), trained on A* expert trajectories with eco-safe weights."
    )

    # Artifacts
    meta_path = ART / "transformer_policy_meta.joblib"
    state_path = ART / "transformer_policy.pt"

    if not (meta_path.exists() and state_path.exists()):
        st.warning("No Transformer policy found. Train it (from A* demos) to enable the visualisation.")
        if st.button("Train Transformer policy now"):
            import runpy
            with st.spinner("Training Transformer policy from A* demonstrations..."):
                runpy.run_path(str(ROOT / "src" / "train_transformer_policy.py"))
            st.success("Training complete. Reloading artifacts...")
            st.rerun()
    else:
        # Load policy
        meta = joblib.load(meta_path)
        cfg_pol = PolicyConfig()
        model = TransformerPolicy(in_dim=8, cfg=cfg_pol)
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        model.eval()

        # Settings
        patch = int(meta.get("patch", 9))
        MOVES8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

        def extract_patch_tokens(y, x, goal):
            r = patch // 2
            feats = []
            gy, gx = goal
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W:
                        h = float(hazard[ny, nx])
                        m = 1.0 if mpa_mask[ny, nx] else 0.0
                        o = 1.0 if obstacles[ny, nx] else 0.0
                    else:
                        h, m, o = 1.0, 0.0, 1.0  # out of bounds => high hazard, blocked
                    rel_y = dy / max(1, r)
                    rel_x = dx / max(1, r)
                    goal_dy = (goal[0] - y) / max(1, H)
                    goal_dx = (goal[1] - x) / max(1, W)
                    dist_goal = float(np.hypot(goal[0]-y, goal[1]-x) / np.hypot(H, W))
                    feats.append([h, m, o, rel_y, rel_x, goal_dy, goal_dx, dist_goal])
            return torch.tensor([feats], dtype=torch.float32)

        def rollout(max_steps=2000):
            path = [start]
            visited = {start}
            s = start
            for _ in range(max_steps):
                if s == goal:
                    break
                x_tokens = extract_patch_tokens(s[0], s[1], goal)
                with torch.no_grad():
                    logits = model(x_tokens)
                    order = torch.argsort(logits, dim=1, descending=True).squeeze(0).tolist()
                moved = False
                for a in order:
                    dy, dx = MOVES8[int(a)]
                    ny, nx = s[0] + dy, s[1] + dx
                    if 0 <= ny < H and 0 <= nx < W and not obstacles[ny, nx]:
                        s = (ny, nx)
                        if s in visited:
                            # loop detected -> stop
                            moved = False
                        else:
                            visited.add(s)
                            path.append(s)
                            moved = True
                        break
                if not moved:
                    break
            return path

        path = rollout()

        # Plot path over the same background style
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.imshow(hazard, cmap="inferno", origin="upper", alpha=0.6)
        oy, ox = np.where(obstacles); ax3.scatter(ox, oy, s=6, c="black", label="Obstacles")
        mpy, mpx = np.where(mpa_mask); ax3.scatter(mpx, mpy, s=6, c="deepskyblue", alpha=0.2, label="MPA")

        if path:
            y = [p[0] for p in path]; x = [p[1] for p in path]
            ax3.plot(x, y, linewidth=3.0, label="Transformer (greedy)", color="orange")

        ax3.scatter([start[1]], [start[0]], c="blue", s=60, marker="o")
        ax3.scatter([goal[1]], [goal[0]], c="red", s=60, marker="*")
        ax3.set_title("Transformer policy route (imitation of A*)")
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.legend(loc="lower right")
        st.pyplot(fig3)

        # Summary (same style as RL)
        def hazard_sum(p):
            return float(np.sum([hazard[yy, xx] for (yy, xx) in p])) if p else None
        rows = [{
            "Route": "Transformer (greedy)",
            "Length": len(path) if path else None,
            "EcoCells": sum(1 for (yy, xx) in path if mpa_mask[yy, xx]) if path else None,
            "HazardSum": round(hazard_sum(path), 1) if path else None
        }]
        st.dataframe(pd.DataFrame(rows))