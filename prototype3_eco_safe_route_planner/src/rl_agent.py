import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class RLConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    episodes: int = 800
    max_steps: int = 600
    diag: bool = True
    w_dist: float = 1.0
    w_hazard: float = 1.5
    w_mpa: float = 8.0
    goal_reward: float = 10.0
    seed: int = 123

class GridEnv:
    def __init__(self, hazard, mpa_mask, obstacles, start, goal, cfg: RLConfig):
        self.hazard = hazard
        self.mpa = mpa_mask
        self.obs = obstacles
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.cfg = cfg
        self.H, self.W = hazard.shape
        if cfg.diag:
            self.moves = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        else:
            self.moves = [(1,0),(-1,0),(0,1),(0,-1)]
        self.nA = len(self.moves)

    def in_bounds(self, y, x):
        return 0 <= y < self.H and 0 <= x < self.W

    def step_cost(self, y, x):
        base = self.cfg.w_dist
        hz = self.cfg.w_hazard * float(self.hazard[y, x])
        eco = self.cfg.w_mpa * (1.0 if self.mpa[y, x] else 0.0)
        return base + hz + eco

    def reset(self):
        return self.start

    def step(self, state, action):
        y, x = state
        dy, dx = self.moves[action]
        ny, nx = y + dy, x + dx
        if not self.in_bounds(ny, nx) or self.obs[ny, nx]:
            # invalid move: small penalty, stay in place
            return (y, x), - (self.cfg.w_dist * 1.5), False
        # reward is negative cost per step
        reward = - self.step_cost(ny, nx)
        done = False
        if (ny, nx) == self.goal:
            reward += self.cfg.goal_reward
            done = True
        return (ny, nx), reward, done

def train_qlearning(hazard, mpa_mask, obstacles, start, goal, cfg: RLConfig):
    env = GridEnv(hazard, mpa_mask, obstacles, start, goal, cfg)
    q = np.zeros((env.H, env.W, env.nA), dtype=np.float32)
    rng = np.random.default_rng(cfg.seed)

    for ep in range(cfg.episodes):
        s = env.reset()
        for t in range(cfg.max_steps):
            if rng.random() < cfg.epsilon:
                a = rng.integers(0, env.nA)
            else:
                a = int(np.argmax(q[s[0], s[1], :]))
            s2, r, done = env.step(s, a)
            # Q-learning update
            best_next = np.max(q[s2[0], s2[1], :])
            q[s[0], s[1], a] = (1 - cfg.alpha) * q[s[0], s[1], a] + cfg.alpha * (r + cfg.gamma * best_next)
            s = s2
            if done:
                break
    return q, env

def greedy_path(q, env: GridEnv, max_len=2000):
    path = [env.start]
    s = env.start
    for _ in range(max_len):
        if s == env.goal:
            break
        a = int(np.argmax(q[s[0], s[1], :]))
        s2, r, done = env.step(s, a)
        # prevent loops: if no movement, random kick
        if s2 == s:
            # pick the best legal move
            best = None; best_q = -1e9
            for a2 in range(env.nA):
                dy, dx = env.moves[a2]
                ny, nx = s[0]+dy, s[1]+dx
                if env.in_bounds(ny, nx) and not env.obs[ny, nx]:
                    if q[ny, nx, :].max() > best_q:
                        best_q = q[ny, nx, :].max(); best = (ny, nx)
            if best is None:
                break
            s2 = best
        path.append(s2)
        s = s2
        if done:
            break
    return path
