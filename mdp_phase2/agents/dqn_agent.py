"""
DQN Agent — Phase II RL Prompt Selector
=========================================
Deep Q-Network with:
  • Experience replay buffer (random sampling)
  • Separate online + target networks
  • Epsilon-greedy exploration with linear decay
  • Huber loss (smooth L1)
  • Periodic hard target-network update

Architecture (PyTorch):
  Linear(state_dim, 128) → ReLU
  Linear(128, 64)        → ReLU
  Linear(64, num_actions)          (Q-values, no activation)

Reference: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
"""

from __future__ import annotations
import numpy as np
import random
from collections import deque


# ── Try PyTorch; fall back to numpy-only minimal network ─────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _QNetwork(nn.Module):
        """2-hidden-layer MLP producing Q-values for all actions."""

        def __init__(self, state_dim: int, num_actions: int,
                     hidden: tuple[int, ...] = (128, 64)):
            super().__init__()
            layers = []
            in_dim = state_dim
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, num_actions))
            self.net = nn.Sequential(*layers)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    BACKEND = 'torch'
except ImportError:
    BACKEND = 'numpy'


class _NumpyQNetwork:
    """Minimal 2-hidden-layer MLP in pure numpy (no external DL framework)."""

    def __init__(self, state_dim: int, num_actions: int,
                 hidden: tuple[int, ...] = (128, 64), lr: float = 1e-3):
        self.lr = lr
        dims = [state_dim] + list(hidden) + [num_actions]
        self.W = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0/dims[i])
                  for i in range(len(dims)-1)]
        self.b = [np.zeros((1, dims[i+1])) for i in range(len(dims)-1)]

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w + b
            if i < len(self.W) - 1:
                x = np.maximum(0, x)   # ReLU
        return x   # shape: (batch, num_actions)

    def train_on_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        """Simple SGD with MSE loss (sufficient for demonstration)."""
        bs = x.shape[0]
        # forward
        acts, z = [x], []
        cur = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z_i = cur @ w + b
            z.append(z_i)
            cur = np.maximum(0, z_i) if i < len(self.W)-1 else z_i
            acts.append(cur)
        pred = acts[-1]
        loss = np.mean((pred - y) ** 2)
        # backward
        delta = 2.0 * (pred - y) / bs
        for i in reversed(range(len(self.W))):
            dW = acts[i].T @ delta
            db = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = delta @ self.W[i].T
                delta[acts[i] <= 0] = 0
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db
        return float(loss)

    def get_weights(self):
        return [(w.copy(), b.copy()) for w, b in zip(self.W, self.b)]

    def set_weights(self, weights):
        for i, (w, b) in enumerate(weights):
            self.W[i] = w.copy()
            self.b[i] = b.copy()


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self.buf.append((s, a, r, s_, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(s_, dtype=np.float32),
                np.array(d,  dtype=bool))

    def __len__(self):
        return len(self.buf)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Parameters
    ----------
    state_dim   : dimensionality of the state vector (15 for Phase II env)
    num_actions : number of discrete actions (5 strategies)
    gamma       : discount factor
    lr          : learning rate
    eps_start   : initial ε for ε-greedy exploration
    eps_end     : minimum ε
    eps_decay   : ε is decreased by this amount each training step
    batch_size  : replay batch size
    buffer_size : maximum replay buffer length
    target_update_freq : hard-copy online → target every N gradient steps
    hidden      : sizes of hidden layers
    """

    def __init__(
        self,
        state_dim:          int   = 15,
        num_actions:        int   = 5,
        gamma:              float = 0.90,
        lr:                 float = 1e-3,
        eps_start:          float = 1.0,
        eps_end:            float = 0.05,
        eps_decay:          float = 0.002,
        batch_size:         int   = 64,
        buffer_size:        int   = 10_000,
        target_update_freq: int   = 50,
        hidden:             tuple = (128, 64),
    ):
        self.state_dim          = state_dim
        self.num_actions        = num_actions
        self.gamma              = gamma
        self.eps                = eps_start
        self.eps_end            = eps_end
        self.eps_decay          = eps_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.hidden             = hidden

        self.replay      = ReplayBuffer(buffer_size)
        self.train_steps = 0

        # build networks
        if BACKEND == 'torch':
            self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.online    = _QNetwork(state_dim, num_actions, hidden).to(self.device)
            self.target    = _QNetwork(state_dim, num_actions, hidden).to(self.device)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
            self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        else:
            self.online = _NumpyQNetwork(state_dim, num_actions, hidden, lr)
            self.target = _NumpyQNetwork(state_dim, num_actions, hidden, lr)
            self.target.set_weights(self.online.get_weights())

    # ── Action selection ──────────────────────────────────────────────────

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and random.random() < self.eps:
            return random.randrange(self.num_actions)
        q = self._predict_online(state)
        return int(np.argmax(q))

    def act_greedy(self, state: np.ndarray) -> int:
        """Always pick the greedy action (used at evaluation time)."""
        return int(np.argmax(self._predict_online(state)))

    # ── Learning ──────────────────────────────────────────────────────────

    def remember(self, s, a, r, s_, done):
        self.replay.push(s, a, r, s_, done)

    def learn(self) -> float | None:
        """One gradient update step; returns loss or None if buffer too small."""
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, s_, done = self.replay.sample(self.batch_size)

        if BACKEND == 'torch':
            loss = self._torch_train_step(s, a, r, s_, done)
        else:
            # Bellman targets
            q_next = self.target.predict(s_)
            q_target_vals = r + self.gamma * np.max(q_next, axis=1) * (~done)
            q_pred = self.online.predict(s)
            q_pred[np.arange(self.batch_size), a] = q_target_vals.astype(np.float32)
            loss = self.online.train_on_batch(s, q_pred)

        # decay ε
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        # periodic target update
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self._copy_to_target()

        return loss

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        if BACKEND == 'torch':
            torch.save({
                'online': self.online.state_dict(),
                'target': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, path + '_dqn.pt')
        else:
            import pickle
            with open(path + '_weights.pkl', 'wb') as f:
                pickle.dump(self.online.get_weights(), f)

    def load(self, path: str):
        if BACKEND == 'torch':
            ckpt = torch.load(path + '_dqn.pt', map_location=self.device)
            self.online.load_state_dict(ckpt['online'])
            self.target.load_state_dict(ckpt['target'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            import pickle
            with open(path + '_weights.pkl', 'rb') as f:
                w = pickle.load(f)
            self.online.set_weights(w)
            self.target.set_weights(w)

    # ── Private helpers ───────────────────────────────────────────────────

    def _torch_train_step(self, s, a, r, s_, done) -> float:
        s_t    = torch.as_tensor(s,    dtype=torch.float32, device=self.device)
        a_t    = torch.as_tensor(a,    dtype=torch.long,    device=self.device)
        r_t    = torch.as_tensor(r,    dtype=torch.float32, device=self.device)
        s_t_   = torch.as_tensor(s_,   dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # current Q(s, a)
        q_all  = self.online(s_t)
        q_pred = q_all.gather(1, a_t.unsqueeze(1)).squeeze(1)

        # target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target(s_t_).max(dim=1).values
            q_tgt  = r_t + self.gamma * q_next * (1.0 - done_t)

        loss = F.smooth_l1_loss(q_pred, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def _predict_online(self, s: np.ndarray) -> np.ndarray:
        if BACKEND == 'torch':
            s_t = torch.as_tensor(s.reshape(1, -1), dtype=torch.float32,
                                  device=self.device)
            with torch.no_grad():
                return self.online(s_t).cpu().numpy().flatten()
        return self.online.predict(s).flatten()

    def _copy_to_target(self):
        if BACKEND == 'torch':
            self.target.load_state_dict(self.online.state_dict())
        else:
            self.target.set_weights(self.online.get_weights())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions (for visualisation)."""
        return self._predict_online(state)

    @property
    def name(self) -> str:
        return "DQN"
