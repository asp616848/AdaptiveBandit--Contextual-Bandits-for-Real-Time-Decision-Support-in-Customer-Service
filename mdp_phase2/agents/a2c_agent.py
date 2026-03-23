"""
A2C Agent — Phase II RL Prompt Selector
=========================================
Advantage Actor-Critic (synchronous, single-worker):

  • Shared encoder trunk (2 hidden layers)
  • Actor head  → softmax policy π(a | s)
  • Critic head → scalar value estimate V(s)
  • n-step returns for lower variance
  • Entropy regularisation to prevent premature convergence
  • Gradient clipping for stable training

Loss:
  L = L_policy + c_v * L_value - c_e * H(π)

  L_policy = -E[log π(a|s) * A(s,a)]      (policy gradient)
  L_value  = MSE(V(s), R_t)                (critic regression)
  H(π)     = -Σ π(a) log π(a)             (entropy bonus)

Reference: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
"""

from __future__ import annotations
import numpy as np
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _ActorCriticNet(nn.Module):
        """Shared-trunk Actor-Critic network."""

        def __init__(self, state_dim: int, num_actions: int,
                     hidden: tuple[int, ...] = (128, 64)):
            super().__init__()
            layers = []
            in_dim = state_dim
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.trunk = nn.Sequential(*layers)
            self.actor  = nn.Linear(in_dim, num_actions)
            self.critic = nn.Linear(in_dim, 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor):
            h      = self.trunk(x)
            logits = self.actor(h)
            probs  = torch.softmax(logits, dim=-1)
            value  = self.critic(h).squeeze(-1)
            return probs, value

    BACKEND = 'torch'
except ImportError:
    BACKEND = 'numpy'


class _NumpyActorCritic:
    """Minimal shared-trunk Actor-Critic in pure numpy."""

    def __init__(self, state_dim: int, num_actions: int,
                 hidden: tuple[int, ...] = (128, 64), lr: float = 5e-4):
        self.lr   = lr
        self.na   = num_actions
        dims = [state_dim] + list(hidden)
        # shared trunk
        self.Ws = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0/dims[i])
                   for i in range(len(dims)-1)]
        self.bs = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]
        # actor (policy logits)
        self.Wa = np.random.randn(dims[-1], num_actions) * 0.01
        self.ba = np.zeros(num_actions)
        # critic
        self.Wc = np.random.randn(dims[-1], 1) * 0.01
        self.bc = np.zeros(1)

    def _forward(self, x):
        for w, b in zip(self.Ws, self.bs):
            x = np.maximum(0, x @ w + b)
        logits = x @ self.Wa + self.ba
        logits -= logits.max(axis=-1, keepdims=True)   # numeric stability
        probs  = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        value  = (x @ self.Wc + self.bc).squeeze(-1)
        return probs, value, x   # x = trunk activations

    def predict(self, state: np.ndarray):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        probs, value, _ = self._forward(state)
        return probs, value

    def train_step(self, states, actions, advantages, returns,
                   c_value=0.5, c_entropy=0.01):
        probs, values, trunk = self._forward(states)
        bs     = len(states)
        eps    = 1e-8

        # policy gradient loss
        log_p  = np.log(probs + eps)
        log_pa = log_p[np.arange(bs), actions]
        loss_p = -(log_pa * advantages).mean()

        # value loss
        loss_v = 0.5 * ((returns - values) ** 2).mean()

        # entropy
        entropy = -(probs * log_p).sum(axis=1).mean()
        loss    = loss_p + c_value * loss_v - c_entropy * entropy

        # ---- very basic gradient step via finite-differences  ----
        # (sufficient for correctness demo; proper backprop omitted for brevity)
        delta_a = probs.copy()
        delta_a[np.arange(bs), actions] -= 1.0
        delta_a *= (-advantages / bs)[:, None]

        self.Wa -= self.lr * trunk.T @ delta_a
        self.ba -= self.lr * delta_a.mean(axis=0)
        delta_c  = (2.0 * (values - returns) / bs)[:, None]
        self.Wc -= self.lr * trunk.T @ delta_c

        return float(loss)


class A2CAgent:
    """
    Parameters
    ----------
    state_dim        : state vector size (15)
    num_actions      : number of strategies (5)
    gamma            : discount factor
    lr               : learning rate
    n_steps          : n-step return window
    c_value          : coefficient for value loss
    c_entropy        : entropy bonus coefficient
    gradient_clip    : max gradient norm
    hidden           : hidden layer sizes
    """

    def __init__(
        self,
        state_dim:     int   = 15,
        num_actions:   int   = 5,
        gamma:         float = 0.90,
        lr:            float = 5e-4,
        n_steps:       int   = 8,
        c_value:       float = 0.50,
        c_entropy:     float = 0.02,
        gradient_clip: float = 0.50,
        hidden:        tuple = (128, 64),
    ):
        self.state_dim     = state_dim
        self.num_actions   = num_actions
        self.gamma         = gamma
        self.n_steps       = n_steps
        self.c_value       = c_value
        self.c_entropy     = c_entropy
        self.gradient_clip = gradient_clip
        self.hidden        = hidden
        self.train_steps   = 0

        # rollout buffer for n-step updates
        self._reset_buffer()

        if BACKEND == 'torch':
            self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model     = _ActorCriticNet(state_dim, num_actions, hidden).to(self.device)
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), lr=lr)
        else:
            self._model = _NumpyActorCritic(state_dim, num_actions, hidden, lr)

    # ── Action selection ──────────────────────────────────────────────────

    def act(self, state: np.ndarray, training: bool = True) -> tuple[int, float]:
        """
        Sample action from policy π(·|state).
        Returns (action, log_prob) — log_prob needed for variance diagnostics.
        """
        probs = self._get_probs(state)
        if training:
            action = np.random.choice(self.num_actions, p=probs)
        else:
            action = int(np.argmax(probs))
        lp = float(np.log(probs[action] + 1e-8))
        return action, lp

    def act_greedy(self, state: np.ndarray) -> int:
        return int(np.argmax(self._get_probs(state)))

    # ── Rollout storage ───────────────────────────────────────────────────

    def _reset_buffer(self):
        self._states:  list = []
        self._actions: list = []
        self._rewards: list = []
        self._dones:   list = []

    def store(self, state, action, reward, done):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)

    # ── Learning ──────────────────────────────────────────────────────────

    def learn(self, next_state: np.ndarray) -> float | None:
        """
        Compute n-step returns and do one gradient update.
        Call at the end of each episode OR every n_steps.
        """
        if not self._states:
            return None

        T = len(self._states)
        returns = np.zeros(T, dtype=np.float32)

        # bootstrap value for non-terminal last step
        if not self._dones[-1]:
            _, v_next = self._get_value(next_state)
            R = float(v_next)
        else:
            R = 0.0

        # discounted returns
        for t in reversed(range(T)):
            R = self._rewards[t] + self.gamma * R * (1 - self._dones[t])
            returns[t] = R

        s = np.array(self._states,  dtype=np.float32)
        a = np.array(self._actions, dtype=np.int32)

        if BACKEND == 'torch':
            s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            a_t = torch.as_tensor(a, dtype=torch.long,    device=self.device)
            R_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            loss = self._torch_train_step(s_t, a_t, R_t)
        else:
            _, values = self._model.predict(s)
            advantages = returns - values
            loss = self._model.train_step(
                s, a, advantages, returns, self.c_value, self.c_entropy)

        self._reset_buffer()
        self.train_steps += 1
        return float(loss)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        if BACKEND == 'torch':
            torch.save({
                'model': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, path + '_a2c.pt')
        else:
            import pickle
            with open(path + '_a2c.pkl', 'wb') as f:
                pickle.dump({'Ws': self._model.Ws, 'bs': self._model.bs,
                             'Wa': self._model.Wa, 'ba': self._model.ba,
                             'Wc': self._model.Wc, 'bc': self._model.bc}, f)

    def load(self, path: str):
        if BACKEND == 'torch':
            ckpt = torch.load(path + '_a2c.pt', map_location=self.device)
            self._model.load_state_dict(ckpt['model'])
            self._optimizer.load_state_dict(ckpt['optimizer'])
        else:
            import pickle
            with open(path + '_a2c.pkl', 'rb') as f:
                d = pickle.load(f)
            for k, v in d.items():
                setattr(self._model, k, v)

    # ── Private helpers ───────────────────────────────────────────────────

    def _get_probs(self, state: np.ndarray) -> np.ndarray:
        if BACKEND == 'torch':
            s_t = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32,
                                  device=self.device)
            with torch.no_grad():
                probs, _ = self._model(s_t)
            return probs.cpu().numpy().flatten()
        probs, _ = self._model.predict(state)
        return probs.flatten()

    def _get_value(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        if BACKEND == 'torch':
            s_t = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32,
                                  device=self.device)
            with torch.no_grad():
                probs, value = self._model(s_t)
            return probs.cpu().numpy().flatten(), float(value.item())
        probs, value = self._model.predict(state)
        return probs.flatten(), float(value.flatten()[0])

    def _torch_train_step(self, s_t, a_t, R_t) -> float:
        probs, values = self._model(s_t)

        advantages = (R_t - values).detach()

        # policy loss
        dist      = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(a_t)
        loss_p    = -(log_probs * advantages).mean()

        # value loss
        loss_v = F.mse_loss(values, R_t)

        # entropy bonus
        entropy = dist.entropy().mean()

        loss = loss_p + self.c_value * loss_v - self.c_entropy * entropy

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clip)
        self._optimizer.step()
        return float(loss.item())

    def get_policy(self, state: np.ndarray) -> np.ndarray:
        """Return full action-probability vector (for visualisation)."""
        return self._get_probs(state)

    @property
    def name(self) -> str:
        return "A2C"
