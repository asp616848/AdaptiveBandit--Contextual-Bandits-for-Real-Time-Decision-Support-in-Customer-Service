"""
PPO Agent — Phase II RL Prompt Selector
=========================================
Proximal Policy Optimisation (clip variant):

  • Collects T-step rollouts then does K epochs of minibatch updates
  • Clipped surrogate objective prevents destructively large policy updates
  • Generalised Advantage Estimation (GAE) for lower-variance advantages
  • Shared Actor-Critic network (same architecture as A2C)
  • Value function clipping for stable critic updates
  • Entropy bonus to sustain exploration

Clipped objective:
  L^CLIP = E[ min( r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]

  where  r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _PPONet(nn.Module):
        """Shared-trunk Actor-Critic network for PPO (tanh activations, orthogonal init)."""

        def __init__(self, state_dim: int, num_actions: int,
                     hidden: tuple[int, ...] = (128, 64)):
            super().__init__()
            layers = []
            in_dim = state_dim
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.Tanh())
                in_dim = h
            self.trunk  = nn.Sequential(*layers)
            self.actor  = nn.Linear(in_dim, num_actions)
            self.critic = nn.Linear(in_dim, 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
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


# ── Rollout buffer ────────────────────────────────────────────────────────────

class PPORolloutBuffer:
    """Stores a complete rollout for PPO minibatch updates."""

    def __init__(self):
        self.states:     list = []
        self.actions:    list = []
        self.rewards:    list = []
        self.dones:      list = []
        self.values:     list = []
        self.log_probs:  list = []

    def add(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.9, lam: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute discounted returns and GAE advantages.

        GAE: A_t = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}
             where δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        """
        T      = len(self.rewards)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones  = np.array(self.dones,   dtype=np.float32)
        rews   = np.array(self.rewards, dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(T)):
            delta      = rews[t] + gamma * values[t+1] * (1.0 - dones[t]) - values[t]
            gae        = delta + gamma * lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:T]

        # normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Parameters
    ----------
    state_dim      : state dimension (15)
    num_actions    : number of strategies (5)
    gamma          : discount factor
    lam            : GAE lambda (bias-variance trade-off)
    lr             : learning rate
    clip_eps       : PPO clipping parameter ε
    c_value        : value loss coefficient
    c_entropy      : entropy bonus coefficient
    n_epochs       : optimisation epochs per rollout
    batch_size     : minibatch size for each epoch
    gradient_clip  : max gradient norm
    rollout_len    : steps per rollout collection
    hidden         : hidden layer sizes
    """

    def __init__(
        self,
        state_dim:     int   = 15,
        num_actions:   int   = 5,
        gamma:         float = 0.90,
        lam:           float = 0.95,
        lr:            float = 3e-4,
        clip_eps:      float = 0.20,
        c_value:       float = 0.50,
        c_entropy:     float = 0.01,
        n_epochs:      int   = 4,
        batch_size:    int   = 32,
        gradient_clip: float = 0.50,
        rollout_len:   int   = 128,
        hidden:        tuple = (128, 64),
    ):
        self.state_dim     = state_dim
        self.num_actions   = num_actions
        self.gamma         = gamma
        self.lam           = lam
        self.clip_eps      = clip_eps
        self.c_value       = c_value
        self.c_entropy     = c_entropy
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.gradient_clip = gradient_clip
        self.rollout_len   = rollout_len
        self.hidden        = hidden
        self.train_steps   = 0

        self.buffer = PPORolloutBuffer()

        if BACKEND == 'torch':
            self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model     = _PPONet(state_dim, num_actions, hidden).to(self.device)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        else:
            # fall back to a very basic numpy actor-critic (for display purposes)
            try:
                from .a2c_agent import _NumpyActorCritic
            except ImportError:
                from a2c_agent import _NumpyActorCritic
            self._model = _NumpyActorCritic(state_dim, num_actions, hidden, lr)

    # ── Action selection ──────────────────────────────────────────────────

    def act(self, state: np.ndarray,
            training: bool = True) -> tuple[int, float, float]:
        """
        Sample action.
        Returns (action, log_prob, value_estimate).
        """
        probs, value = self._forward(state)
        if training:
            action = np.random.choice(self.num_actions, p=probs)
        else:
            action = int(np.argmax(probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, float(value)

    def act_greedy(self, state: np.ndarray) -> int:
        probs, _ = self._forward(state)
        return int(np.argmax(probs))

    # ── Buffer management ─────────────────────────────────────────────────

    def store(self, state, action, reward, done, value, log_prob):
        self.buffer.add(state, action, reward, done, value, log_prob)

    def buffer_full(self) -> bool:
        return len(self.buffer) >= self.rollout_len

    # ── Learning ──────────────────────────────────────────────────────────

    def learn(self, last_state: np.ndarray) -> float:
        """
        Run PPO update on the current rollout.
        Should be called when the buffer is full OR at episode end.
        """
        _, last_value = self._forward(last_state)

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.lam)

        s_all = np.array(self.buffer.states,    dtype=np.float32)
        a_all = np.array(self.buffer.actions,   dtype=np.int32)
        lp_old= np.array(self.buffer.log_probs, dtype=np.float32)
        T     = len(s_all)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                idx  = indices[start : start + self.batch_size]
                if len(idx) == 0:
                    continue
                loss = self._update_step(
                    s_all[idx], a_all[idx], lp_old[idx],
                    returns[idx], advantages[idx])
                total_loss += loss

        self.buffer.clear()
        self.train_steps += 1
        n_updates = self.n_epochs * max(1, T // self.batch_size)
        return total_loss / max(n_updates, 1)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        if BACKEND == 'torch':
            torch.save({
                'model': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, path + '_ppo.pt')

    def load(self, path: str):
        if BACKEND == 'torch':
            ckpt = torch.load(path + '_ppo.pt', map_location=self.device)
            self._model.load_state_dict(ckpt['model'])
            self._optimizer.load_state_dict(ckpt['optimizer'])

    # ── Private helpers ───────────────────────────────────────────────────

    def _forward(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        if BACKEND == 'torch':
            s_t = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32,
                                  device=self.device)
            with torch.no_grad():
                probs, value = self._model(s_t)
            return probs.cpu().numpy().flatten(), float(value.item())
        probs, value = self._model.predict(state)
        return probs.flatten(), float(value.flatten()[0])

    def _update_step(self, states, actions, old_log_probs,
                     returns, advantages) -> float:
        if BACKEND != 'torch':
            return 0.0   # simplified for numpy path

        s_t    = torch.as_tensor(states,        dtype=torch.float32, device=self.device)
        a_t    = torch.as_tensor(actions,        dtype=torch.long,    device=self.device)
        lp_old = torch.as_tensor(old_log_probs,  dtype=torch.float32, device=self.device)
        R_t    = torch.as_tensor(returns,        dtype=torch.float32, device=self.device)
        adv_t  = torch.as_tensor(advantages,     dtype=torch.float32, device=self.device)

        probs, values = self._model(s_t)
        dist          = torch.distributions.Categorical(probs)
        new_lp        = dist.log_prob(a_t)

        # importance ratio + clipped surrogate
        ratio = torch.exp(new_lp - lp_old)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
        loss_p = -torch.min(surr1, surr2).mean()

        # value loss (clipped)
        v_clipped = torch.clamp(values, R_t - 0.2, R_t + 0.2)
        loss_v = torch.max(
            F.mse_loss(values, R_t, reduction='none'),
            F.mse_loss(v_clipped, R_t, reduction='none')
        ).mean()

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
        probs, _ = self._forward(state)
        return probs

    @property
    def name(self) -> str:
        return "PPO"
