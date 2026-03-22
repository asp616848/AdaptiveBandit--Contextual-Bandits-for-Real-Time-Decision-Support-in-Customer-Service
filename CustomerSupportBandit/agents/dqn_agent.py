"""
DQN Agent for full MDP customer support dialogue.

Implements Deep Q-Network with experience replay and target network
for the multi-turn dialogue MDP with actions:
{ask_info, provide_solution, escalate, close}.

This is Phase II of the proposal — used for Business+/Enterprise tiers
where multi-turn planning provides value over single-step bandits.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, List
import random

from .base_agent import BaseAgent

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DQN_CONFIG, MDP_ACTIONS, NUM_MDP_ACTIONS


# ═══════════════════════════════════════════════════════════
# Neural Network (pure NumPy implementation — no PyTorch needed)
# ═══════════════════════════════════════════════════════════

class NumpyMLP:
    """Simple MLP implemented in pure NumPy for portability."""

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, lr: float = 1e-3):
        self.lr = lr
        self.layers = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            # He initialization
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.layers.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass with ReLU activations (linear output)."""
        activations = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.layers, self.biases)):
            z = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(0, z)  # ReLU
            else:
                h = z  # Linear output
            activations.append(h)
        return h, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning Q-values."""
        out, _ = self.forward(x)
        return out

    def train_step(self, x: np.ndarray, targets: np.ndarray,
                   actions: np.ndarray) -> float:
        """
        One training step with MSE loss on selected action Q-values.

        Parameters
        ----------
        x : np.ndarray, shape (batch, input_dim)
        targets : np.ndarray, shape (batch,)  — target Q-values
        actions : np.ndarray, shape (batch,)  — action indices

        Returns
        -------
        float
            MSE loss.
        """
        batch_size = x.shape[0]

        # Forward
        q_values, activations = self.forward(x)

        # Compute loss gradient only for selected actions
        q_selected = q_values[np.arange(batch_size), actions.astype(int)]
        td_errors = q_selected - targets
        loss = np.mean(td_errors ** 2)

        # Gradient of output w.r.t. selected actions
        grad_output = np.zeros_like(q_values)
        grad_output[np.arange(batch_size), actions.astype(int)] = \
            2.0 * td_errors / batch_size

        # Backprop through layers
        grad = grad_output
        for i in range(len(self.layers) - 1, -1, -1):
            h_in = activations[i]

            # Gradient for weights and biases
            dw = h_in.T @ grad
            db = np.sum(grad, axis=0)

            # Gradient for input (to pass to previous layer)
            grad = grad @ self.layers[i].T

            # ReLU gradient (except for output layer)
            if i > 0:
                grad = grad * (activations[i] > 0).astype(float)

            # Update weights
            self.layers[i] -= self.lr * np.clip(dw, -1, 1)
            self.biases[i] -= self.lr * np.clip(db, -1, 1)

        return loss

    def copy_from(self, other: 'NumpyMLP'):
        """Copy weights from another network."""
        for i in range(len(self.layers)):
            self.layers[i] = other.layers[i].copy()
            self.biases[i] = other.biases[i].copy()


# ═══════════════════════════════════════════════════════════
# Experience Replay Buffer
# ═══════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-size experience replay buffer with uniform sampling."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=float),
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════
# DQN Agent
# ═══════════════════════════════════════════════════════════

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for the full MDP formulation.

    MDP: M = <S, A, P, R, gamma>
    - S: conversational state (feature vector)
    - A: {ask_info, provide_solution, escalate, close}
    - R: economic reward (tier-specific)
    - gamma: discount factor
    """

    def __init__(self, feature_dim: int = 23,
                 config: Optional[Dict] = None):
        n_actions = NUM_MDP_ACTIONS
        super().__init__(n_actions, feature_dim, name="DQN")

        cfg = config or DQN_CONFIG

        # Networks
        self.q_network = NumpyMLP(
            feature_dim, cfg['hidden_dims'], n_actions, cfg['learning_rate']
        )
        self.target_network = NumpyMLP(
            feature_dim, cfg['hidden_dims'], n_actions, cfg['learning_rate']
        )
        self.target_network.copy_from(self.q_network)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(cfg['buffer_size'])

        # Hyperparameters
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batch_size = cfg['batch_size']
        self.target_update_freq = cfg['target_update_freq']

        self.train_step_count = 0
        self.losses = []

    def select_action(self, context: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        """
        self.t += 1

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            x = context.reshape(1, -1)
            q_values = self.q_network.predict(x)[0]
            action = int(np.argmax(q_values))

        self.action_counts[action] += 1
        return action

    def update(self, context: np.ndarray, action: int, reward: float,
               next_context: Optional[np.ndarray] = None,
               done: bool = False) -> Optional[float]:
        """
        Store transition and train if buffer has enough samples.

        Parameters
        ----------
        context : np.ndarray
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_context : np.ndarray, optional
            Next state. If None, treated as terminal.
        done : bool
            Whether episode ended.

        Returns
        -------
        float or None
            Training loss if a training step occurred.
        """
        self.total_reward += reward
        self.reward_history.append(reward)

        if next_context is None:
            next_context = np.zeros_like(context)
            done = True

        self.replay_buffer.push(context, action, reward, next_context, done)

        # Train if enough samples
        loss = None
        if len(self.replay_buffer) >= self.batch_size:
            loss = self._train_step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end,
                          self.epsilon * self.epsilon_decay)

        return loss

    def _train_step(self) -> float:
        """Single training step from replay buffer."""
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Compute targets: r + gamma * max_a' Q_target(s', a') * (1 - done)
        next_q = self.target_network.predict(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Train Q-network
        loss = self.q_network.train_step(states, targets, actions)
        self.losses.append(loss)

        self.train_step_count += 1

        # Update target network periodically
        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)

        return loss

    def get_q_values(self, context: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a context."""
        return self.q_network.predict(context.reshape(1, -1))[0]

    def get_policy_info(self) -> Dict:
        info = super().get_policy_info()
        info['epsilon'] = self.epsilon
        info['buffer_size'] = len(self.replay_buffer)
        info['train_steps'] = self.train_step_count
        info['avg_loss'] = np.mean(self.losses[-100:]) if self.losses else 0.0
        return info

    def reset(self) -> None:
        super().reset()
        self.epsilon = DQN_CONFIG['epsilon_start']
        self.train_step_count = 0
        self.losses = []
        self.replay_buffer = ReplayBuffer(DQN_CONFIG['buffer_size'])
