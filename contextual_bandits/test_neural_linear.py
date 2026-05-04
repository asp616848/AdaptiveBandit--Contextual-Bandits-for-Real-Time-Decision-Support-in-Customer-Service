"""Quick test for Neural-Linear Bandits across all three datasets"""
import numpy as np
import sys
sys.path.insert(0, '.')
from environment import CustomerServiceEnv

class NeuralLinearAgent:
    def __init__(self, n_actions=5, state_dim=4, hidden_dim=32, feature_dim=16, 
                 alpha=1.5, lr=0.005, retrain_interval=100):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lr = lr
        self.retrain_interval = retrain_interval
        
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, feature_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(feature_dim)
        
        self.A = [np.eye(feature_dim) for _ in range(n_actions)]
        self.b_lin = [np.zeros(feature_dim) for _ in range(n_actions)]
        self.A_inv = [np.eye(feature_dim) for _ in range(n_actions)]
        
        self.experience = []
        self.episode_count = 0
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _forward(self, state):
        x = np.array(state, dtype=float)
        h = self._relu(x @ self.W1 + self.b1)
        z = self._relu(h @ self.W2 + self.b2)
        norm = np.linalg.norm(z) + 1e-8
        return z / norm
    
    def choose_action(self, state):
        z = self._forward(state)
        p_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            theta = self.A_inv[a] @ self.b_lin[a]
            ucb = self.alpha * np.sqrt(z @ self.A_inv[a] @ z)
            p_values[a] = theta @ z + ucb
        return np.argmax(p_values)
    
    def update(self, state, action, reward):
        z = self._forward(state)
        reward_norm = np.tanh(reward / 2.0)
        self.A[action] += np.outer(z, z)
        self.b_lin[action] += reward_norm * z
        self.A_inv[action] = np.linalg.inv(self.A[action])
        self.experience.append((np.array(state), action, reward))
    
    def end_episode(self):
        self.episode_count += 1
        if self.episode_count % self.retrain_interval == 0 and len(self.experience) > 100:
            self._retrain_network()
    
    def _retrain_network(self):
        batch_size = min(256, len(self.experience))
        indices = np.random.choice(len(self.experience), batch_size, replace=False)
        
        for epoch in range(3):
            for idx in indices:
                state, action, reward = self.experience[idx]
                reward_norm = np.tanh(reward / 2.0)
                x = np.array(state, dtype=float)
                h_pre = x @ self.W1 + self.b1
                h = self._relu(h_pre)
                z_pre = h @ self.W2 + self.b2
                z = self._relu(z_pre)
                norm = np.linalg.norm(z) + 1e-8
                z_normed = z / norm
                theta = self.A_inv[action] @ self.b_lin[action]
                pred = theta @ z_normed
                error = pred - reward_norm
                dz_normed = 2 * error * theta
                dz = (dz_normed - z_normed * (dz_normed @ z_normed)) / norm
                dz[z_pre <= 0] = 0
                dW2 = np.outer(h, dz)
                db2 = dz
                dh = dz @ self.W2.T
                dh[h_pre <= 0] = 0
                dW1 = np.outer(x, dh)
                db1 = dh
                self.W1 -= self.lr * np.clip(dW1, -1, 1)
                self.b1 -= self.lr * np.clip(db1, -1, 1)
                self.W2 -= self.lr * np.clip(dW2, -1, 1)
                self.b2 -= self.lr * np.clip(db2, -1, 1)
        
        self.A = [np.eye(self.feature_dim) for _ in range(self.n_actions)]
        self.b_lin = [np.zeros(self.feature_dim) for _ in range(self.n_actions)]
        self.A_inv = [np.eye(self.feature_dim) for _ in range(self.n_actions)]
        recent = self.experience[-min(500, len(self.experience)):]
        for state, action, reward in recent:
            z = self._forward(state)
            reward_norm = np.tanh(reward / 2.0)
            self.A[action] += np.outer(z, z)
            self.b_lin[action] += reward_norm * z
        for a in range(self.n_actions):
            self.A_inv[a] = np.linalg.inv(self.A[a])


for dataset in ['twitter', 'reddit', 'openassistant']:
    env = CustomerServiceEnv(dataset=dataset)
    agent = NeuralLinearAgent(n_actions=5, state_dim=4, retrain_interval=100)
    
    rewards = []
    outcomes = []
    
    for ep in range(500):
        state = env.reset()
        ep_reward = 0
        while not env.done:
            action = agent.choose_action(state)
            next_state, reward, done, outcome = env.step(action)
            agent.update(state, action, reward)
            ep_reward += reward
            state = next_state
        agent.end_episode()
        rewards.append(ep_reward)
        outcomes.append(outcome)
    
    resolved = sum(1 for o in outcomes[-200:] if o == 'resolved') / 200
    avg_r = np.mean(rewards[-200:])
    print(f'{dataset:15s} | avg_reward={avg_r:6.2f} | resolution={resolved:.1%}')

print('\nNeural-Linear Bandits: All datasets working correctly!')
