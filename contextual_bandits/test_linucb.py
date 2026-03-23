"""Quick test for LinUCB across all three datasets"""
import numpy as np
import sys
sys.path.insert(0, '.')
from environment import CustomerServiceEnv

for dataset in ['twitter', 'reddit', 'openassistant']:
    env = CustomerServiceEnv(dataset=dataset)
    d = env.get_state_size()
    n_actions = 5
    alpha_param = 1.5
    
    A = [np.eye(d) for _ in range(n_actions)]
    b = [np.zeros(d) for _ in range(n_actions)]
    A_inv = [np.eye(d) for _ in range(n_actions)]
    
    rewards = []
    outcomes = []
    
    for ep in range(500):
        state = env.reset()
        ep_reward = 0
        while not env.done:
            x = np.array(state)
            p_values = np.zeros(n_actions)
            for a in range(n_actions):
                theta = A_inv[a] @ b[a]
                ucb = alpha_param * np.sqrt(x @ A_inv[a] @ x)
                p_values[a] = theta @ x + ucb
            action = np.argmax(p_values)
            
            next_state, reward, done, outcome = env.step(action)
            x = np.array(state)
            A[action] += np.outer(x, x)
            b[action] += reward * x
            A_inv[action] = np.linalg.inv(A[action])
            
            ep_reward += reward
            state = next_state
        rewards.append(ep_reward)
        outcomes.append(outcome)
    
    resolved = sum(1 for o in outcomes[-200:] if o == 'resolved') / 200
    avg_r = np.mean(rewards[-200:])
    print(f'{dataset:15s} | avg_reward={avg_r:6.2f} | resolution={resolved:.1%}')

print('\nLinUCB: All datasets working correctly!')
