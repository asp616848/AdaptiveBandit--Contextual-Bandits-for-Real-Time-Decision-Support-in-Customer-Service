"""Quick test to verify notebooks will run correctly"""
import numpy as np
from environment import CustomerServiceEnv

# test all three datasets
for dataset in ['twitter', 'reddit', 'openassistant']:
    env = CustomerServiceEnv(dataset=dataset)
    
    # simple thompson sampling
    n_bins = 4
    n_contexts = n_bins * n_bins
    alpha = np.ones((n_contexts, 5))
    beta_param = np.ones((n_contexts, 5))
    
    rewards = []
    outcomes = []
    
    for ep in range(500):
        state = env.reset()
        ep_reward = 0
        while not env.done:
            s_bin = min(int(state[0]*n_bins), n_bins-1)
            f_bin = min(int(state[1]*n_bins), n_bins-1)
            ctx = s_bin * n_bins + f_bin
            samples = np.random.beta(alpha[ctx], beta_param[ctx])
            action = np.argmax(samples)
            next_state, reward, done, outcome = env.step(action)
            normalized = np.clip((reward + 3) / 8, 0, 1)
            alpha[ctx, action] += normalized
            beta_param[ctx, action] += (1 - normalized)
            ep_reward += reward
            state = next_state
        rewards.append(ep_reward)
        outcomes.append(outcome)
    
    resolved = sum(1 for o in outcomes[-200:] if o == 'resolved') / 200
    avg_r = np.mean(rewards[-200:])
    print(f'{dataset:15s} | avg_reward={avg_r:6.2f} | resolution={resolved:.1%}')

print('\nAll datasets working correctly!')
