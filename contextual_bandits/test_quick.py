"""Quick test to verify notebooks will run correctly"""
import numpy as np
import sys
sys.path.insert(0, '.')
from environment import CustomerServiceEnv

# test all three datasets
for dataset in ['twitter', 'reddit', 'openassistant']:
    env = CustomerServiceEnv(dataset=dataset)
    
    # fixed thompson sampling with 3D discretization
    n_bins = 4
    n_contexts = n_bins * n_bins * n_bins  # sentiment x frustration x info_gathered
    alpha = np.ones((n_contexts, 5))
    beta_param = np.ones((n_contexts, 5))
    
    rewards = []
    outcomes = []
    
    for ep in range(500):
        state = env.reset()
        ep_reward = 0
        while not env.done:
            # 3D context: sentiment, frustration, info_gathered
            s_bin = min(int(state[0]*n_bins), n_bins-1)
            f_bin = min(int(state[1]*n_bins), n_bins-1)
            i_bin = min(int(state[2]*n_bins), n_bins-1)
            ctx = (s_bin * n_bins + f_bin) * n_bins + i_bin
            samples = np.random.beta(alpha[ctx], beta_param[ctx])
            action = np.argmax(samples)
            next_state, reward, done, outcome = env.step(action)
            # threshold-based update instead of lossy normalization
            if reward > 0:
                alpha[ctx, action] += 1
            else:
                beta_param[ctx, action] += 1
            ep_reward += reward
            state = next_state
        rewards.append(ep_reward)
        outcomes.append(outcome)
    
    resolved = sum(1 for o in outcomes[-200:] if o == 'resolved') / 200
    avg_r = np.mean(rewards[-200:])
    print(f'{dataset:15s} | avg_reward={avg_r:6.2f} | resolution={resolved:.1%}')

print('\nAll datasets working correctly!')
