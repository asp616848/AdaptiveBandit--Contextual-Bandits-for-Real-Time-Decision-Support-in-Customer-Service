import numpy as np
from collections import Counter
from stable_baselines3 import PPO
from Simulation_4.env import SupportEnv

model = PPO.load('Simulation_4/artifacts/phase10_v2/models/best_model')
env = SupportEnv(artifacts_root='Simulation_4/artifacts', nlg_enabled=False)

terminal_types = Counter()
turn_counts_by_terminal = {}
action_counts = Counter()
close_progress_values = []
close_turn_values = []

for ep in range(1000):
    obs, _ = env.reset(seed=70000 + ep)
    done = False
    ep_turns = 0

    while not done:
        pre_progress = float(env.state.get('progress', 0.0))
        pre_turn = int(env.state.get('turn_count', 0))

        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        action_counts[a] += 1

        if a == 4:  # Close
            close_progress_values.append(pre_progress)
            close_turn_values.append(pre_turn)

        obs, reward, done, truncated, info = env.step(a)
        ep_turns += 1
        if truncated:
            done = True

    # Get exact terminal type from info
    outcome = info.get('last_transition_outcome', {})
    terminal_type = outcome.get('terminal_type', 'unknown')
    terminal_types[terminal_type] += 1

    if terminal_type not in turn_counts_by_terminal:
        turn_counts_by_terminal[terminal_type] = []
    turn_counts_by_terminal[terminal_type].append(ep_turns)

print('=== TERMINAL TYPE BREAKDOWN (1000 episodes) ===')
for t, count in sorted(terminal_types.items(), key=lambda x: -x[1]):
    turns = turn_counts_by_terminal.get(t, [])
    mean_t = np.mean(turns) if turns else 0
    print(f'{t:<25}: {count:>4} episodes ({count/10:.1f}%)  mean_turns={mean_t:.1f}')

print(f'\nTotal episodes: {sum(terminal_types.values())}')
print(f'All accounted for: {sum(terminal_types.values()) == 1000}')

print('\n=== ACTION DISTRIBUTION ===')
total_actions = sum(action_counts.values())
names = {0: 'AskInfo', 1: 'ProvideSolution', 2: 'AffectiveRepair', 3: 'Escalate', 4: 'Close'}
for a in range(5):
    print(f'{names[a]:<20}: {action_counts[a]/total_actions:.1%}')

print('\n=== CLOSE ACTION DETAILS ===')
if close_progress_values:
    print(f'Total Close actions: {len(close_progress_values)}')
    print(f'Progress at Close - mean: {np.mean(close_progress_values):.3f}, '
          f'min: {np.min(close_progress_values):.3f}, '
          f'max: {np.max(close_progress_values):.3f}')
    print(f'Turn at Close - mean: {np.mean(close_turn_values):.1f}, '
          f'min: {np.min(close_turn_values)}, '
          f'max: {np.max(close_turn_values)}')
    print(f'Close at progress < 0.20: {sum(1 for p in close_progress_values if p < 0.20)}')
    print(f'Close at progress 0.20-0.50: {sum(1 for p in close_progress_values if 0.20 <= p < 0.50)}')
    print(f'Close at progress 0.50-0.85: {sum(1 for p in close_progress_values if 0.50 <= p < 0.85)}')
    print(f'Close at progress >= 0.85: {sum(1 for p in close_progress_values if p >= 0.85)}')
    print(f'Close at turn < 5: {sum(1 for t in close_turn_values if t < 5)}')
    print(f'Close at turn 5-10: {sum(1 for t in close_turn_values if 5 <= t < 10)}')
    print(f'Close at turn 10-15: {sum(1 for t in close_turn_values if 10 <= t < 15)}')
    print(f'Close at turn >= 15: {sum(1 for t in close_turn_values if t >= 15)}')
else:
    print('No Close actions taken')
