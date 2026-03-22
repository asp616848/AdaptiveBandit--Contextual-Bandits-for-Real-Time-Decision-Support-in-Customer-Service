"""
Training & Evaluation Utilities — Phase II RL Prompt Selector
=============================================================
Provides:

  train_dqn()   — training loop for DQN
  train_a2c()   — training loop for A2C
  train_ppo()   — training loop for PPO
  evaluate()    — shared evaluation loop (all agents)
  run_demo_conversation() — interactive demo showing the full pipeline

All training functions return a TrainingResult object with:
  - episode_rewards  (list)
  - success_rates    (list, smoothed)
  - strategy_usage   (dict of lists)
  - losses           (list)
  - evaluation snapshots at regular intervals

This module is imported by the individual notebooks so logic isn't duplicated.
"""

from __future__ import annotations
import numpy as np
import time
import sys, os
from dataclasses import dataclass, field
from collections import deque

# ── local imports  (adjust path if running from a notebook) ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from environment import MDPCustomerServiceEnv
from reward     import RewardShaper, EpisodeStats
from mock_llm   import MockLLM
from strategy_prompts import STRATEGY_NAMES, STRATEGY_LABELS

# ── Training configuration ────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    n_episodes:          int   = 3000
    eval_every:          int   = 100    # run evaluation snapshot every N episodes
    eval_episodes:       int   = 50     # episodes per evaluation snapshot
    smooth_window:       int   = 50     # rolling window for progress printing
    print_every:         int   = 200    # print to console every N episodes
    save_path:           str   = ''     # if non-empty, save best model here
    seed:                int   = 42


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    agent_name:          str
    dataset:             str
    episode_rewards:     list = field(default_factory=list)
    episode_lengths:     list = field(default_factory=list)
    losses:              list = field(default_factory=list)
    strategy_usage_ep:   list = field(default_factory=list)   # per episode
    eval_snapshots:      list = field(default_factory=list)   # list of EvalResult
    train_time_s:        float = 0.0

    def smooth_rewards(self, window: int = 50) -> list:
        r = np.array(self.episode_rewards)
        if len(r) < window:
            return r.tolist()
        kernel = np.ones(window) / window
        return np.convolve(r, kernel, mode='valid').tolist()

    def final_success_rate(self) -> float:
        if self.eval_snapshots:
            return self.eval_snapshots[-1].success_rate
        return 0.0

    def aggregate_strategy_usage(self) -> dict:
        """Sum strategy counts across all episodes."""
        agg = {name: 0 for name in STRATEGY_NAMES}
        for usage in self.strategy_usage_ep:
            for name, count in usage.items():
                agg[name] += count
        total = sum(agg.values()) or 1
        return {k: v / total for k, v in agg.items()}


@dataclass
class EvalResult:
    episode:       int
    success_rate:  float
    avg_reward:    float
    avg_turns:     float
    escalation_rate: float
    abandon_rate:  float
    strategy_dist: dict


# ── Shared evaluation function ────────────────────────────────────────────────

def evaluate(agent, env: MDPCustomerServiceEnv, n_episodes: int = 50,
             shaper: RewardShaper | None = None,
             episode_num: int = 0) -> EvalResult:
    """
    Run the agent greedily for n_episodes and collect statistics.
    No exploration, no learning.
    """
    shaper = shaper or RewardShaper()
    llm    = MockLLM(dataset=env.dataset)

    outcomes    = []
    total_r     = []
    total_turns = []
    act_counts  = np.zeros(env.NUM_ACTIONS)

    for _ in range(n_episodes):
        state  = env.reset()
        ep_r   = 0.0
        prev_s = state.copy()

        while not env.done:
            # greedy action
            if hasattr(agent, 'act_greedy'):
                action = agent.act_greedy(state)
            else:
                action = int(np.argmax(agent.get_q_values(state)))

            next_state, raw_r, done, info = env.step(action)

            shaped_r, _ = shaper.shape(
                raw_r, prev_s, next_state, action, done,
                info.get('outcome'), env.turn, env.consecutive_count)

            ep_r   += shaped_r
            prev_s  = next_state.copy()
            state   = next_state
            act_counts[action] += 1

        outcomes.append(env.outcome)
        total_r.append(ep_r)
        total_turns.append(env.turn)

    success_rate   = outcomes.count('resolved')  / n_episodes
    escalation_rate= outcomes.count('escalated') / n_episodes
    abandon_rate   = outcomes.count('abandoned') / n_episodes

    act_freq = act_counts / act_counts.sum() if act_counts.sum() > 0 else act_counts
    strategy_dist = {STRATEGY_NAMES[i]: float(act_freq[i])
                     for i in range(env.NUM_ACTIONS)}

    return EvalResult(
        episode        = episode_num,
        success_rate   = success_rate,
        avg_reward     = float(np.mean(total_r)),
        avg_turns      = float(np.mean(total_turns)),
        escalation_rate= escalation_rate,
        abandon_rate   = abandon_rate,
        strategy_dist  = strategy_dist,
    )


# ── DQN training loop ─────────────────────────────────────────────────────────

def train_dqn(agent, dataset: str = 'twitter',
              cfg: TrainConfig | None = None) -> TrainingResult:
    cfg    = cfg or TrainConfig()
    env    = MDPCustomerServiceEnv(dataset)
    shaper = RewardShaper()
    llm    = MockLLM(dataset=dataset)
    result = TrainingResult(agent_name="DQN", dataset=dataset)

    np.random.seed(cfg.seed)
    best_success = 0.0
    recent_r     = deque(maxlen=cfg.smooth_window)
    t0           = time.time()

    print(f"Training DQN on {env.dataset_name}  ({cfg.n_episodes} episodes)...")
    print(f"{'Episode':>8}  {'Avg Reward':>11}  {'ε':>6}  {'Success%':>9}  {'Loss':>8}")
    print("─" * 52)

    for ep in range(1, cfg.n_episodes + 1):
        state  = env.reset()
        ep_r   = 0.0
        losses = []
        stats  = EpisodeStats()
        prev_s = state.copy()

        while not env.done:
            action    = agent.act(state, training=True)
            next_state, raw_r, done, info = env.step(action)

            # call mock LLM (records response in env transcript)
            response = llm.generate(action, info.get('context_for_llm'))
            env.add_agent_response_to_transcript(response)

            shaped_r, bd = shaper.shape(
                raw_r, prev_s, next_state, action, done,
                info.get('outcome'), env.turn, env.consecutive_count)

            agent.remember(prev_s, action, shaped_r, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            stats.record(shaped_r, bd, action)
            ep_r   += shaped_r
            prev_s  = next_state.copy()
            state   = next_state

        stats.outcome = env.outcome
        result.episode_rewards.append(ep_r)
        result.episode_lengths.append(env.turn)
        result.losses.append(float(np.mean(losses)) if losses else 0.0)
        result.strategy_usage_ep.append(stats.strategy_distribution())
        recent_r.append(ep_r)

        # evaluation snapshot
        if ep % cfg.eval_every == 0:
            ev = evaluate(agent, MDPCustomerServiceEnv(dataset),
                          cfg.eval_episodes, shaper, ep)
            result.eval_snapshots.append(ev)
            if ev.success_rate > best_success and cfg.save_path:
                best_success = ev.success_rate
                agent.save(cfg.save_path + f'/dqn_{dataset}')

        # progress print
        if ep % cfg.print_every == 0:
            avg_r  = np.mean(recent_r)
            sr     = result.eval_snapshots[-1].success_rate * 100 if result.eval_snapshots else 0.0
            loss_v = result.losses[-1]
            print(f"{ep:>8}  {avg_r:>+11.3f}  {agent.eps:>6.3f}  {sr:>8.1f}%  {loss_v:>8.4f}")

    result.train_time_s = time.time() - t0
    print(f"\nDone in {result.train_time_s:.1f}s  |  "
          f"Final success rate: {result.final_success_rate()*100:.1f}%")
    return result


# ── A2C training loop ─────────────────────────────────────────────────────────

def train_a2c(agent, dataset: str = 'twitter',
              cfg: TrainConfig | None = None) -> TrainingResult:
    cfg    = cfg or TrainConfig()
    env    = MDPCustomerServiceEnv(dataset)
    shaper = RewardShaper()
    llm    = MockLLM(dataset=dataset)
    result = TrainingResult(agent_name="A2C", dataset=dataset)

    np.random.seed(cfg.seed)
    best_success = 0.0
    recent_r     = deque(maxlen=cfg.smooth_window)
    t0           = time.time()

    print(f"Training A2C on {env.dataset_name}  ({cfg.n_episodes} episodes)...")
    print(f"{'Episode':>8}  {'Avg Reward':>11}  {'Success%':>9}  {'Loss':>8}  {'Entropy':>8}")
    print("─" * 56)

    for ep in range(1, cfg.n_episodes + 1):
        state  = env.reset()
        ep_r   = 0.0
        stats  = EpisodeStats()
        prev_s = state.copy()

        while not env.done:
            action, _lp = agent.act(state, training=True)
            next_state, raw_r, done, info = env.step(action)

            response = llm.generate(action, info.get('context_for_llm'))
            env.add_agent_response_to_transcript(response)

            shaped_r, bd = shaper.shape(
                raw_r, prev_s, next_state, action, done,
                info.get('outcome'), env.turn, env.consecutive_count)

            agent.store(prev_s, action, shaped_r, done)
            stats.record(shaped_r, bd, action)
            ep_r   += shaped_r
            prev_s  = next_state.copy()
            state   = next_state

        loss = agent.learn(state)

        stats.outcome = env.outcome
        result.episode_rewards.append(ep_r)
        result.episode_lengths.append(env.turn)
        result.losses.append(float(loss) if loss is not None else 0.0)
        result.strategy_usage_ep.append(stats.strategy_distribution())
        recent_r.append(ep_r)

        if ep % cfg.eval_every == 0:
            ev = evaluate(agent, MDPCustomerServiceEnv(dataset),
                          cfg.eval_episodes, shaper, ep)
            result.eval_snapshots.append(ev)
            if ev.success_rate > best_success and cfg.save_path:
                best_success = ev.success_rate
                agent.save(cfg.save_path + f'/a2c_{dataset}')

        if ep % cfg.print_every == 0:
            avg_r = np.mean(recent_r)
            sr    = result.eval_snapshots[-1].success_rate * 100 if result.eval_snapshots else 0.0
            loss_v= result.losses[-1]
            print(f"{ep:>8}  {avg_r:>+11.3f}  {sr:>8.1f}%  {loss_v:>8.4f}")

    result.train_time_s = time.time() - t0
    print(f"\nDone in {result.train_time_s:.1f}s  |  "
          f"Final success rate: {result.final_success_rate()*100:.1f}%")
    return result


# ── PPO training loop ─────────────────────────────────────────────────────────

def train_ppo(agent, dataset: str = 'twitter',
              cfg: TrainConfig | None = None) -> TrainingResult:
    cfg    = cfg or TrainConfig()
    env    = MDPCustomerServiceEnv(dataset)
    shaper = RewardShaper()
    llm    = MockLLM(dataset=dataset)
    result = TrainingResult(agent_name="PPO", dataset=dataset)

    np.random.seed(cfg.seed)
    best_success = 0.0
    recent_r     = deque(maxlen=cfg.smooth_window)
    t0           = time.time()

    print(f"Training PPO on {env.dataset_name}  ({cfg.n_episodes} episodes)...")
    print(f"{'Episode':>8}  {'Avg Reward':>11}  {'Success%':>9}  {'Clip Frac':>10}")
    print("─" * 48)

    state  = env.reset()
    ep_r   = 0.0
    ep_idx = 0
    stats  = EpisodeStats()
    prev_s = state.copy()

    while ep_idx < cfg.n_episodes:
        action, lp, val = agent.act(state, training=True)
        next_state, raw_r, done, info = env.step(action)

        response = llm.generate(action, info.get('context_for_llm'))
        env.add_agent_response_to_transcript(response)

        shaped_r, bd = shaper.shape(
            raw_r, prev_s, next_state, action, done,
            info.get('outcome'), env.turn, env.consecutive_count)

        agent.store(prev_s, action, shaped_r, done, val, lp)
        stats.record(shaped_r, bd, action)
        ep_r   += shaped_r
        prev_s  = next_state.copy()
        state   = next_state

        # learn when buffer full or episode ends
        if agent.buffer_full() or done:
            loss = agent.learn(state)

        if done:
            ep_idx += 1
            stats.outcome = env.outcome
            result.episode_rewards.append(ep_r)
            result.episode_lengths.append(env.turn)
            result.losses.append(float(loss) if 'loss' in dir() else 0.0)
            result.strategy_usage_ep.append(stats.strategy_distribution())
            recent_r.append(ep_r)

            if ep_idx % cfg.eval_every == 0:
                ev = evaluate(agent, MDPCustomerServiceEnv(dataset),
                              cfg.eval_episodes, shaper, ep_idx)
                result.eval_snapshots.append(ev)
                if ev.success_rate > best_success and cfg.save_path:
                    best_success = ev.success_rate
                    agent.save(cfg.save_path + f'/ppo_{dataset}')

            if ep_idx % cfg.print_every == 0:
                avg_r = np.mean(recent_r)
                sr    = result.eval_snapshots[-1].success_rate * 100 if result.eval_snapshots else 0.0
                print(f"{ep_idx:>8}  {avg_r:>+11.3f}  {sr:>8.1f}%")

            # reset for next episode
            state   = env.reset()
            ep_r    = 0.0
            stats   = EpisodeStats()
            prev_s  = state.copy()

    result.train_time_s = time.time() - t0
    print(f"\nDone in {result.train_time_s:.1f}s  |  "
          f"Final success rate: {result.final_success_rate()*100:.1f}%")
    return result


# ── Demo conversation (for notebook sections) ─────────────────────────────────

def run_demo_conversation(agent, dataset: str = 'twitter',
                          verbose: bool = True) -> dict:
    """
    Run ONE episode with the trained agent, printing the full LLM pipeline
    at each turn so the prompt-selection mechanism is visible.

    Returns a summary dict for display in notebooks.
    """
    env    = MDPCustomerServiceEnv(dataset)
    llm    = MockLLM(dataset=dataset)
    shaper = RewardShaper()
    state  = env.reset()
    prev_s = state.copy()

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  DEMO CONVERSATION  |  {env.dataset_name}")
        print(f"  Trained agent: {agent.name}")
        print(f"{'═'*65}")
        print(f"\n  Customer opens with:\n    \"{env.transcript[0]['text']}\"\n")

    total_reward = 0.0
    turn_details = []

    while not env.done:
        if hasattr(agent, 'act_greedy'):
            action = agent.act_greedy(state)
        else:
            action = agent.act(state)[0] if isinstance(agent.act(state), tuple) else agent.act(state)

        strategy_name = env.ACTIONS[action]

        # build prompt and generate response via mock LLM
        context = env.get_llm_context()
        prompt, response = llm.generate_with_prompt(action, context)

        if verbose:
            print(f"  ── Turn {env.turn + 1} ──")
            print(f"  State: sentiment={state[0]:.2f}  "
                  f"frustration={state[1]:.2f}  info={state[2]:.2f}")
            print(f"  Agent selects strategy: [{action}] {strategy_name.upper()}")
            print(f"  Prompt (excerpt):\n    {prompt[50:200]}...")
            print(f"  Response:\n    {response}\n")

        next_state, raw_r, done, info = env.step(action)
        env.add_agent_response_to_transcript(response)

        shaped_r, _ = shaper.shape(
            raw_r, prev_s, next_state, action, done,
            info.get('outcome'), env.turn, env.consecutive_count)

        total_reward += shaped_r

        if verbose and env.transcript:
            last_cust = [t for t in env.transcript if t['role'] == 'customer'][-1]["text"]
            print(f"  Customer responds:\n    \"{last_cust}\"\n")

        turn_details.append({
            "turn":     env.turn,
            "strategy": strategy_name,
            "reward":   shaped_r,
            "response": response,
        })
        prev_s = next_state.copy()
        state  = next_state

    if verbose:
        print(f"{'─'*65}")
        print(f"  OUTCOME  : {env.outcome.upper()}")
        print(f"  Total reward  : {total_reward:+.2f}")
        print(f"  Turns taken   : {env.turn}")
        print(f"{'═'*65}\n")

    return {
        "outcome":      env.outcome,
        "total_reward": total_reward,
        "turns":        env.turn,
        "turn_details": turn_details,
    }
