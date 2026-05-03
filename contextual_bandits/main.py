#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Simulation_4.env.support_env import SupportEnv
from contextual_bandits.env_factory_FINAL import build_contextual_bandit_env_FINAL
from contextual_bandits.bandit_metrics import compute_bandit_metrics, summarize_episode_outcomes
from contextual_bandits.epsilon_greedy import LinearEpsilonGreedy
from contextual_bandits.linucb import LinUCB
from contextual_bandits.strategy_policies import STRATEGY_NAMES, build_strategies, extract_strategy_context
from contextual_bandits.thompson_sampling import LinearThompsonSampling
from contextual_bandits.truncated_policy_hybrid import TruncatedHybridPolicy


OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output" / "Contextual_bandit"


@dataclass
class RunConfig:
    artifacts_root: str
    collect_episodes: int
    eval_episodes: int
    horizons: list[int]
    seeds: list[int]
    algorithms: list[str]
    hybrid_k: list[int]
    output_root: str


@dataclass
class RunResult:
    algorithm: str
    mode: str
    horizon: int
    seed: int
    avg_per_turn_reward: float
    early_turn_reward: float
    sentiment_improvement: float
    immediate_regret: float
    resolution_rate: float
    mean_episode_reward: float
    success_count: int
    escalation_count: int
    dropout_count: int
    timeout_count: int


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _build_policy(name: str, n_actions: int, d: int):
    n = name.lower()
    if n == "linucb":
        return LinUCB(n_actions=n_actions, d=d, alpha=1.0)
    if n in {"thompson", "thompson_sampling", "lints"}:
        return LinearThompsonSampling(n_actions=n_actions, d=d, v=0.5)
    if n in {"epsilon", "epsilon_greedy", "egreedy"}:
        return LinearEpsilonGreedy(n_actions=n_actions, d=d, epsilon=0.2)
    raise ValueError(f"Unsupported algorithm: {name}")


def _turn_record(info: dict[str, Any], reward: float, action: int, turn: int) -> dict[str, Any]:
    return {
        "turn": int(turn),
        "action": int(action),
        "reward": float(reward),
        "delta_sentiment": float(info.get("delta_sentiment", 0.0)),
        "delta_frustration": float(info.get("delta_frustration", 0.0)),
        "delta_progress": float(info.get("delta_progress", 0.0)),
        "delta_info": float(info.get("delta_info", 0.0)),
        "reward_per_turn": float(info.get("reward_per_turn", 0.0)),
    }


def run_action_bandit(
    algorithm: str,
    artifacts_root: str,
    collect_episodes: int,
    eval_episodes: int,
    horizon: int,
    seed: int,
) -> tuple[RunResult, list[dict[str, Any]], Any]:
    set_seed(seed)
    env = build_contextual_bandit_env_FINAL(
        artifacts_root=artifacts_root,
        use_nlp=False,
        use_nlg=False,
        use_masking=True,
        horizon=horizon,
    )

    policy = _build_policy(algorithm, n_actions=5, d=9)

    for ep in range(collect_episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        done = False
        turn = 0
        while not done and turn < horizon:
            mask = env.action_masks()
            action = int(policy.select_action(obs, mask))
            next_obs, reward, done, truncated, _ = env.step(action)
            policy.update(obs, action, float(reward))
            obs = next_obs
            turn += 1
            if truncated:
                break

    eval_logs: list[dict[str, Any]] = []
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed * 200_000 + ep)
        done = False
        turn = 0
        episode_reward = 0.0
        turns = []
        last_info: dict[str, Any] = {}

        while not done and turn < horizon:
            mask = env.action_masks()
            action = int(policy.predict(obs, mask))
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += float(reward)
            turns.append(_turn_record(info, reward, action, turn))
            last_info = info
            turn += 1
            if truncated:
                break

        terminal_type = str((last_info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout"))
        eval_logs.append(
            {
                "episode": ep,
                "terminal_type": terminal_type,
                "episode_reward": episode_reward,
                "turn_count": turn,
                "turns": turns,
            }
        )

    metrics = compute_bandit_metrics(eval_logs, early_k=min(3, max(horizon, 1)))
    counts = summarize_episode_outcomes(eval_logs)
    result = RunResult(
        algorithm=algorithm,
        mode="action_cb",
        horizon=horizon,
        seed=seed,
        avg_per_turn_reward=metrics["avg_per_turn_reward"],
        early_turn_reward=metrics["early_turn_reward"],
        sentiment_improvement=metrics["sentiment_improvement"],
        immediate_regret=metrics["immediate_regret"],
        resolution_rate=metrics["resolution_rate"],
        mean_episode_reward=metrics["mean_episode_reward"],
        success_count=counts["success"],
        escalation_count=counts["escalation"],
        dropout_count=counts["dropout"],
        timeout_count=counts["timeout"],
    )
    return result, eval_logs, policy


def run_strategy_bandit(
    algorithm: str,
    artifacts_root: str,
    collect_episodes: int,
    eval_episodes: int,
    horizon: int,
    seed: int,
) -> tuple[RunResult, list[dict[str, Any]]]:
    set_seed(seed)
    env = build_contextual_bandit_env_FINAL(
        artifacts_root=artifacts_root,
        use_nlp=False,
        use_nlg=False,
        use_masking=True,
        horizon=horizon,
    )

    strategies = build_strategies()
    policy = _build_policy(algorithm, n_actions=len(strategies), d=3)

    for ep in range(collect_episodes):
        obs, _ = env.reset(seed=seed * 300_000 + ep)
        context = extract_strategy_context(env.state)
        strategy_id = int(policy.select_action(context))
        strategy = strategies[strategy_id]

        done = False
        turn = 0
        episode_reward = 0.0
        while not done and turn < horizon:
            mask = env.action_masks()
            action = strategy.action(obs, turn, mask)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)
            turn += 1
            if truncated:
                break
        policy.update(context, strategy_id, episode_reward)

    eval_logs: list[dict[str, Any]] = []
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed * 400_000 + ep)
        context = extract_strategy_context(env.state)
        strategy_id = int(policy.predict(context))
        strategy = strategies[strategy_id]

        done = False
        turn = 0
        episode_reward = 0.0
        turns = []
        last_info: dict[str, Any] = {}

        while not done and turn < horizon:
            mask = env.action_masks()
            action = strategy.action(obs, turn, mask)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += float(reward)
            turns.append(_turn_record(info, reward, action, turn))
            last_info = info
            turn += 1
            if truncated:
                break

        terminal_type = str((last_info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout"))
        eval_logs.append(
            {
                "episode": ep,
                "terminal_type": terminal_type,
                "episode_reward": episode_reward,
                "turn_count": turn,
                "turns": turns,
                "strategy_id": strategy_id,
                "strategy_name": STRATEGY_NAMES.get(strategy_id, str(strategy_id)),
            }
        )

    metrics = compute_bandit_metrics(eval_logs, early_k=min(3, max(horizon, 1)))
    counts = summarize_episode_outcomes(eval_logs)
    result = RunResult(
        algorithm=algorithm,
        mode="strategy_cb",
        horizon=horizon,
        seed=seed,
        avg_per_turn_reward=metrics["avg_per_turn_reward"],
        early_turn_reward=metrics["early_turn_reward"],
        sentiment_improvement=metrics["sentiment_improvement"],
        immediate_regret=metrics["immediate_regret"],
        resolution_rate=metrics["resolution_rate"],
        mean_episode_reward=metrics["mean_episode_reward"],
        success_count=counts["success"],
        escalation_count=counts["escalation"],
        dropout_count=counts["dropout"],
        timeout_count=counts["timeout"],
    )
    return result, eval_logs


def run_truncated_hybrid(
    trained_policy: Any,
    artifacts_root: str,
    eval_episodes: int,
    horizon: int,
    k_turns: int,
    seed: int,
    algorithm_name: str,
) -> tuple[RunResult, list[dict[str, Any]]]:
    set_seed(seed)
    env = build_contextual_bandit_env_FINAL(
        artifacts_root=artifacts_root,
        use_nlp=False,
        use_nlg=False,
        use_masking=True,
        horizon=horizon,
    )
    hybrid = TruncatedHybridPolicy(action_policy=trained_policy, k_turns=int(k_turns))

    eval_logs: list[dict[str, Any]] = []
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed * 500_000 + ep)
        done = False
        turn = 0
        episode_reward = 0.0
        turns = []
        last_info: dict[str, Any] = {}

        while not done and turn < horizon:
            mask = env.action_masks()
            action = hybrid.action(obs, turn, mask)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += float(reward)
            turns.append(_turn_record(info, reward, action, turn))
            last_info = info
            turn += 1
            if truncated:
                break

        terminal_type = str((last_info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout"))
        eval_logs.append(
            {
                "episode": ep,
                "terminal_type": terminal_type,
                "episode_reward": episode_reward,
                "turn_count": turn,
                "turns": turns,
            }
        )

    metrics = compute_bandit_metrics(eval_logs, early_k=min(3, max(horizon, 1)))
    counts = summarize_episode_outcomes(eval_logs)
    result = RunResult(
        algorithm=f"{algorithm_name}_hybrid_k{k_turns}",
        mode="truncated_hybrid",
        horizon=horizon,
        seed=seed,
        avg_per_turn_reward=metrics["avg_per_turn_reward"],
        early_turn_reward=metrics["early_turn_reward"],
        sentiment_improvement=metrics["sentiment_improvement"],
        immediate_regret=metrics["immediate_regret"],
        resolution_rate=metrics["resolution_rate"],
        mean_episode_reward=metrics["mean_episode_reward"],
        success_count=counts["success"],
        escalation_count=counts["escalation"],
        dropout_count=counts["dropout"],
        timeout_count=counts["timeout"],
    )
    return result, eval_logs


def _save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def print_results_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Print comprehensive results summary to console."""
    print("\n" + "█" * 80)
    print("  CONTEXTUAL BANDIT EXPERIMENT RESULTS")
    print("█" * 80 + "\n")
    
    print(f"📊 Output Directory: {out_dir}\n")
    print(f"Total Runs: {len(df)}\n")
    
    # ========================================================================
    # By Mode (ActionCB, StrategyCB, Hybrid)
    # ========================================================================
    print("=" * 80)
    print("RESULTS BY MODE")
    print("=" * 80)
    for mode in df["mode"].unique():
        mode_df = df[df["mode"] == mode]
        print(f"\n{mode.upper()}")
        print("-" * 80)
        print(f"  Runs: {len(mode_df)}")
        print(f"  Avg Resolution Rate: {mode_df['resolution_rate'].mean()*100:.2f}% ± {mode_df['resolution_rate'].std()*100:.2f}%")
        print(f"  Avg Per-Turn Reward: {mode_df['avg_per_turn_reward'].mean():.4f} ± {mode_df['avg_per_turn_reward'].std():.4f}")
        print(f"  Avg Episode Reward: {mode_df['mean_episode_reward'].mean():.4f} ± {mode_df['mean_episode_reward'].std():.4f}")
        print(f"  Sentiment Improvement: {mode_df['sentiment_improvement'].mean():.4f} ± {mode_df['sentiment_improvement'].std():.4f}")
    
    # ========================================================================
    # By Algorithm
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS BY ALGORITHM")
    print("=" * 80)
    for algo in sorted(df["algorithm"].unique()):
        algo_df = df[df["algorithm"] == algo]
        print(f"\n{algo.upper()}")
        print("-" * 80)
        print(f"  Runs: {len(algo_df)}")
        print(f"  Avg Resolution Rate: {algo_df['resolution_rate'].mean()*100:.2f}% ± {algo_df['resolution_rate'].std()*100:.2f}%")
        print(f"  Avg Per-Turn Reward: {algo_df['avg_per_turn_reward'].mean():.4f} ± {algo_df['avg_per_turn_reward'].std():.4f}")
        print(f"  Avg Episode Reward: {algo_df['mean_episode_reward'].mean():.4f} ± {algo_df['mean_episode_reward'].std():.4f}")
        print(f"  Sentiment Improvement: {algo_df['sentiment_improvement'].mean():.4f} ± {algo_df['sentiment_improvement'].std():.4f}")
    
    # ========================================================================
    # By Horizon
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS BY HORIZON (Action-Level CB)")
    print("=" * 80)
    action_df = df[df["mode"] == "action_cb"]
    for horizon in sorted(action_df["horizon"].unique()):
        h_df = action_df[action_df["horizon"] == horizon]
        print(f"\nHorizon T={horizon}")
        print("-" * 80)
        print(f"  Runs: {len(h_df)}")
        print(f"  Avg Resolution Rate: {h_df['resolution_rate'].mean()*100:.2f}% ± {h_df['resolution_rate'].std()*100:.2f}%")
        print(f"  Avg Per-Turn Reward: {h_df['avg_per_turn_reward'].mean():.4f} ± {h_df['avg_per_turn_reward'].std():.4f}")
        print(f"  Avg Episode Reward: {h_df['mean_episode_reward'].mean():.4f} ± {h_df['mean_episode_reward'].std():.4f}")
        print(f"  Success: {h_df['success_count'].sum():3d} | Escalations: {h_df['escalation_count'].sum():3d} | "
              f"Dropouts: {h_df['dropout_count'].sum():3d} | Timeouts: {h_df['timeout_count'].sum():3d}")
    
    # ========================================================================
    # Top Performers (by resolution rate)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by resolution rate)")
    print("=" * 80)
    top_10 = df.nlargest(10, "resolution_rate")[
        ["algorithm", "mode", "horizon", "resolution_rate", "avg_per_turn_reward", "mean_episode_reward"]
    ]
    for idx, row in top_10.iterrows():
        print(f"\n  {row['algorithm']:15s} | {row['mode']:15s} | T={row['horizon']:2d} | "
              f"Res={row['resolution_rate']*100:5.1f}% | "
              f"Reward/Turn={row['avg_per_turn_reward']:+.4f} | "
              f"Episode={row['mean_episode_reward']:+.4f}")
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"\n  Total Configurations Evaluated: {len(df)}")
    print(f"  Average Resolution Rate: {df['resolution_rate'].mean()*100:.2f}%")
    print(f"  Best Resolution Rate: {df['resolution_rate'].max()*100:.2f}% ({df[df['resolution_rate']==df['resolution_rate'].max()]['algorithm'].iloc[0]})")
    print(f"  Worst Resolution Rate: {df['resolution_rate'].min()*100:.2f}%")
    print(f"  Avg Per-Turn Reward: {df['avg_per_turn_reward'].mean():.4f}")
    print(f"  Avg Sentiment Improvement: {df['sentiment_improvement'].mean():.4f}")
    print(f"  Total Successful Resolutions: {df['success_count'].sum():,}")
    print(f"  Total Escalations: {df['escalation_count'].sum():,}")
    print(f"  Total Dropouts: {df['dropout_count'].sum():,}")
    print(f"  Total Timeouts: {df['timeout_count'].sum():,}")
    
    print("\n" + "█" * 80)
    print(f"✓ Full results saved to: {out_dir}")
    print(f"  - summary.csv (per-run metrics)")
    print(f"  - aggregate.csv (aggregated by mode/algorithm/horizon)")
    print(f"  - results.json (configuration metadata)")
    print(f"  - logs/ (JSONL episode logs)")
    print(f"  - models/ (trained policy pickles)")
    print("█" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contextual bandit experiment suite")
    parser.add_argument("--artifacts-root", type=str, default="Simulation_4/artifacts")
    parser.add_argument("--collect-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--horizons", type=str, default="1,2,3,5,8,20")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--algorithms", type=str, default="linucb,thompson,epsilon")
    parser.add_argument("--hybrid-k", type=str, default="1,2,3,5")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    algorithms = [x.strip().lower() for x in args.algorithms.split(",") if x.strip()]
    hybrid_k = [int(x.strip()) for x in args.hybrid_k.split(",") if x.strip()]

    cfg = RunConfig(
        artifacts_root=args.artifacts_root,
        collect_episodes=int(args.collect_episodes),
        eval_episodes=int(args.eval_episodes),
        horizons=horizons,
        seeds=seeds,
        algorithms=algorithms,
        hybrid_k=hybrid_k,
        output_root=args.output_root,
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_root) / run_id
    logs_dir = out_dir / "logs"
    models_dir = out_dir / "models"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[RunResult] = []

    for algo in cfg.algorithms:
        for horizon in cfg.horizons:
            for seed in cfg.seeds:
                print(f"[ActionCB] algo={algo} horizon={horizon} seed={seed}")
                action_res, action_logs, trained_policy = run_action_bandit(
                    algorithm=algo,
                    artifacts_root=cfg.artifacts_root,
                    collect_episodes=cfg.collect_episodes,
                    eval_episodes=cfg.eval_episodes,
                    horizon=horizon,
                    seed=seed,
                )
                all_results.append(action_res)
                _save_jsonl(logs_dir / f"action_{algo}_h{horizon}_s{seed}.jsonl", action_logs)
                if hasattr(trained_policy, "save"):
                    trained_policy.save(models_dir / f"action_{algo}_h{horizon}_s{seed}.pkl")

                print(f"[StrategyCB] algo={algo} horizon={horizon} seed={seed}")
                strategy_res, strategy_logs = run_strategy_bandit(
                    algorithm=algo,
                    artifacts_root=cfg.artifacts_root,
                    collect_episodes=cfg.collect_episodes,
                    eval_episodes=cfg.eval_episodes,
                    horizon=horizon,
                    seed=seed,
                )
                all_results.append(strategy_res)
                _save_jsonl(logs_dir / f"strategy_{algo}_h{horizon}_s{seed}.jsonl", strategy_logs)

                for k in cfg.hybrid_k:
                    if k > horizon:
                        continue
                    print(f"[Hybrid] algo={algo} horizon={horizon} seed={seed} k={k}")
                    hybrid_res, hybrid_logs = run_truncated_hybrid(
                        trained_policy=trained_policy,
                        artifacts_root=cfg.artifacts_root,
                        eval_episodes=cfg.eval_episodes,
                        horizon=horizon,
                        k_turns=k,
                        seed=seed,
                        algorithm_name=algo,
                    )
                    all_results.append(hybrid_res)
                    _save_jsonl(logs_dir / f"hybrid_{algo}_h{horizon}_k{k}_s{seed}.jsonl", hybrid_logs)

    df = pd.DataFrame([asdict(r) for r in all_results])
    df.to_csv(out_dir / "summary.csv", index=False)

    aggregate = (
        df.groupby(["mode", "algorithm", "horizon"], as_index=False)
        .agg(
            resolution_rate_mean=("resolution_rate", "mean"),
            resolution_rate_std=("resolution_rate", "std"),
            avg_per_turn_reward_mean=("avg_per_turn_reward", "mean"),
            early_turn_reward_mean=("early_turn_reward", "mean"),
            mean_episode_reward_mean=("mean_episode_reward", "mean"),
        )
        .fillna(0.0)
    )
    aggregate.to_csv(out_dir / "aggregate.csv", index=False)

    results_payload = {
        "run_id": run_id,
        "config": asdict(cfg),
        "n_rows": int(len(df)),
        "output_dir": str(out_dir),
    }
    (out_dir / "results.json").write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    
    # Print comprehensive results summary
    print_results_summary(df, out_dir)


if __name__ == "__main__":
    main()
