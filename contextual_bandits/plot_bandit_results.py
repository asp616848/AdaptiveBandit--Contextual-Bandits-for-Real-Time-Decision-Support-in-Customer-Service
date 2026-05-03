#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextual_bandits.strategy_policies import STRATEGY_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot contextual bandit experiment results")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing summary.csv and aggregate.csv")
    return parser.parse_args()


def _load_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = run_dir / "summary.csv"
    aggregate_path = run_dir / "aggregate.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_path}")
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing aggregate.csv: {aggregate_path}")

    summary_df = pd.read_csv(summary_path)
    aggregate_df = pd.read_csv(aggregate_path)
    return summary_df, aggregate_df


def _ensure_plots_dir(run_dir: Path) -> Path:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_resolution_vs_horizon(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    found = False

    for mode, label in [("action_cb", "Action CB"), ("strategy_cb", "Strategy CB"), ("ppo", "PPO")]:
        subset = summary_df[summary_df["mode"] == mode]
        if subset.empty:
            continue

        grouped = subset.groupby("horizon", as_index=True)["resolution_rate"].mean().sort_index()
        if grouped.empty:
            continue

        ax.plot(grouped.index, grouped.values, marker="o", label=label)
        found = True

    horizons = sorted(summary_df["horizon"].dropna().unique().tolist())
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Resolution Rate")
    ax.set_title("Resolution vs Horizon")
    if horizons:
        ax.set_xticks(horizons)
    ax.set_ylim(bottom=0.0)
    if found:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "resolution_vs_horizon.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    order = ["action_cb", "strategy_cb", "truncated_hybrid"]
    labels = ["Action CB", "Strategy CB", "Hybrid"]
    values = []
    used_labels = []

    for mode, label in zip(order, labels):
        subset = summary_df[summary_df["mode"] == mode]
        if subset.empty:
            continue
        values.append(float(subset["resolution_rate"].mean()))
        used_labels.append(label)

    if not values:
        ax.set_title("Model Comparison")
        fig.tight_layout()
        fig.savefig(plots_dir / "model_comparison.png", dpi=150)
        plt.close(fig)
        return

    ax.bar(used_labels, values)
    ax.set_xlabel("Method")
    ax.set_ylabel("Resolution Rate")
    ax.set_title("Model Comparison")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(plots_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_reward_vs_horizon(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    found = False

    reward_column = "mean_episode_reward" if "mean_episode_reward" in summary_df.columns else "avg_per_turn_reward"

    for mode, label in [("action_cb", "Action CB"), ("strategy_cb", "Strategy CB"), ("truncated_hybrid", "Hybrid")]:
        subset = summary_df[summary_df["mode"] == mode]
        if subset.empty:
            continue

        grouped = subset.groupby("horizon", as_index=True)[reward_column].mean().sort_index()
        if grouped.empty:
            continue

        ax.plot(grouped.index, grouped.values, marker="o", label=label)
        found = True

    horizons = sorted(summary_df["horizon"].dropna().unique().tolist())
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Reward")
    ax.set_title(f"Reward vs Horizon ({reward_column})")
    if horizons:
        ax.set_xticks(horizons)
    if found:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "reward_vs_horizon.png", dpi=150)
    plt.close(fig)


def plot_algorithm_comparison(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    algorithms = [
        ("linucb", "LinUCB"),
        ("thompson", "Thompson Sampling"),
        ("epsilon", "Epsilon Greedy"),
    ]
    mode_panels = [
        ("action_cb", "Action CB"),
        ("strategy_cb", "Strategy CB"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    if len(mode_panels) == 1:
        axes = [axes]

    any_data = False
    for ax, (mode, mode_label) in zip(axes, mode_panels):
        subset = summary_df[summary_df["mode"] == mode]
        if subset.empty:
            ax.set_title(mode_label)
            ax.set_xlabel("Horizon")
            continue

        found = False
        for algorithm, label in algorithms:
            algo_subset = subset[subset["algorithm"] == algorithm]
            if algo_subset.empty:
                continue

            grouped = algo_subset.groupby("horizon", as_index=True)["resolution_rate"].mean().sort_index()
            if grouped.empty:
                continue

            ax.plot(grouped.index, grouped.values, marker="o", label=label)
            found = True
            any_data = True

        horizons = sorted(subset["horizon"].dropna().unique().tolist())
        ax.set_title(mode_label)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Resolution Rate")
        if horizons:
            ax.set_xticks(horizons)
        ax.set_ylim(bottom=0.0)
        if found:
            ax.legend()

    fig.suptitle("Algorithm Comparison: Resolution vs Horizon")
    fig.tight_layout()
    fig.savefig(plots_dir / "algorithm_comparison.png", dpi=150)
    plt.close(fig)

    if not any_data:
        return


def plot_strategy_usage(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    strategy_df = summary_df[summary_df["mode"] == "strategy_cb"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    if strategy_df.empty:
        ax.set_title("Strategy Usage Distribution")
        ax.set_xlabel("Usage (%)")
        ax.set_ylabel("Strategy")
        fig.tight_layout()
        fig.savefig(plots_dir / "strategy_usage_distribution.png", dpi=150)
        plt.close(fig)
        return

    strategy_label_column = None
    if "strategy_name" in strategy_df.columns:
        strategy_label_column = "strategy_name"
    elif "strategy_id" in strategy_df.columns:
        strategy_label_column = "strategy_id"

    if strategy_label_column is None:
        ax.set_title("Strategy Usage Distribution")
        ax.set_xlabel("Usage (%)")
        ax.set_ylabel("Strategy")
        fig.tight_layout()
        fig.savefig(plots_dir / "strategy_usage_distribution.png", dpi=150)
        plt.close(fig)
        return

    if strategy_label_column == "strategy_id":
        strategy_df["strategy_label"] = strategy_df["strategy_id"].map(lambda value: STRATEGY_NAMES.get(int(value), str(value)))
    else:
        strategy_df["strategy_label"] = strategy_df["strategy_name"].astype(str)

    counts = strategy_df["strategy_label"].value_counts().sort_values(ascending=True)
    percentages = 100.0 * counts / counts.sum()

    ax.barh(percentages.index.tolist(), percentages.values.tolist())
    ax.set_xlabel("Usage (%)")
    ax.set_ylabel("Strategy")
    ax.set_title("Strategy Usage Distribution")
    ax.set_xlim(left=0.0)
    fig.tight_layout()
    fig.savefig(plots_dir / "strategy_usage_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    plots_dir = _ensure_plots_dir(run_dir)
    summary_df, aggregate_df = _load_tables(run_dir)

    if summary_df.empty:
        raise ValueError(f"summary.csv is empty: {run_dir / 'summary.csv'}")

    plot_resolution_vs_horizon(summary_df, plots_dir)
    plot_model_comparison(summary_df, plots_dir)
    plot_algorithm_comparison(summary_df, plots_dir)
    plot_reward_vs_horizon(summary_df, plots_dir)
    plot_strategy_usage(summary_df, plots_dir)

    (plots_dir / "plot_inputs.txt").write_text(
        f"summary_rows={len(summary_df)}\naggregate_rows={len(aggregate_df)}\n",
        encoding="utf-8",
    )

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()