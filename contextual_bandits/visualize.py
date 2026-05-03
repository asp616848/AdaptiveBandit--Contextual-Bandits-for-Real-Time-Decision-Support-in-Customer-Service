#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize contextual bandit experiment results")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        print("summary.csv is empty; skipping plots")
        return

    # 1. Resolution rate by horizon
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="horizon", y="resolution_rate", hue="algorithm", style="mode", marker="o", ax=ax)
    ax.set_title("Resolution Rate by Horizon")
    ax.set_ylabel("Resolution Rate")
    fig.tight_layout()
    fig.savefig(plots_dir / "resolution_by_horizon.png", dpi=140)
    plt.close(fig)

    # 2. Avg per-turn reward by algorithm/mode
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="algorithm", y="avg_per_turn_reward", hue="mode", ax=ax)
    ax.set_title("Average Per-Turn Reward")
    fig.tight_layout()
    fig.savefig(plots_dir / "avg_per_turn_reward.png", dpi=140)
    plt.close(fig)

    # 3. Early turn reward heatmap
    pivot = (
        df.groupby(["algorithm", "horizon"], as_index=False)["early_turn_reward"]
        .mean()
        .pivot(index="algorithm", columns="horizon", values="early_turn_reward")
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Early-Turn Reward Heatmap")
    fig.tight_layout()
    fig.savefig(plots_dir / "early_turn_reward_heatmap.png", dpi=140)
    plt.close(fig)

    # 4. Sentiment improvement distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="mode", y="sentiment_improvement", ax=ax)
    ax.set_title("Sentiment Improvement by Mode")
    fig.tight_layout()
    fig.savefig(plots_dir / "sentiment_improvement_box.png", dpi=140)
    plt.close(fig)

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
