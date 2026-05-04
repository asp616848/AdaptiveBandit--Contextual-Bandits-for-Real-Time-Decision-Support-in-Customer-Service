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
    """
    Main figure: Resolution vs Horizon with error bars (std over seeds).
    This is the key finding that Strategy CB improves with horizon while Action CB plateaus.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    found = False
    
    # Refined color scheme with visual hierarchy
    colors = {"action_cb": "#4C72B0", "strategy_cb": "#DD8452", "ppo": "#55A868"}
    line_styles = {"action_cb": "--", "strategy_cb": "-", "ppo": "--"}
    line_widths = {"action_cb": 2, "strategy_cb": 3.5, "ppo": 2}

    strategy_data = None
    for mode, label in [("action_cb", "Action CB"), ("strategy_cb", "Strategy CB"), ("ppo", "PPO")]:
        subset = summary_df[summary_df["mode"] == mode]
        if subset.empty:
            continue

        # Group by horizon and compute mean/std over seeds
        grouped_mean = subset.groupby("horizon", as_index=True)["resolution_rate"].mean().sort_index()
        grouped_std = subset.groupby("horizon", as_index=True)["resolution_rate"].std().sort_index()
        
        if grouped_mean.empty:
            continue

        # Plot with error bars
        ax.errorbar(
            grouped_mean.index, 
            grouped_mean.values, 
            yerr=grouped_std.values,
            marker="o",
            markersize=8,
            linewidth=line_widths.get(mode, 2),
            capsize=5,
            capthick=1.5,
            label=label,
            color=colors.get(mode),
            linestyle=line_styles.get(mode),
            alpha=0.9,
            zorder=3 if mode == "strategy_cb" else 2
        )
        
        # Store Strategy CB data for annotation
        if mode == "strategy_cb":
            strategy_data = (grouped_mean, grouped_std)
        
        found = True

    # Annotate peak Strategy CB result
    if strategy_data is not None:
        mean_vals, std_vals = strategy_data
        peak_idx = mean_vals.idxmax()
        peak_val = mean_vals[peak_idx]
        ax.annotate(
            f"Peak: {peak_val:.1%}",
            xy=(peak_idx, peak_val),
            xytext=(peak_idx - 2, peak_val + 0.025),
            fontsize=10,
            fontweight="bold",
            color="#DD8452",
            arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.5, alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#DD8452", linewidth=1.5, alpha=0.9)
        )

    horizons = sorted(summary_df["horizon"].dropna().unique().tolist())
    ax.set_xlabel("Horizon (turns)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Resolution Rate", fontsize=13, fontweight="bold")
    ax.set_title(
        "Strategy-Level Bandits Improve with Horizon\nWhile Action-Level Bandits Plateau",
        fontsize=14,
        fontweight="bold",
        pad=20
    )
    if horizons:
        ax.set_xticks(horizons)
    
    # Tight y-axis to emphasize differences
    y_min, y_max = 0.0, max(0.2, summary_df["resolution_rate"].max() * 1.15)
    ax.set_ylim(y_min, y_max)
    
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    if found:
        ax.legend(fontsize=12, loc="upper left", framealpha=0.95, edgecolor="black", fancybox=True)
    fig.tight_layout()
    fig.savefig(plots_dir / "01_main_resolution_vs_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved: 01_main_resolution_vs_horizon.png (Figure 1)")


def plot_model_comparison(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Secondary figure: Compare Action CB vs Strategy CB vs Hybrid with error bars and value labels.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    order = ["action_cb", "strategy_cb", "truncated_hybrid"]
    labels = ["Action CB", "Strategy CB", "Hybrid"]
    values = []
    stds = []
    used_labels = []

    for mode, label in zip(order, labels):
        subset = summary_df[summary_df["mode"] == mode]
        if subset.empty:
            continue
        mean_val = float(subset["resolution_rate"].mean())
        std_val = float(subset["resolution_rate"].std())
        values.append(mean_val)
        stds.append(std_val)
        used_labels.append(label)

    if not values:
        ax.set_title("Model Comparison")
        fig.tight_layout()
        fig.savefig(plots_dir / "02_model_comparison.png", dpi=150)
        plt.close(fig)
        return

    # Bar chart with refined colors
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(used_labels, values, color=colors[:len(used_labels)], alpha=0.8, edgecolor="black", linewidth=1.5)
    
    # Add error bars
    ax.errorbar(
        range(len(values)),
        values,
        yerr=stds,
        fmt="none",
        color="black",
        capsize=6,
        capthick=2.5,
        elinewidth=2,
        alpha=0.8
    )
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, values, stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + std + 0.008,
            f"{val:.1%}\n±{std:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )
    
    ax.set_ylabel("Resolution Rate", fontsize=13, fontweight="bold")
    ax.set_title("Figure 2: Method Comparison", fontsize=14, fontweight="bold", pad=20)
    
    # Tighter y-axis to reduce empty space
    y_max = max(0.1, max(values) + max(stds) + 0.03)
    ax.set_ylim(0.0, y_max)
    
    ax.grid(True, alpha=0.4, axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(plots_dir / "02_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved: 02_model_comparison.png (Figure 2)")


def plot_reward_vs_horizon(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Supporting analysis: Reward signal vs Horizon.
    Note: This is secondary - the main story is resolution rate, not reward.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    found = False

    reward_column = "mean_episode_reward" if "mean_episode_reward" in summary_df.columns else "avg_per_turn_reward"
    colors = {"action_cb": "#4C72B0", "strategy_cb": "#DD8452", "truncated_hybrid": "#55A868"}
    line_widths = {"action_cb": 2, "strategy_cb": 2.5, "truncated_hybrid": 2}

    for mode, label in [("action_cb", "Action CB"), ("strategy_cb", "Strategy CB"), ("truncated_hybrid", "Hybrid")]:
        subset = summary_df[summary_df["mode"] == mode]
        if subset.empty:
            continue

        grouped_mean = subset.groupby("horizon", as_index=True)[reward_column].mean().sort_index()
        grouped_std = subset.groupby("horizon", as_index=True)[reward_column].std().sort_index()
        
        if grouped_mean.empty:
            continue

        ax.errorbar(
            grouped_mean.index,
            grouped_mean.values,
            yerr=grouped_std.values,
            marker="o",
            markersize=6,
            linewidth=line_widths.get(mode, 2),
            capsize=4,
            capthick=1.5,
            label=label,
            color=colors.get(mode),
            alpha=0.85
        )
        found = True

    horizons = sorted(summary_df["horizon"].dropna().unique().tolist())
    ax.set_xlabel("Horizon (turns)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward", fontsize=12, fontweight="bold")
    ax.set_title("Supporting Analysis: Reward vs Horizon", fontsize=13, fontweight="bold", pad=20)
    if horizons:
        ax.set_xticks(horizons)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    if found:
        ax.legend(fontsize=11, loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(plots_dir / "03_supporting_reward_vs_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved: 03_supporting_reward_vs_horizon.png (Supporting)")


def plot_algorithm_comparison(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Appendix figure: Algorithm comparison (LinUCB vs Thompson vs Epsilon).
    Note: This is NOT the main story - main story is horizon + decision structure.
    """
    algorithms = [
        ("linucb", "LinUCB"),
        ("thompson", "Thompson"),
        ("epsilon", "Epsilon Greedy"),
    ]
    mode_panels = [
        ("action_cb", "Action CB"),
        ("strategy_cb", "Strategy CB"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    if len(mode_panels) == 1:
        axes = [axes]

    any_data = False
    for ax, (mode, mode_label) in zip(axes, mode_panels):
        subset = summary_df[summary_df["mode"] == mode]
        if subset.empty:
            ax.set_title(mode_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Horizon", fontsize=11)
            continue

        found = False
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for (algorithm, label), color in zip(algorithms, colors):
            algo_subset = subset[subset["algorithm"] == algorithm]
            if algo_subset.empty:
                continue

            grouped_mean = algo_subset.groupby("horizon", as_index=True)["resolution_rate"].mean().sort_index()
            grouped_std = algo_subset.groupby("horizon", as_index=True)["resolution_rate"].std().sort_index()
            
            if grouped_mean.empty:
                continue

            ax.errorbar(
                grouped_mean.index,
                grouped_mean.values,
                yerr=grouped_std.values,
                marker="o",
                markersize=6,
                linewidth=2,
                capsize=4,
                capthick=1.5,
                label=label,
                color=color,
                alpha=0.75
            )
            found = True
            any_data = True

        horizons = sorted(subset["horizon"].dropna().unique().tolist())
        ax.set_title(mode_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Horizon", fontsize=11)
        ax.set_ylabel("Resolution Rate", fontsize=11)
        if horizons:
            ax.set_xticks(horizons)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3, linestyle="--")
        if found:
            ax.legend(fontsize=10)

    fig.suptitle("Appendix: Algorithm Comparison (NOT main story)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "04_appendix_algorithm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    if any_data:
        print("✓ Saved: 04_appendix_algorithm_comparison.png (Appendix)")


def main() -> None:
    # Set publication-quality font defaults
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "font.family": "sans-serif"
    })
    
    args = parse_args()
    run_dir = Path(args.run_dir)
    plots_dir = _ensure_plots_dir(run_dir)
    summary_df, aggregate_df = _load_tables(run_dir)

    if summary_df.empty:
        raise ValueError(f"summary.csv is empty: {run_dir / 'summary.csv'}")

    print("\n" + "="*60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*60)
    
    print("\n📊 Main Figures (for paper):")
    print("-" * 60)
    plot_resolution_vs_horizon(summary_df, plots_dir)
    plot_model_comparison(summary_df, plots_dir)
    
    print("\n📈 Supporting Analysis:")
    print("-" * 60)
    plot_reward_vs_horizon(summary_df, plots_dir)
    
    print("\n🔬 Appendix Figures:")
    print("-" * 60)
    plot_algorithm_comparison(summary_df, plots_dir)
    
    # Save metadata
    (plots_dir / "plot_inputs.txt").write_text(
        f"summary_rows={len(summary_df)}\n"
        f"aggregate_rows={len(aggregate_df)}\n"
        f"modes={', '.join(summary_df['mode'].unique())}\n"
        f"horizons={', '.join(map(str, sorted(summary_df['horizon'].unique())))}\n",
        encoding="utf-8",
    )

    print("\n" + "="*60)
    print(f"✅ ALL FIGURES SAVED to {plots_dir}")
    print("="*60)
    print("\n📋 Figure Summary:")
    print("  01_main_resolution_vs_horizon.png  → Figure 1 (PRIMARY)")
    print("  02_model_comparison.png             → Figure 2 (PRIMARY)")
    print("  03_supporting_reward_vs_horizon.png → Supporting Analysis")
    print("  04_appendix_algorithm_comparison.png → Appendix")
    print("\n")


if __name__ == "__main__":
    main()