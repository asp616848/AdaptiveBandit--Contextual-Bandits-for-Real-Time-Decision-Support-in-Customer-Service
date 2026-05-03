#!/usr/bin/env python3
"""
Phase 13 Training Analysis & Visualization

Extract training metrics from TensorBoard event files and create professional plots
to analyze training progress and inform reward function improvements.

Usage:
    python phase13_training_analysis.py --run PPO_1 --output analysis
    python phase13_training_analysis.py --run PPO_2 --compare
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
except ImportError:
    print("⚠  TensorFlow not available. Install: pip install tensorflow")
    print("   Alternatively, use --json mode if logs are saved as JSON")
    summary_iterator = None


def extract_tensorboard_metrics(event_file_path):
    """Extract metrics from TensorBoard event file"""
    if summary_iterator is None:
        raise RuntimeError("TensorFlow required. Install: pip install tensorflow")
    
    metrics = defaultdict(list)
    
    try:
        for summary in summary_iterator(str(event_file_path)):
            step = summary.step
            
            for value in summary.summary.value:
                tag = value.tag
                
                # Extract scalar values
                if value.HasField('simple_value'):
                    metrics[tag].append((step, value.simple_value))
    
    except Exception as e:
        print(f"⚠  Error reading event file {event_file_path}: {e}")
        return {}
    
    return metrics


def load_training_logs(run_name="PPO_1"):
    """Load training metrics from tensorboard folder"""
    tb_dir = Path("RL Out/tensorboard") / run_name
    
    if not tb_dir.exists():
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")
    
    print(f"📊 Loading training logs from: {tb_dir}")
    
    all_metrics = defaultdict(list)
    
    # Find all event files
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    print(f"   Found {len(event_files)} event file(s)")
    
    for event_file in sorted(event_files):
        metrics = extract_tensorboard_metrics(event_file)
        for key, values in metrics.items():
            all_metrics[key].extend(values)
    
    # Convert to DataFrames
    df_dict = {}
    for key, values in all_metrics.items():
        if values:
            steps, vals = zip(*values)
            df_dict[key] = pd.DataFrame({
                'step': steps,
                'value': vals
            }).drop_duplicates('step').sort_values('step')
    
    print(f"   Extracted {len(df_dict)} metrics")
    return df_dict


def create_plots(df_dict, output_dir="phase13_training_analysis"):
    """Create comprehensive training analysis plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📈 Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # ====================================================================
    # 1. Mean Reward Over Time
    # ====================================================================
    if 'rollout/ep_rew_mean' in df_dict:
        fig, ax = plt.subplots(figsize=(12, 6))
        df = df_dict['rollout/ep_rew_mean']
        ax.plot(df['step'], df['value'], linewidth=2, color='#3498db', label='Episode Reward')
        ax.fill_between(df['step'], df['value'].min(), df['value'], alpha=0.2, color='#3498db')
        ax.set_xlabel('Training Step', fontsize=12, weight='bold')
        ax.set_ylabel('Mean Episode Reward', fontsize=12, weight='bold')
        ax.set_title('Training Progress: Mean Reward Over Time', fontsize=14, weight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_mean_reward.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Mean reward plot")
    
    # ====================================================================
    # 2. Resolution Rate Over Time
    # ====================================================================
    if 'eval/resolution_rate' in df_dict:
        fig, ax = plt.subplots(figsize=(12, 6))
        df = df_dict['eval/resolution_rate']
        ax.plot(df['step'], df['value'] * 100, linewidth=2.5, color='#2ecc71', marker='o', markersize=4, label='Resolution Rate')
        ax.fill_between(df['step'], df['value'].min() * 100, df['value'] * 100, alpha=0.2, color='#2ecc71')
        ax.set_xlabel('Training Step', fontsize=12, weight='bold')
        ax.set_ylabel('Resolution Rate (%)', fontsize=12, weight='bold')
        ax.set_title('Model Performance: Resolution Rate Over Training', fontsize=14, weight='bold', pad=15)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_resolution_rate.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Resolution rate plot")
    
    # ====================================================================
    # 3. Escalation Rate Over Time
    # ====================================================================
    if 'eval/escalation_rate' in df_dict:
        fig, ax = plt.subplots(figsize=(12, 6))
        df = df_dict['eval/escalation_rate']
        ax.plot(df['step'], df['value'] * 100, linewidth=2.5, color='#e74c3c', marker='s', markersize=4, label='Escalation Rate')
        ax.fill_between(df['step'], 0, df['value'] * 100, alpha=0.2, color='#e74c3c')
        ax.set_xlabel('Training Step', fontsize=12, weight='bold')
        ax.set_ylabel('Escalation Rate (%)', fontsize=12, weight='bold')
        ax.set_title('Model Behavior: Escalation Rate Over Training', fontsize=14, weight='bold', pad=15)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_escalation_rate.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Escalation rate plot")
    
    # ====================================================================
    # 4. Resolution vs Escalation (Combined)
    # ====================================================================
    if 'eval/resolution_rate' in df_dict and 'eval/escalation_rate' in df_dict:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_res = df_dict['eval/resolution_rate']
        df_esc = df_dict['eval/escalation_rate']
        
        ax.plot(df_res['step'], df_res['value'] * 100, linewidth=2.5, label='Resolution', color='#2ecc71', marker='o')
        ax.plot(df_esc['step'], df_esc['value'] * 100, linewidth=2.5, label='Escalation', color='#e74c3c', marker='s')
        
        ax.set_xlabel('Training Step', fontsize=12, weight='bold')
        ax.set_ylabel('Rate (%)', fontsize=12, weight='bold')
        ax.set_title('Resolution vs Escalation During Training', fontsize=14, weight='bold', pad=15)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_resolution_vs_escalation.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Resolution vs Escalation plot")
    
    # ====================================================================
    # 5. Loss Metrics (Policy Loss, Value Loss, etc.)
    # ====================================================================
    loss_keys = [k for k in df_dict.keys() if 'loss' in k.lower()]
    if loss_keys:
        fig, axes = plt.subplots(len(loss_keys), 1, figsize=(12, 5 * len(loss_keys)))
        if len(loss_keys) == 1:
            axes = [axes]
        
        for idx, key in enumerate(loss_keys):
            df = df_dict[key]
            axes[idx].plot(df['step'], df['value'], linewidth=2, color='#9b59b6', alpha=0.8)
            axes[idx].fill_between(df['step'], df['value'].min(), df['value'], alpha=0.2, color='#9b59b6')
            axes[idx].set_ylabel(key.replace('train/', '').replace('_', ' ').title(), fontsize=11, weight='bold')
            axes[idx].grid(alpha=0.3)
        
        axes[-1].set_xlabel('Training Step', fontsize=12, weight='bold')
        fig.suptitle('Loss Metrics Over Training', fontsize=14, weight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_loss_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Loss metrics plot")
    
    # ====================================================================
    # 6. Episode Length Over Time
    # ====================================================================
    if 'rollout/ep_len_mean' in df_dict:
        fig, ax = plt.subplots(figsize=(12, 6))
        df = df_dict['rollout/ep_len_mean']
        ax.plot(df['step'], df['value'], linewidth=2, color='#f39c12', marker='D', markersize=4)
        ax.fill_between(df['step'], df['value'].min(), df['value'], alpha=0.2, color='#f39c12')
        ax.set_xlabel('Training Step', fontsize=12, weight='bold')
        ax.set_ylabel('Mean Episode Length (turns)', fontsize=12, weight='bold')
        ax.set_title('Efficiency: Average Turns Per Episode', fontsize=14, weight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/06_episode_length.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Episode length plot")
    
    # ====================================================================
    # 7. KL Divergence (Policy Stability)
    # ====================================================================
    if 'train/approx_kl' in df_dict:
        fig, ax = plt.subplots(figsize=(12, 6))
        df = df_dict['train/approx_kl']
        ax.semilogy(df['step'], df['value'], linewidth=2, color='#1abc9c', marker='o', markersize=3)
        ax.set_xlabel('Training Step', fontsize=12, weight='bold')
        ax.set_ylabel('Approx KL Divergence (log scale)', fontsize=12, weight='bold')
        ax.set_title('Policy Stability: KL Divergence Over Training', fontsize=14, weight='bold', pad=15)
        ax.grid(alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/07_kl_divergence.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ KL divergence plot")
    
    # ====================================================================
    # 8. Dashboard Summary (All Key Metrics)
    # ====================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    plot_idx = 0
    key_metrics = [
        ('rollout/ep_rew_mean', 'Mean Reward', '#3498db'),
        ('eval/resolution_rate', 'Resolution Rate (%)', '#2ecc71'),
        ('eval/escalation_rate', 'Escalation Rate (%)', '#e74c3c'),
        ('rollout/ep_len_mean', 'Episode Length', '#f39c12'),
        ('train/approx_kl', 'KL Divergence (log)', '#1abc9c'),
    ]
    
    for key, label, color in key_metrics:
        if key in df_dict:
            row = plot_idx // 2
            col = plot_idx % 2
            ax = fig.add_subplot(gs[row, col])
            
            df = df_dict[key]
            
            if 'rate' in key.lower():
                ax.plot(df['step'], df['value'] * 100, linewidth=2, color=color, marker='o', markersize=3)
            elif 'kl' in key.lower():
                ax.semilogy(df['step'], df['value'], linewidth=2, color=color, marker='o', markersize=3)
            else:
                ax.plot(df['step'], df['value'], linewidth=2, color=color, marker='o', markersize=3)
            
            ax.set_title(label, fontsize=11, weight='bold')
            ax.grid(alpha=0.3, which='both')
            ax.set_xlabel('Step', fontsize=9)
            
            plot_idx += 1
    
    fig.suptitle('Phase 13 Training Dashboard', fontsize=16, weight='bold', y=0.995)
    plt.savefig(f"{output_dir}/08_training_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Training dashboard")
    
    print(f"\n✅ All plots saved to: {output_dir}/")


def print_summary_stats(df_dict):
    """Print summary statistics from training"""
    print("\n" + "="*70)
    print("TRAINING SUMMARY STATISTICS")
    print("="*70 + "\n")
    
    key_metrics = [
        ('rollout/ep_rew_mean', 'Mean Episode Reward'),
        ('eval/resolution_rate', 'Resolution Rate'),
        ('eval/escalation_rate', 'Escalation Rate'),
        ('rollout/ep_len_mean', 'Mean Episode Length'),
        ('train/approx_kl', 'KL Divergence'),
    ]
    
    for key, label in key_metrics:
        if key in df_dict:
            df = df_dict[key]
            initial = df['value'].iloc[0]
            final = df['value'].iloc[-1]
            best = df['value'].max() if 'loss' not in key.lower() else df['value'].min()
            
            if 'rate' in key.lower():
                print(f"{label:30s}: Initial={initial*100:6.1f}% | Final={final*100:6.1f}% | Best={best*100:6.1f}%")
            else:
                print(f"{label:30s}: Initial={initial:10.4f} | Final={final:10.4f} | Best={best:10.4f}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 13 Training Analysis - Visualize RL training metrics"
    )
    parser.add_argument(
        "--run",
        default="PPO_1",
        help="TensorBoard run to analyze (default: PPO_1)"
    )
    parser.add_argument(
        "--output",
        default="phase13_training_analysis",
        help="Output directory for plots (default: phase13_training_analysis)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  PHASE 13 TRAINING ANALYSIS")
    print("="*70 + "\n")
    
    try:
        # Load metrics
        df_dict = load_training_logs(args.run)
        
        if not df_dict:
            print("⚠  No metrics found. Ensure TensorFlow is installed:")
            print("   pip install tensorflow")
            return
        
        # Print stats
        print_summary_stats(df_dict)
        
        # Create plots
        create_plots(df_dict, args.output)
        
        print("✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
