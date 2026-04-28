from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Work around duplicate OpenMP runtime loading on some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.env.support_env import SupportEnv
from simulation.training.action_masking import ActionMaskedEnv
from simulation.training.train_ppo import continue_ppo_from_checkpoint, train_ppo
from simulation.validation.baseline_policies import POLICY_REGISTRY
from simulation.validation.level1_statistical import run_level1
from simulation.validation.level2_transition import run_level2
from simulation.validation.level3_rl_signal import run_level3
from simulation.validation.level4_rag_coverage import run_level4


INFORMATIONAL_CHECK_KEYS = {"summary_level1_metrics", "policy_metrics"}


def _terminal_from_info(info: dict[str, Any]) -> str:
    lto = info.get("last_transition_outcome", {})
    if isinstance(lto, dict):
        return str(lto.get("terminal_type", "timeout"))
    return "timeout"


def _build_ppo_env(
    artifacts_root: str,
    nlg_enabled: bool,
):
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=bool(nlg_enabled))
    env = ActionMaskedEnv(env)
    return env


def run_validation_suite(
    artifacts_root: str,
    level1_episodes: int,
    level2_episodes: int,
    level3_episodes: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "level1_statistical": run_level1(artifacts_root, n_episodes=int(level1_episodes)),
        "level2_transition": run_level2(artifacts_root, n_episodes=int(level2_episodes)),
        "level3_rl_signal": run_level3(artifacts_root, n_episodes=int(level3_episodes)),
        "level4_rag_coverage": run_level4(artifacts_root),
    }

    checks: list[bool] = []
    for level_key in ["level1_statistical", "level2_transition", "level3_rl_signal", "level4_rag_coverage"]:
        level = results[level_key]
        for check_name, check_result in level.items():
            if check_name in INFORMATIONAL_CHECK_KEYS:
                continue
            checks.append(bool(check_result.get("passed", False)))

    results["total_checks"] = int(len(checks))
    results["passed_checks"] = int(sum(checks))
    results["failed_checks"] = int(len(checks) - sum(checks))
    results["overall_pass"] = bool(all(checks))
    return results


def evaluate_ppo_model(
    model_path: str,
    artifacts_root: str,
    n_episodes: int,
    seed_offset: int,
    nlg_enabled: bool,
) -> dict[str, Any]:
    model = PPO.load(model_path)
    env = _build_ppo_env(
        artifacts_root=artifacts_root,
        nlg_enabled=bool(nlg_enabled),
    )

    rewards: list[float] = []
    outcomes = {"success": 0, "escalation": 0, "dropout": 0, "timeout": 0}
    turns_to_resolution: list[int] = []
    turns_all: list[int] = []
    action_counts = {name: 0 for name in env.ACTION_NAMES.values()}
    escalation_by_tier = {"Free": 0, "Pro": 0, "Business": 0, "Enterprise": 0}
    tier_counts = {"Free": 0, "Pro": 0, "Business": 0, "Enterprise": 0}
    persona_counts: dict[str, int] = {}
    persona_resolution_counts: dict[str, int] = {}

    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=int(seed_offset + ep))
        done = False
        ep_reward = 0.0
        info: dict[str, Any] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_id = int(action)
            action_name = env.ACTION_NAMES[action_id]
            action_counts[action_name] = int(action_counts[action_name]) + 1

            obs, reward, done, truncated, info = env.step(action_id)
            ep_reward += float(reward)
            if truncated:
                break

        terminal = _terminal_from_info(info)
        rewards.append(ep_reward)
        turns_all.append(int(env.state.get("turn_count", 0)))

        tier = str(getattr(env, "state", {}).get("tier", "Free"))
        if tier in tier_counts:
            tier_counts[tier] += 1

        persona = str(getattr(env, "state", {}).get("persona_label", "unknown"))
        persona_counts[persona] = int(persona_counts.get(persona, 0)) + 1

        if terminal == "success":
            outcomes["success"] += 1
            turns_to_resolution.append(int(getattr(env, "state", {}).get("turn_count", 0)))
            persona_resolution_counts[persona] = int(persona_resolution_counts.get(persona, 0)) + 1
        elif terminal == "escalation":
            outcomes["escalation"] += 1
            if tier in escalation_by_tier:
                escalation_by_tier[tier] += 1
        elif terminal == "dropout":
            outcomes["dropout"] += 1
        else:
            outcomes["timeout"] += 1

    env.close()

    n = float(max(int(n_episodes), 1))
    persona_resolution_rate = {
        persona: float(persona_resolution_counts.get(persona, 0) / max(count, 1))
        for persona, count in persona_counts.items()
    }

    return {
        "model_path": str(model_path),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "resolution_rate": float(outcomes["success"] / n),
        "escalation_rate": float(outcomes["escalation"] / n),
        "dropout_rate": float(outcomes["dropout"] / n),
        "timeout_rate": float(outcomes["timeout"] / n),
        "mean_turns": float(np.mean(turns_all)) if turns_all else 0.0,
        "p90_turns": float(np.percentile(turns_all, 90)) if turns_all else 0.0,
        "mean_turns_to_resolution": float(np.mean(turns_to_resolution)) if turns_to_resolution else 0.0,
        "action_distribution": {
            name: float(count / max(sum(action_counts.values()), 1))
            for name, count in action_counts.items()
        },
        "escalation_by_tier": {
            tier: float(escalation_by_tier[tier] / max(tier_counts[tier], 1)) for tier in tier_counts
        },
        "persona_resolution_rate": persona_resolution_rate,
    }


def evaluate_baselines(artifacts_root: str, n_episodes: int, seed_offset: int) -> dict[str, Any]:
    env = ActionMaskedEnv(SupportEnv(artifacts_root=artifacts_root, nlg_enabled=False))
    baselines = ["document_guided", "threshold_escalate", "random", "always_solve", "always_escalate"]
    results: dict[str, Any] = {}

    for policy_name in baselines:
        policy_fn = POLICY_REGISTRY[policy_name]
        rng = np.random.default_rng(123)

        rewards: list[float] = []
        outcomes = {"success": 0, "escalation": 0, "dropout": 0, "timeout": 0}
        turns_to_resolution: list[int] = []
        turns_all: list[int] = []

        for ep in range(int(n_episodes)):
            obs, _ = env.reset(seed=int(seed_offset + ep))
            done = False
            info: dict[str, Any] = {}
            ep_reward = 0.0

            while not done:
                state = dict(env.state)
                try:
                    action = int(policy_fn(obs, state, rng))
                except TypeError:
                    action = int(policy_fn(obs, state))
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)
                if truncated:
                    break

            terminal = _terminal_from_info(info)
            rewards.append(ep_reward)
            turns_all.append(int(env.state.get("turn_count", 0)))

            if terminal == "success":
                outcomes["success"] += 1
                turns_to_resolution.append(int(env.state.get("turn_count", 0)))
            elif terminal == "escalation":
                outcomes["escalation"] += 1
            elif terminal == "dropout":
                outcomes["dropout"] += 1
            else:
                outcomes["timeout"] += 1

        n = float(max(int(n_episodes), 1))
        results[policy_name] = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "resolution_rate": float(outcomes["success"] / n),
            "escalation_rate": float(outcomes["escalation"] / n),
            "dropout_rate": float(outcomes["dropout"] / n),
            "timeout_rate": float(outcomes["timeout"] / n),
            "mean_turns": float(np.mean(turns_all)) if turns_all else 0.0,
            "mean_turns_to_resolution": float(np.mean(turns_to_resolution)) if turns_to_resolution else 0.0,
        }

    env.close()
    return results


def generate_demo_rollouts(
    model_path: str,
    artifacts_root: str,
    output_file: Path,
    n_episodes: int = 10,
    seed_offset: int = 99000,
    nlg_enabled: bool = False,
) -> None:
    model = PPO.load(model_path)
    env = _build_ppo_env(
        artifacts_root=artifacts_root,
        nlg_enabled=bool(nlg_enabled),
    )
    rows: list[dict[str, Any]] = []

    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=int(seed_offset + ep))
        done = False
        trajectory: list[dict[str, Any]] = []
        info: dict[str, Any] = {}

        while not done:
            pre = dict(getattr(env, "state", {}))
            action, _ = model.predict(obs, deterministic=True)
            action_id = int(action)
            obs, reward, done, truncated, info = env.step(action_id)
            post = dict(getattr(env, "state", {}))
            transition = dict(info.get("last_transition_outcome", {}) or {})

            trajectory.append(
                {
                    "action": env.ACTION_NAMES[action_id],
                    "reward": float(reward),
                    "pre_information": float(pre.get("information", 0.0)),
                    "pre_progress": float(pre.get("progress", 0.0)),
                    "pre_frustration": float(pre.get("frustration", 0.0)),
                    "post_information": float(post.get("information", 0.0)),
                    "post_progress": float(post.get("progress", 0.0)),
                    "post_frustration": float(post.get("frustration", 0.0)),
                    "terminal_type": transition.get("terminal_type"),
                }
            )

            if truncated:
                break

        rows.append(
            {
                "episode": int(ep),
                "subflow": str(env.state.get("subflow", "")),
                "tier": str(env.state.get("tier", "")),
                "persona": str(env.state.get("persona_label", "")),
                "turn_count": int(env.state.get("turn_count", 0)),
                "resolved": int(env.state.get("resolved", 0)),
                "escalated": int(env.state.get("escalated", 0)),
                "dropped_off": int(env.state.get("dropped_off", 0)),
                "terminal_type": _terminal_from_info(info),
                "trajectory": trajectory,
            }
        )

    env.close()
    output_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def pick_checkpoint(artifacts_root: Path, preferred: str) -> str | None:
    if preferred and preferred.lower() != "auto":
        p = Path(preferred)
        if p.exists():
            return str(p)
        return None

    candidates = [
        artifacts_root / "phase10_final" / "models" / "best_model.zip",
        artifacts_root / "phase10_final" / "models" / "final_model.zip",
        artifacts_root / "phase10_v3" / "models" / "best_model.zip",
        artifacts_root / "phase10_v3" / "models" / "final_model.zip",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def find_model_in_output(artifacts_root: Path, output_subdir: str) -> str:
    candidates = [
        artifacts_root / output_subdir / "models" / "best_model.zip",
        artifacts_root / output_subdir / "models" / "final_model.zip",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"No model found in output directory for subdir={output_subdir}")


def train_agent(
    artifacts_root: Path,
    output_subdir: str,
    timesteps: int,
    continue_from: str | None,
    n_envs: int,
    seed: int,
    nlg_enabled: bool,
) -> dict[str, Any]:
    if continue_from:
        summary = continue_ppo_from_checkpoint(
            artifacts_root=str(artifacts_root),
            checkpoint_path=str(continue_from),
            additional_timesteps=int(timesteps),
            use_curriculum=True,
            use_reward_shaping=False,
            output_subdir=output_subdir,
            n_envs=int(n_envs),
            seed=int(seed),
            tb_log_name=f"ppo_{output_subdir}",
            nlg_enabled=bool(nlg_enabled),
        )
        summary["training_mode"] = "continue"
        return summary

    summary = train_ppo(
        artifacts_root=str(artifacts_root),
        timesteps=int(timesteps),
        use_curriculum=True,
        use_reward_shaping=False,
        output_subdir=output_subdir,
        n_envs=int(n_envs),
        seed=int(seed),
        nlg_enabled=bool(nlg_enabled),
    )
    summary["training_mode"] = "fresh"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Full simulator validation + PPO train/eval runner")
    parser.add_argument("--artifacts-root", type=str, default="simulation/artifacts")
    parser.add_argument("--output-subdir", type=str, default="phase10_prod")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--continue-from", type=str, default="auto")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=1500)
    parser.add_argument("--validation-level1-episodes", type=int, default=1500)
    parser.add_argument("--validation-level2-episodes", type=int, default=1200)
    parser.add_argument("--validation-level3-episodes", type=int, default=1500)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--demo-episodes", type=int, default=12)
    parser.add_argument("--nlg-enabled", action="store_true")
    # LLM backend selection (used when --nlg-enabled)
    parser.add_argument("--llm-backend", type=str, choices=["ollama", "hf"], default="ollama",
                        help="LLM backend for NLG: 'ollama' (default) or 'hf' (local HuggingFace model)")
    parser.add_argument("--ollama-model", type=str, default="",
                        help="Ollama model name (e.g. llama3, qwen2.5:7b). Falls back to OLLAMA_MODEL env var.")
    parser.add_argument("--hf-model-path", type=str, default="",
                        help="Path to local HF model dir. Falls back to HF_MODEL_PATH env var.")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    run_root = artifacts_root / args.output_subdir
    run_root.mkdir(parents=True, exist_ok=True)

    # Wire up LLM backend env vars so NLGLayer / backends.py pick them up.
    if args.nlg_enabled:
        backend = str(args.llm_backend or "ollama").strip().lower()
        os.environ["SUPPORT_SIM_LLM_BACKEND"] = backend
        if backend == "hf":
            hf_path = str(args.hf_model_path or os.getenv("HF_MODEL_PATH", "")).strip()
            if hf_path:
                os.environ["SUPPORT_SIM_HF_MODEL_PATH"] = hf_path
        else:
            model = str(args.ollama_model or os.getenv("OLLAMA_MODEL", "llama3")).strip()
            endpoint = os.getenv("OLLAMA_ENDPOINT", os.getenv("SUPPORT_SIM_LLM_ENDPOINT", "http://localhost:11434/v1"))
            os.environ["SUPPORT_SIM_LLM_MODEL"] = model
            os.environ["SUPPORT_SIM_LLM_ENDPOINT"] = endpoint

    start = time.time()
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
    }

    print("=" * 72)
    print("Phase 10 Full Pipeline: validate -> train -> evaluate -> demos")
    print("=" * 72)

    if not args.skip_validation:
        print("\n[1/4] Running simulator validation suite...")
        validation_report = run_validation_suite(
            artifacts_root=str(artifacts_root),
            level1_episodes=int(args.validation_level1_episodes),
            level2_episodes=int(args.validation_level2_episodes),
            level3_episodes=int(args.validation_level3_episodes),
        )
        report["validation"] = validation_report
        (run_root / "validation_report_pretrain.json").write_text(
            json.dumps(validation_report, indent=2),
            encoding="utf-8",
        )
        print(
            f"Validation pass rate: {validation_report['passed_checks']}/{validation_report['total_checks']} "
            f"(overall_pass={validation_report['overall_pass']})"
        )
    else:
        report["validation"] = {"skipped": True}

    checkpoint = pick_checkpoint(artifacts_root, args.continue_from)
    report["selected_checkpoint"] = checkpoint

    if not args.skip_training:
        print("\n[2/4] Training PPO agent...")
        if checkpoint:
            print(f"Training mode: continue from {checkpoint}")
        else:
            print("Training mode: fresh")

        train_summary = train_agent(
            artifacts_root=artifacts_root,
            output_subdir=args.output_subdir,
            timesteps=int(args.timesteps),
            continue_from=checkpoint,
            n_envs=int(args.n_envs),
            seed=int(args.seed),
            nlg_enabled=bool(args.nlg_enabled),
        )
        report["training"] = train_summary
        print(
            f"Training done: best_eval_reward={train_summary.get('best_eval_reward', 0.0):.4f}, "
            f"final_resolution_rate={train_summary.get('final_resolution_rate', 0.0):.4f}"
        )
    else:
        report["training"] = {"skipped": True}

    model_path = find_model_in_output(artifacts_root, args.output_subdir)
    report["model_path"] = model_path

    print("\n[3/4] Evaluating PPO vs baselines...")
    ppo_metrics = evaluate_ppo_model(
        model_path=model_path,
        artifacts_root=str(artifacts_root),
        n_episodes=int(args.eval_episodes),
        seed_offset=60_000,
        nlg_enabled=bool(args.nlg_enabled),
    )
    baseline_metrics = evaluate_baselines(
        artifacts_root=str(artifacts_root),
        n_episodes=int(args.eval_episodes),
        seed_offset=90_000,
    )

    leaderboard = {"ppo_trained": ppo_metrics}
    leaderboard.update(baseline_metrics)
    sorted_policies = sorted(leaderboard.items(), key=lambda kv: kv[1]["mean_reward"], reverse=True)

    report["evaluation"] = {
        "ppo": ppo_metrics,
        "baselines": baseline_metrics,
        "leaderboard": [{"policy": k, "mean_reward": float(v["mean_reward"])} for k, v in sorted_policies],
        "ppo_beats_document_guided": bool(ppo_metrics["mean_reward"] > baseline_metrics["document_guided"]["mean_reward"]),
    }

    print("Leaderboard by mean reward:")
    for rank, (name, metrics) in enumerate(sorted_policies, start=1):
        print(
            f"  {rank}. {name:<18} reward={metrics['mean_reward']:.3f} "
            f"resolve={metrics['resolution_rate']:.1%} escalate={metrics['escalation_rate']:.1%}"
        )

    print("\n[4/4] Generating demo rollouts...")
    demo_path = run_root / "demo_rollouts.json"
    generate_demo_rollouts(
        model_path=model_path,
        artifacts_root=str(artifacts_root),
        output_file=demo_path,
        n_episodes=int(args.demo_episodes),
        seed_offset=120_000,
        nlg_enabled=bool(args.nlg_enabled),
    )
    report["demo_rollouts_path"] = str(demo_path)

    elapsed = float(time.time() - start)
    report["runtime_seconds"] = elapsed

    report_path = run_root / "full_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nArtifacts written:")
    print(f"  - {report_path}")
    if not args.skip_validation:
        print(f"  - {run_root / 'validation_report_pretrain.json'}")
    print(f"  - {demo_path}")
    print(f"  - {artifacts_root / args.output_subdir / 'models' / 'best_model.zip'}")
    print(f"\nRuntime: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
