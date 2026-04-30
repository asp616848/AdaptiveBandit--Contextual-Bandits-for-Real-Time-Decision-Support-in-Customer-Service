from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Simulation_4.training.train_ppo import continue_ppo_from_checkpoint, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO with Simulation_4 and fine-tuned Qwen customer model.")
    parser.add_argument("--artifacts-root", default="Simulation_4/artifacts")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output-subdir", default="rl_qwen_long")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=os.getenv("SUPPORT_SIM_LOCAL_MODEL_PATH", "Qwen2.5-7B-Instruct-merged"))
    parser.add_argument("--intent-model", default=os.getenv("SUPPORT_SIM_INTENT_MODEL", "local-qwen"))
    parser.add_argument("--agent-model", default=os.getenv("SUPPORT_SIM_AGENT_MODEL", "local-qwen"))
    parser.add_argument("--customer-model", default=os.getenv("SUPPORT_SIM_LLM_MODEL", "local-qwen"))
    parser.add_argument("--metrics-eval-freq", type=int, default=int(os.getenv("METRICS_EVAL_FREQ", "500")))
    parser.add_argument("--heartbeat-freq-steps", type=int, default=int(os.getenv("HEARTBEAT_FREQ_STEPS", "25")))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--no-shaping", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Local Qwen model path not found: {model_path}")

    os.environ["SUPPORT_SIM_LLM_BACKEND"] = "local"
    os.environ["SUPPORT_SIM_LOCAL_MODEL_PATH"] = str(model_path)
    os.environ["SUPPORT_SIM_LLM_MODEL"] = args.customer_model
    os.environ["SUPPORT_SIM_AGENT_MODEL"] = args.agent_model
    os.environ["SUPPORT_SIM_INTENT_MODEL"] = args.intent_model

    kwargs = {
        "artifacts_root": args.artifacts_root,
        "use_curriculum": not args.no_curriculum,
        "use_reward_shaping": not args.no_shaping,
        "output_subdir": args.output_subdir,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "nlg_enabled": True,
        "observation_mode": "nlp",
        "intent_model": args.intent_model,
        "agent_model": args.agent_model,
        "run_id": args.run_id,
        "metrics_eval_freq": args.metrics_eval_freq,
        "heartbeat_freq_steps": args.heartbeat_freq_steps,
    }

    if args.checkpoint:
        summary = continue_ppo_from_checkpoint(
            checkpoint_path=args.checkpoint,
            additional_timesteps=args.timesteps,
            **kwargs,
        )
    else:
        summary = train_ppo(
            timesteps=args.timesteps,
            **kwargs,
        )

    print(json.dumps(summary, indent=2))
    run_dir = summary.get("run_dir")
    if run_dir:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "env_config.json").write_text(
            json.dumps(
                {
                    "backend": "local",
                    "model_path": str(model_path),
                    "customer_model": args.customer_model,
                    "intent_model": args.intent_model,
                    "agent_model": args.agent_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
