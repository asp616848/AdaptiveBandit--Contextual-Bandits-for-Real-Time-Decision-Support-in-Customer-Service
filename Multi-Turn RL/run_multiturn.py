from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_phase10(args: argparse.Namespace, *, text_only: bool) -> None:
    root = _repo_root()

    pipeline = root / "Multi-Turn RL" / "Simulation_4" / "scripts" / "phase10_full_pipeline.py"
    exporter = root / "tools" / "export_multiturn_results.py"

    artifacts_root = Path(args.artifacts_root)
    out_dir = Path(args.out_dir)

    _run(
        [
            sys.executable,
            str(pipeline),
            "--artifacts-root",
            str(artifacts_root),
            "--output-subdir",
            args.run_subdir,
            "--timesteps",
            str(args.timesteps),
            "--eval-episodes",
            str(args.eval_episodes),
            "--demo-episodes",
            str(args.demo_episodes),
            "--n-envs",
            str(args.n_envs),
            "--continue-from",
            "_none_",
            "--skip-validation",
        ]
        + (["--text-only-observation"] if text_only else [])
    )

    _run(
        [
            sys.executable,
            str(exporter),
            "--artifacts-root",
            str(artifacts_root),
            "--run-subdir",
            args.run_subdir,
            "--out-dir",
            str(out_dir),
        ]
    )


def cmd_phase13(args: argparse.Namespace) -> None:
    root = _repo_root()
    script = root / "Multi-Turn RL" / "Simulation_4" / "scripts" / "phase13_train.py"

    _run(
        [
            sys.executable,
            str(script),
            "--timesteps",
            str(args.timesteps),
            "--intent-model",
            args.intent_model,
            "--n-envs",
            str(args.n_envs),
        ]
        + (["--agent-model", args.agent_model] if args.agent_model else [])
        + (["--no-curriculum"] if args.no_curriculum else [])
        + (["--no-shaping"] if args.no_shaping else [])
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified runner for the Multi-Turn RL pipelines (optional; run.sh is the graded entrypoint)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p10_common = argparse.ArgumentParser(add_help=False)
    p10_common.add_argument(
        "--artifacts-root",
        default=str(Path("Multi-Turn RL") / "Simulation_4" / "artifacts"),
        help="Artifacts root directory",
    )
    p10_common.add_argument("--run-subdir", required=True, help="Subdir under artifacts_root to write this run")
    p10_common.add_argument(
        "--out-dir",
        required=True,
        help="Method output directory (will contain logs/ and plots/)",
    )
    p10_common.add_argument("--timesteps", type=int, default=20000)
    p10_common.add_argument("--eval-episodes", type=int, default=200)
    p10_common.add_argument("--demo-episodes", type=int, default=6)
    p10_common.add_argument("--n-envs", type=int, default=1)

    p_num = sub.add_parser("numerical", parents=[p10_common], help="Phase10 numerical/state observation")
    p_num.set_defaults(func=lambda a: cmd_phase10(a, text_only=False))

    p_txt = sub.add_parser("text", parents=[p10_common], help="Phase10 text-only observation (offline-safe)")
    p_txt.set_defaults(func=lambda a: cmd_phase10(a, text_only=True))

    p13 = sub.add_parser("llm", help="Phase13 full NLP agent (LLM-backed)")
    p13.add_argument("--timesteps", type=int, default=500_000)
    p13.add_argument("--intent-model", type=str, default="phi3")
    p13.add_argument("--agent-model", type=str, default=None)
    p13.add_argument("--n-envs", type=int, default=2)
    p13.add_argument("--no-curriculum", action="store_true")
    p13.add_argument("--no-shaping", action="store_true")
    p13.set_defaults(func=cmd_phase13)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
