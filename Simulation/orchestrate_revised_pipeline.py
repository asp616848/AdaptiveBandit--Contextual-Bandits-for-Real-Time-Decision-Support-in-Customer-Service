from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.simulation_core.data.data_pipeline_v2 import run_pipeline
from src.simulation_core.persona.persona_model import train_persona_cvae
from src.simulation_core.rewards.reward_model import train_reward_model
from src.simulation_core.training.training_pipeline import run_training_pipeline
from src.simulation_core.evaluation.validation_suite import ValidationSuite


def main() -> None:
    outputs = run_pipeline()
    twitter = pd.read_csv(outputs["twitter_labeled"])
    opena = pd.read_csv(outputs["openassistant_labeled"])
    labeled = pd.concat([twitter, opena], ignore_index=True)

    artifacts_root = Path(__file__).resolve().parent / "artifacts" / "revised_pipeline"
    persona_dir = artifacts_root / "persona"
    reward_dir = artifacts_root / "reward"
    training_dir = artifacts_root / "training"
    eval_dir = artifacts_root / "validation"

    persona_metrics = train_persona_cvae(labeled, persona_dir)
    reward_metrics = train_reward_model(labeled, reward_dir)
    train_metrics = run_training_pipeline(labeled, reward_dir / "reward_weights.npy", training_dir)

    # Proxy simulated df for validation run-through using current labeled set shape.
    suite = ValidationSuite(real_df=labeled[labeled["annotator_confidence"].fillna(0.0) >= 0.65])
    simulated_df = labeled.copy()
    validation = suite.run(simulated_df=simulated_df, out_dir=eval_dir)

    summary = {
        "pipeline_outputs": {k: str(v) for k, v in outputs.items()},
        "persona": persona_metrics,
        "reward": reward_metrics,
        "training": train_metrics,
        "validation": validation,
    }

    summary_path = artifacts_root / "orchestration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
