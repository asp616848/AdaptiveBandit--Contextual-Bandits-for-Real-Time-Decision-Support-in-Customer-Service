from __future__ import annotations

from pathlib import Path

import pandas as pd


class CurriculumScheduler:
    def __init__(
        self,
        artifacts_root: str,
        stages: list[dict] | None = None,
    ):
        self.stages = stages or [
            {
                "name": "easy",
                "difficulty_max": 0.85,
                "min_timesteps": 0,
                "description": "Easy subflows only - high p_success",
            },
            {
                "name": "medium",
                "difficulty_max": 1.20,
                "min_timesteps": 100_000,
                "description": "Add medium difficulty subflows",
            },
            {
                "name": "full",
                "difficulty_max": 999,
                "min_timesteps": 300_000,
                "description": "All subflows including hard ones",
            },
        ]
        self.current_stage_idx = 0
        self.subflow_difficulties: dict[str, float] = {}
        self._load_subflow_difficulties(artifacts_root)

    def _load_subflow_difficulties(self, artifacts_root: str):
        root = Path(artifacts_root)
        candidates = [
            root / "phase 1" / "extract3_subflow_stats.csv",
            root / "phase1" / "extract3_subflow_stats.csv",
        ]

        csv_path = None
        for candidate in candidates:
            if candidate.exists():
                csv_path = candidate
                break

        if csv_path is None:
            raise FileNotFoundError(f"extract3_subflow_stats.csv not found under: {candidates}")

        df = pd.read_csv(csv_path)
        mean_action = float(df["mean_action_count"].mean())
        denom = max(mean_action, 1e-8)

        self.subflow_difficulties = {
            str(row["subflow"]): float(row["mean_action_count"]) / denom
            for _, row in df.iterrows()
        }

    def get_subflow_filter(self, total_timesteps: int) -> list[str] | None:
        for i, stage in enumerate(self.stages):
            if int(total_timesteps) >= int(stage["min_timesteps"]):
                if i > self.current_stage_idx:
                    self.current_stage_idx = i
                    print(f"Curriculum advanced to stage: {stage['name']}")

        stage = self.stages[self.current_stage_idx]
        if float(stage["difficulty_max"]) >= 999:
            return None

        max_diff = float(stage["difficulty_max"])
        return [sf for sf, diff in self.subflow_difficulties.items() if float(diff) <= max_diff]

    def get_current_stage_name(self) -> str:
        return str(self.stages[self.current_stage_idx]["name"])
