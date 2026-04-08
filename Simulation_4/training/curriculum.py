from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
            self.subflow_difficulties = self._load_fallback_difficulties(root)
            return

        df = pd.read_csv(csv_path)
        mean_action = float(df["mean_action_count"].mean())
        denom = max(mean_action, 1e-8)

        self.subflow_difficulties = {
            str(row["subflow"]): float(row["mean_action_count"]) / denom
            for _, row in df.iterrows()
        }

    def _load_fallback_difficulties(self, root: Path) -> dict[str, float]:
        psuccess_candidates = [
            root / "phase 4" / "psuccess_model.json",
            root / "phase4" / "psuccess_model.json",
        ]

        psuccess_path = None
        for candidate in psuccess_candidates:
            if candidate.exists():
                psuccess_path = candidate
                break

        if psuccess_path is None:
            return {"generic_support": 1.0}

        payload = json.loads(psuccess_path.read_text(encoding="utf-8"))
        offsets = payload.get("subflow_offsets", {})
        subflows = sorted(str(k) for k in offsets.keys())
        if not subflows:
            return {"generic_support": 1.0}

        values = np.array([float(offsets.get(sf, 0.0)) for sf in subflows], dtype=float)
        if len(values) <= 1 or float(np.max(values) - np.min(values)) < 1e-9:
            hardness = np.full_like(values, 0.5)
        else:
            hardness = (float(np.max(values)) - values) / (float(np.max(values)) - float(np.min(values)))

        return {
            sf: float(0.7 + 0.9 * hard)
            for sf, hard in zip(subflows, hardness)
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
