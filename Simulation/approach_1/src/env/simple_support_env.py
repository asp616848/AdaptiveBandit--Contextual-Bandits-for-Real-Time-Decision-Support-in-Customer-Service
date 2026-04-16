from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class TaskDefinition:
    subflow: str
    canonical_actions: list[str]
    support_count: int
    empirical_success_rate: float


def load_subflow_catalog(path: str | Path) -> list[TaskDefinition]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("subflows", [])

    tasks: list[TaskDefinition] = []
    for row in rows:
        actions = [str(x) for x in row.get("canonical_actions", [])]
        if not actions:
            continue
        tasks.append(
            TaskDefinition(
                subflow=str(row.get("subflow", "unknown_subflow")),
                canonical_actions=actions,
                support_count=int(row.get("support_count", 1)),
                empirical_success_rate=float(row.get("empirical_success_rate", 0.0)),
            )
        )
    if not tasks:
        raise ValueError("Catalog contains no usable tasks")
    return tasks


class SimpleSupportEnv(gym.Env):
    """Minimal deterministic support environment (Approach 1)."""

    metadata = {"render_modes": []}

    ASK_INFO = 0
    ESCALATE = 1
    CLOSE = 2
    ACTION_OFFSET = 3

    def __init__(
        self,
        tasks: list[TaskDefinition],
        reward_config: dict[str, float],
        seed: int | None = None,
    ):
        super().__init__()
        self.tasks = tasks
        self.reward_config = reward_config

        self.subflows = sorted({t.subflow for t in self.tasks})
        self.subflow_to_idx = {s: i for i, s in enumerate(self.subflows)}

        action_vocab = sorted({a for t in self.tasks for a in t.canonical_actions})
        self.action_to_id = {a: i + self.ACTION_OFFSET for i, a in enumerate(action_vocab)}
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}

        self.action_space = gym.spaces.Discrete(self.ACTION_OFFSET + len(action_vocab))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.timeout_slack = int(self.reward_config.get("timeout_slack", 3))
        self.rng = np.random.default_rng(seed)

        self.current_task: TaskDefinition | None = None
        self.step_idx = 0
        self.turn_count = 0
        self.max_turns = 0
        self.done = False

    def _sample_task(self) -> TaskDefinition:
        weights = np.array([max(t.support_count, 1) for t in self.tasks], dtype=np.float64)
        probs = weights / weights.sum()
        idx = int(self.rng.choice(len(self.tasks), p=probs))
        return self.tasks[idx]

    def _norm(self, value: int, denom: int) -> float:
        if denom <= 0:
            return 0.0
        return float(value) / float(denom)

    def _get_obs(self) -> np.ndarray:
        assert self.current_task is not None
        subflow_idx = self.subflow_to_idx[self.current_task.subflow]
        subflow_norm = self._norm(subflow_idx, max(len(self.subflows) - 1, 1))
        progress_norm = self._norm(self.step_idx, max(len(self.current_task.canonical_actions), 1))
        max_steps_norm = min(len(self.current_task.canonical_actions) / 20.0, 1.0)
        turns_remaining = max(self.max_turns - self.turn_count, 0)
        turns_remaining_norm = self._norm(turns_remaining, max(self.max_turns, 1))

        return np.array([subflow_norm, progress_norm, max_steps_norm, turns_remaining_norm], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_task = self._sample_task()
        self.step_idx = 0
        self.turn_count = 0
        self.max_turns = len(self.current_task.canonical_actions) + self.timeout_slack
        self.done = False

        info = {
            "subflow": self.current_task.subflow,
            "required_actions": self.current_task.canonical_actions,
            "action_mapping": self.action_to_id,
        }
        return self._get_obs(), info

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        if self.current_task is None:
            raise RuntimeError("Environment not initialized. Call reset().")

        self.turn_count += 1
        terminal_reason = None

        correct_reward = float(self.reward_config.get("correct_step_reward", 1.0))
        wrong_penalty = float(self.reward_config.get("wrong_action_penalty", -1.0))
        completion_bonus = float(self.reward_config.get("completion_bonus", 5.0))
        ask_info_penalty = float(self.reward_config.get("ask_info_penalty", -0.1))
        escalate_penalty = float(self.reward_config.get("escalate_penalty", -2.0))
        close_early_penalty = float(self.reward_config.get("close_early_penalty", -3.0))
        timeout_penalty = float(self.reward_config.get("timeout_penalty", -5.0))

        reward = 0.0
        expected_action = None
        if self.step_idx < len(self.current_task.canonical_actions):
            expected_action = self.current_task.canonical_actions[self.step_idx]

        if action == self.ASK_INFO:
            reward = ask_info_penalty

        elif action == self.ESCALATE:
            reward = escalate_penalty
            self.done = True
            terminal_reason = "escalate"

        elif action == self.CLOSE:
            if self.step_idx >= len(self.current_task.canonical_actions):
                reward = completion_bonus
                terminal_reason = "close_after_completion"
            else:
                reward = close_early_penalty
                terminal_reason = "close_early"
            self.done = True

        else:
            chosen_action = self.id_to_action.get(int(action))
            if chosen_action is None:
                reward = wrong_penalty
            elif expected_action is not None and chosen_action == expected_action:
                self.step_idx += 1
                reward = correct_reward
                if self.step_idx >= len(self.current_task.canonical_actions):
                    reward += completion_bonus
                    self.done = True
                    terminal_reason = "completed"
            else:
                reward = wrong_penalty

        if not self.done and self.turn_count >= self.max_turns:
            reward = timeout_penalty
            self.done = True
            terminal_reason = "timeout"

        info = {
            "subflow": self.current_task.subflow,
            "expected_action": expected_action,
            "step_idx": self.step_idx,
            "max_steps": len(self.current_task.canonical_actions),
            "turn_count": self.turn_count,
            "terminal_reason": terminal_reason,
        }

        return self._get_obs(), float(reward), self.done, False, info

    @classmethod
    def from_files(
        cls,
        catalog_path: str | Path,
        reward_config_path: str | Path,
        seed: int | None = None,
    ) -> "SimpleSupportEnv":
        tasks = load_subflow_catalog(catalog_path)
        reward_cfg = json.loads(Path(reward_config_path).read_text(encoding="utf-8"))
        return cls(tasks=tasks, reward_config=reward_cfg, seed=seed)
