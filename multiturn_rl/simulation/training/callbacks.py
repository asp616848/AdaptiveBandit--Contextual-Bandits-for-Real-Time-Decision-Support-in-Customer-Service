from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from simulation.training.curriculum import CurriculumScheduler


class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum: CurriculumScheduler, train_env, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.train_env = train_env
        self._last_stage_name = curriculum.get_current_stage_name()

    def _on_step(self) -> bool:
        current_filter = self.curriculum.get_subflow_filter(self.num_timesteps)
        current_stage_name = self.curriculum.get_current_stage_name()
        if current_stage_name != self._last_stage_name:
            self.train_env.env_method("set_subflow_filter", current_filter)
            self._last_stage_name = current_stage_name
            if self.verbose:
                sf_count = "all" if current_filter is None else str(len(current_filter))
                print(f"[Curriculum] Stage -> {current_stage_name} (subflows={sf_count})")
                self.logger.record("curriculum/stage_index", self.curriculum.current_stage_idx)
        return True


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, eval_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = int(eval_freq)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.resolution_count = 0
        self.escalation_count = 0
        self.dropout_count = 0
        self.timeout_count = 0
        self.episode_count = 0
        self.history: list[dict[str, float]] = []

    def _extract_terminal_type(self, info: dict) -> str:
        terminal_type = str(info.get("terminal_type", ""))
        if terminal_type:
            return terminal_type

        lto = info.get("last_transition_outcome", {})
        if isinstance(lto, dict):
            return str(lto.get("terminal_type", ""))

        return ""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))

            terminal_type = self._extract_terminal_type(info)
            if terminal_type == "success":
                self.resolution_count += 1
            elif terminal_type == "escalation":
                self.escalation_count += 1
            elif terminal_type == "dropout":
                self.dropout_count += 1
            elif terminal_type == "timeout":
                self.timeout_count += 1

        if self.num_timesteps % self.eval_freq == 0 and self.episode_count > 0:
            mean_reward = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
            mean_ep_len = float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0

            total = max(
                self.resolution_count + self.escalation_count + self.dropout_count + self.timeout_count,
                1,
            )
            resolution_rate = self.resolution_count / total
            escalation_rate = self.escalation_count / total
            dropout_rate = self.dropout_count / total
            timeout_rate = self.timeout_count / total

            self.logger.record("custom/mean_reward_100ep", mean_reward)
            self.logger.record("custom/mean_episode_length_100ep", mean_ep_len)
            self.logger.record("custom/resolution_rate", resolution_rate)
            self.logger.record("custom/escalation_rate", escalation_rate)
            self.logger.record("custom/dropout_rate", dropout_rate)
            self.logger.record("custom/timeout_rate", timeout_rate)

            self.history.append(
                {
                    "timesteps": int(self.num_timesteps),
                    "mean_reward_100ep": mean_reward,
                    "mean_episode_length_100ep": mean_ep_len,
                    "resolution_rate": float(resolution_rate),
                    "escalation_rate": float(escalation_rate),
                    "dropout_rate": float(dropout_rate),
                    "timeout_rate": float(timeout_rate),
                    "episodes_seen": int(self.episode_count),
                }
            )

            if self.verbose:
                print(
                    f"Step {self.num_timesteps:,} | "
                    f"Mean reward: {mean_reward:.3f} | "
                    f"Resolution: {resolution_rate:.1%} | "
                    f"Escalation: {escalation_rate:.1%}"
                )

        return True


class BestModelCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 200,
        baseline_reward: float = 0.99,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.baseline_reward = float(baseline_reward)
        self.best_mean_reward = -np.inf
        self.baseline_beaten = False
        self.baseline_beaten_step: int | None = None
        self.eval_history: list[dict[str, float]] = []

    def _run_eval(self) -> tuple[float, float, float]:
        rewards: list[float] = []
        resolved = 0
        escalated = 0

        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=90_000 + ep)
            done = False
            ep_reward = 0.0
            info = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(int(action))
                ep_reward += float(reward)
                if truncated:
                    break

            rewards.append(ep_reward)
            terminal_type = str((info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout"))
            if terminal_type == "success":
                resolved += 1
            if terminal_type == "escalation":
                escalated += 1

        mean_reward = float(np.mean(rewards))
        resolution_rate = float(resolved / max(self.n_eval_episodes, 1))
        escalation_rate = float(escalated / max(self.n_eval_episodes, 1))
        return mean_reward, resolution_rate, escalation_rate

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        mean_reward, resolution_rate, escalation_rate = self._run_eval()
        self.eval_history.append(
            {
                "timesteps": int(self.num_timesteps),
                "mean_reward": mean_reward,
                "resolution_rate": resolution_rate,
                "escalation_rate": escalation_rate,
            }
        )

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/resolution_rate", resolution_rate)
        self.logger.record("eval/escalation_rate", escalation_rate)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(str(self.save_path / "best_model"))
            if self.verbose:
                print(f"New best model saved: {mean_reward:.3f}")

        if mean_reward > self.baseline_reward and not self.baseline_beaten:
            self.baseline_beaten = True
            self.baseline_beaten_step = int(self.num_timesteps)
            self.model.save(str(self.save_path / "baseline_beating_model"))
            print(
                f"BASELINE BEATEN at step {self.num_timesteps:,}! "
                f"Mean reward: {mean_reward:.3f} > {self.baseline_reward:.3f}"
            )

        return True
