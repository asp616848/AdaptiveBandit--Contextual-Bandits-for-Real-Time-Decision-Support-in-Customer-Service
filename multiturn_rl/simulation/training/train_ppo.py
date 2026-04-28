from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from simulation.env.support_env import SupportEnv
from simulation.training.action_masking import ActionMaskedEnv
from simulation.training.callbacks import BestModelCallback, CurriculumCallback, TrainingMetricsCallback
from simulation.training.curriculum import CurriculumScheduler
from simulation.training.reward_shaping import RewardShapedWrapper, RewardShaper


PPO_CONFIG: dict[str, Any] = {
    "policy": "MlpPolicy",
    "policy_kwargs": {
        "net_arch": [128, 128, 64],
        "activation_fn": torch.nn.Tanh,
    },
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,
    "total_timesteps": 500_000,
    "n_envs": 4,
    "tensorboard_log": "simulation/artifacts/phase10/tensorboard/",
    "verbose": 1,
}


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _progress_bar_available() -> bool:
    try:
        import tqdm  # type: ignore  # noqa: F401
        import rich  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def make_env(
    artifacts_root: str,
    reward_shaper: RewardShaper,
    subflow_filter: list[str] | None = None,
    nlg_enabled: bool = False,
):
    def _init():
        env = SupportEnv(
            artifacts_root=artifacts_root,
            nlg_enabled=bool(nlg_enabled),
            subflow_filter=subflow_filter,
        )
        env = RewardShapedWrapper(env, reward_shaper)
        env = ActionMaskedEnv(env)
        env = Monitor(env)
        return env

    return _init


def train_ppo(
    artifacts_root: str,
    timesteps: int = 500_000,
    use_curriculum: bool = True,
    use_reward_shaping: bool = True,
    output_subdir: str = "phase10",
    n_envs: int = 4,
    seed: int = 42,
    nlg_enabled: bool = False,
) -> dict[str, Any]:
    artifacts_root_path = Path(artifacts_root)
    phase10_root = artifacts_root_path / output_subdir
    save_dir = phase10_root / "models"
    log_dir = phase10_root / "tensorboard"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    reward_shaper = RewardShaper(enabled=bool(use_reward_shaping), strict_potential=True)
    curriculum = CurriculumScheduler(str(artifacts_root_path)) if use_curriculum else None
    initial_filter = curriculum.get_subflow_filter(0) if curriculum is not None else None

    train_env = make_vec_env(
        make_env(
            str(artifacts_root_path),
            reward_shaper,
            initial_filter,
            nlg_enabled=bool(nlg_enabled),
        ),
        n_envs=n_envs,
        seed=seed,
    )

    eval_shaper = RewardShaper(enabled=False)
    eval_env = make_env(
        str(artifacts_root_path),
        eval_shaper,
        subflow_filter=None,
        nlg_enabled=bool(nlg_enabled),
    )()

    model = PPO(
        policy=PPO_CONFIG["policy"],
        env=train_env,
        policy_kwargs=PPO_CONFIG["policy_kwargs"],
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        ent_coef=PPO_CONFIG["ent_coef"],
        vf_coef=PPO_CONFIG["vf_coef"],
        max_grad_norm=PPO_CONFIG["max_grad_norm"],
        learning_rate=PPO_CONFIG["learning_rate"],
        tensorboard_log=str(log_dir) if _tensorboard_available() else None,
        verbose=PPO_CONFIG["verbose"],
        seed=seed,
    )

    metrics_callback = TrainingMetricsCallback(eval_freq=5000, verbose=1)
    best_model_callback = BestModelCallback(
        save_path=str(save_dir),
        eval_env=eval_env,
        eval_freq=10000,
        n_eval_episodes=200,
        baseline_reward=0.99,
        verbose=1,
    )
    callbacks = [metrics_callback, best_model_callback]

    if curriculum is not None:
        callbacks.append(CurriculumCallback(curriculum, train_env, verbose=1))

    start = time.time()
    model.learn(
        total_timesteps=int(timesteps),
        callback=callbacks,
        progress_bar=_progress_bar_available(),
    )
    elapsed = float(time.time() - start)

    model.save(str(save_dir / "final_model"))

    training_log = {
        "metrics_history": metrics_callback.history,
        "eval_history": best_model_callback.eval_history,
    }
    training_log_path = phase10_root / "training_log.json"
    training_log_path.write_text(json.dumps(training_log, indent=2), encoding="utf-8")

    summary = {
        "total_timesteps": int(timesteps),
        "training_time_seconds": elapsed,
        "curriculum_enabled": bool(use_curriculum),
        "reward_shaping_enabled": bool(use_reward_shaping),
        "nlg_enabled": bool(nlg_enabled),
        "best_eval_reward": float(best_model_callback.best_mean_reward),
        "baseline_beaten": bool(best_model_callback.baseline_beaten),
        "baseline_beaten_step": best_model_callback.baseline_beaten_step,
        "final_resolution_rate": (
            float(metrics_callback.history[-1]["resolution_rate"]) if metrics_callback.history else 0.0
        ),
        "n_envs": int(n_envs),
        "seed": int(seed),
    }

    summary_path = phase10_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_env.close()
    eval_env.close()
    return summary


def continue_ppo_from_checkpoint(
    artifacts_root: str,
    checkpoint_path: str,
    additional_timesteps: int = 500_000,
    use_curriculum: bool = True,
    use_reward_shaping: bool = False,
    output_subdir: str = "phase10_v3_continued",
    n_envs: int = 4,
    seed: int = 42,
    tb_log_name: str = "ppo_v3_continued",
    nlg_enabled: bool = False,
) -> dict[str, Any]:
    artifacts_root_path = Path(artifacts_root)
    phase_root = artifacts_root_path / output_subdir
    save_dir = phase_root / "models"
    log_dir = phase_root / "tensorboard"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    reward_shaper = RewardShaper(enabled=bool(use_reward_shaping), strict_potential=True)

    # Build env first, then initialize curriculum filter from the loaded model step count.
    train_env = make_vec_env(
        make_env(
            str(artifacts_root_path),
            reward_shaper,
            subflow_filter=None,
            nlg_enabled=bool(nlg_enabled),
        ),
        n_envs=n_envs,
        seed=seed,
    )

    model = PPO.load(checkpoint_path, env=train_env)
    # Avoid SB3 raising when tensorboard isn't installed.
    model.tensorboard_log = str(log_dir) if _tensorboard_available() else None
    initial_num_timesteps = int(getattr(model, "num_timesteps", 0))

    curriculum = CurriculumScheduler(str(artifacts_root_path)) if use_curriculum else None
    if curriculum is not None:
        initial_filter = curriculum.get_subflow_filter(initial_num_timesteps)
        train_env.env_method("set_subflow_filter", initial_filter)

    eval_shaper = RewardShaper(enabled=bool(use_reward_shaping))
    eval_env = make_env(
        str(artifacts_root_path),
        eval_shaper,
        subflow_filter=None,
        nlg_enabled=bool(nlg_enabled),
    )()

    metrics_callback = TrainingMetricsCallback(eval_freq=5000, verbose=1)
    best_model_callback = BestModelCallback(
        save_path=str(save_dir),
        eval_env=eval_env,
        eval_freq=10000,
        n_eval_episodes=200,
        baseline_reward=0.99,
        verbose=1,
    )
    callbacks = [metrics_callback, best_model_callback]

    if curriculum is not None:
        callbacks.append(CurriculumCallback(curriculum, train_env, verbose=1))

    start = time.time()
    model.learn(
        total_timesteps=int(additional_timesteps),
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name=tb_log_name,
        progress_bar=_progress_bar_available(),
    )
    elapsed = float(time.time() - start)

    model.save(str(save_dir / "final_model"))

    training_log = {
        "metrics_history": metrics_callback.history,
        "eval_history": best_model_callback.eval_history,
    }
    training_log_path = phase_root / "training_log.json"
    training_log_path.write_text(json.dumps(training_log, indent=2), encoding="utf-8")

    final_num_timesteps = int(getattr(model, "num_timesteps", initial_num_timesteps + int(additional_timesteps)))
    summary = {
        "initial_num_timesteps": initial_num_timesteps,
        "additional_timesteps": int(additional_timesteps),
        "final_num_timesteps": final_num_timesteps,
        "training_time_seconds": elapsed,
        "curriculum_enabled": bool(use_curriculum),
        "reward_shaping_enabled": bool(use_reward_shaping),
        "nlg_enabled": bool(nlg_enabled),
        "best_eval_reward": float(best_model_callback.best_mean_reward),
        "baseline_beaten": bool(best_model_callback.baseline_beaten),
        "baseline_beaten_step": best_model_callback.baseline_beaten_step,
        "final_resolution_rate": (
            float(metrics_callback.history[-1]["resolution_rate"]) if metrics_callback.history else 0.0
        ),
        "n_envs": int(n_envs),
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
    }

    summary_path = phase_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_env.close()
    eval_env.close()
    return summary
