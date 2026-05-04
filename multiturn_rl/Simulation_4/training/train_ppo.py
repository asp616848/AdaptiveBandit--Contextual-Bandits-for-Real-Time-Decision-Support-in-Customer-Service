from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from Simulation_4.env.agent_response_generator import AgentResponseGenerator
from Simulation_4.env.support_env import SupportEnv
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.training.callbacks import (
    BestModelCallback,
    CurriculumCallback,
    ProgressHeartbeatCallback,
    TrainingMetricsCallback,
)
from Simulation_4.training.curriculum import CurriculumScheduler
from Simulation_4.training.nlp_observation import NLPObservationWrapper
from Simulation_4.training.reward_shaping import RewardShapedWrapper, RewardShaper
from Simulation_4.training.text_observation import TextOnlyObservationWrapper


PPO_CONFIG: dict[str, Any] = {
    "policy": "MlpPolicy",
    "policy_kwargs": {
        "net_arch": [128, 128, 64],
        "activation_fn": torch.nn.Tanh,
    },
    "n_steps": 128,
    "batch_size": 32,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,
    "total_timesteps": 150_000,
    "n_envs": 4,
    "tensorboard_log": "Simulation_4/artifacts/phase10/tensorboard/",
    "verbose": 1,
}


def _resolve_observation_mode(
    observation_mode: str | None,
    nlp_observation: bool,
    text_only_observation: bool,
) -> str:
    mode = (observation_mode or ("nlp" if nlp_observation else "text" if text_only_observation else "state")).lower()
    if mode not in {"state", "text", "nlp"}:
        raise ValueError("observation_mode must be one of: state, text, nlp")
    return mode


def make_env(
    artifacts_root: str,
    reward_shaper: RewardShaper,
    subflow_filter: list[str] | None = None,
    nlg_enabled: bool = False,
    text_only_observation: bool = False,
    nlp_observation: bool = False,
    text_observation_dim: int = 512,
    intent_model: str | None = None,
    agent_model: str | None = None,
):
    def _init():
        use_nlp = bool(nlp_observation)
        env = SupportEnv(
            artifacts_root=artifacts_root,
            nlg_enabled=bool(nlg_enabled or use_nlp or text_only_observation),
            subflow_filter=subflow_filter,
        )
        if use_nlp:
            env = NLPObservationWrapper(
                env,
                intent_model=intent_model,
                agent_response_generator=AgentResponseGenerator(model=agent_model),
            )
        elif text_only_observation:
            env = TextOnlyObservationWrapper(env, n_features=int(text_observation_dim))
        env = RewardShapedWrapper(env, reward_shaper)
        env = ActionMaskedEnv(env)
        env = Monitor(env)
        return env

    return _init


def train_ppo(
    artifacts_root: str,
    timesteps: int = 150_000,
    use_curriculum: bool = True,
    use_reward_shaping: bool = True,
    output_subdir: str = "phase10",
    n_envs: int = 4,
    seed: int = 42,
    nlg_enabled: bool = False,
    text_only_observation: bool = False,
    nlp_observation: bool = False,
    observation_mode: str | None = None,
    text_observation_dim: int = 512,
    intent_model: str | None = None,
    agent_model: str | None = None,
    run_id: str | None = None,
    metrics_eval_freq: int = 500,
    heartbeat_freq_steps: int = 25,
) -> dict[str, Any]:
    artifacts_root_path = Path(artifacts_root)
    phase10_root = artifacts_root_path / output_subdir
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = phase10_root / "runs" / run_id
    save_dir = phase10_root / "models"
    log_dir = phase10_root / "tensorboard"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = _resolve_observation_mode(observation_mode, nlp_observation, text_only_observation)
    use_nlp_observation = mode == "nlp"
    use_text_observation = mode == "text"

    reward_shaper = RewardShaper(enabled=bool(use_reward_shaping), strict_potential=True)
    curriculum = CurriculumScheduler(str(artifacts_root_path)) if use_curriculum else None
    initial_filter = curriculum.get_subflow_filter(0) if curriculum is not None else None

    train_env = make_vec_env(
        make_env(
            str(artifacts_root_path),
            reward_shaper,
            initial_filter,
            nlg_enabled=bool(nlg_enabled),
            text_only_observation=bool(use_text_observation),
            nlp_observation=bool(use_nlp_observation),
            text_observation_dim=int(text_observation_dim),
            intent_model=intent_model,
            agent_model=agent_model,
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
        text_only_observation=bool(use_text_observation),
        nlp_observation=bool(use_nlp_observation),
        text_observation_dim=int(text_observation_dim),
        intent_model=intent_model,
        agent_model=agent_model,
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
        tensorboard_log=str(log_dir),
        verbose=PPO_CONFIG["verbose"],
        seed=seed,
    )

    metrics_callback = TrainingMetricsCallback(eval_freq=int(metrics_eval_freq), verbose=1, run_dir=run_dir)
    heartbeat_callback = ProgressHeartbeatCallback(
        total_timesteps=int(timesteps),
        run_dir=run_dir,
        log_freq_steps=int(heartbeat_freq_steps),
        verbose=1,
    )
    best_model_callback = BestModelCallback(
        save_path=str(save_dir),
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=20,
        baseline_reward=0.99,
        verbose=1,
    )
    callbacks = [heartbeat_callback, metrics_callback, best_model_callback]

    if curriculum is not None:
        callbacks.append(CurriculumCallback(curriculum, train_env, verbose=1))

    start = time.time()
    model.learn(total_timesteps=int(timesteps), callback=callbacks, progress_bar=True)
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
        "nlg_enabled": bool(nlg_enabled or use_nlp_observation or use_text_observation),
        "observation_mode": mode,
        "text_only_observation": bool(use_text_observation),
        "nlp_observation": bool(use_nlp_observation),
        "text_observation_dim": int(text_observation_dim),
        "intent_model": intent_model,
        "agent_model": agent_model,
        "best_eval_reward": float(best_model_callback.best_mean_reward),
        "baseline_beaten": bool(best_model_callback.baseline_beaten),
        "baseline_beaten_step": best_model_callback.baseline_beaten_step,
        "final_resolution_rate": (
            float(metrics_callback.history[-1]["resolution_rate"]) if metrics_callback.history else 0.0
        ),
        "n_envs": int(n_envs),
        "seed": int(seed),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics_eval_freq": int(metrics_eval_freq),
        "heartbeat_freq_steps": int(heartbeat_freq_steps),
    }

    summary_path = phase10_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_callback.save_outputs(summary)

    train_env.close()
    eval_env.close()
    return summary


def continue_ppo_from_checkpoint(
    artifacts_root: str,
    checkpoint_path: str,
    additional_timesteps: int = 150_000,
    use_curriculum: bool = True,
    use_reward_shaping: bool = False,
    output_subdir: str = "phase10_v3_continued",
    n_envs: int = 4,
    seed: int = 42,
    tb_log_name: str = "ppo_v3_continued",
    nlg_enabled: bool = False,
    text_only_observation: bool = False,
    nlp_observation: bool = False,
    observation_mode: str | None = None,
    text_observation_dim: int = 512,
    intent_model: str | None = None,
    agent_model: str | None = None,
    run_id: str | None = None,
    metrics_eval_freq: int = 500,
    heartbeat_freq_steps: int = 25,
) -> dict[str, Any]:
    artifacts_root_path = Path(artifacts_root)
    phase_root = artifacts_root_path / output_subdir
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = phase_root / "runs" / run_id
    save_dir = phase_root / "models"
    log_dir = phase_root / "tensorboard"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = _resolve_observation_mode(observation_mode, nlp_observation, text_only_observation)
    use_nlp_observation = mode == "nlp"
    use_text_observation = mode == "text"

    reward_shaper = RewardShaper(enabled=bool(use_reward_shaping), strict_potential=True)

    # Build env first, then initialize curriculum filter from the loaded model step count.
    train_env = make_vec_env(
        make_env(
            str(artifacts_root_path),
            reward_shaper,
            subflow_filter=None,
            nlg_enabled=bool(nlg_enabled),
            text_only_observation=bool(use_text_observation),
            nlp_observation=bool(use_nlp_observation),
            text_observation_dim=int(text_observation_dim),
            intent_model=intent_model,
            agent_model=agent_model,
        ),
        n_envs=n_envs,
        seed=seed,
    )

    custom_objects = {
        "n_epochs": PPO_CONFIG["n_epochs"],
        "ent_coef": PPO_CONFIG["ent_coef"],
        "vf_coef": PPO_CONFIG["vf_coef"],
        "learning_rate": PPO_CONFIG["learning_rate"],
        "clip_range": PPO_CONFIG["clip_range"],
    }
    model = PPO.load(checkpoint_path, env=train_env, custom_objects=custom_objects)
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
        text_only_observation=bool(use_text_observation),
        nlp_observation=bool(use_nlp_observation),
        text_observation_dim=int(text_observation_dim),
        intent_model=intent_model,
        agent_model=agent_model,
    )()

    metrics_callback = TrainingMetricsCallback(eval_freq=int(metrics_eval_freq), verbose=1, run_dir=run_dir)
    heartbeat_callback = ProgressHeartbeatCallback(
        total_timesteps=int(additional_timesteps),
        run_dir=run_dir,
        log_freq_steps=int(heartbeat_freq_steps),
        verbose=1,
    )
    best_model_callback = BestModelCallback(
        save_path=str(save_dir),
        eval_env=eval_env,
        eval_freq=25000,
        n_eval_episodes=20,
        baseline_reward=0.99,
        verbose=1,
    )
    callbacks = [heartbeat_callback, metrics_callback, best_model_callback]

    if curriculum is not None:
        callbacks.append(CurriculumCallback(curriculum, train_env, verbose=1))

    start = time.time()
    model.learn(
        total_timesteps=int(additional_timesteps),
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name=tb_log_name,
        progress_bar=True,
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
        "nlg_enabled": bool(nlg_enabled or use_nlp_observation or use_text_observation),
        "observation_mode": mode,
        "text_only_observation": bool(use_text_observation),
        "nlp_observation": bool(use_nlp_observation),
        "text_observation_dim": int(text_observation_dim),
        "intent_model": intent_model,
        "agent_model": agent_model,
        "best_eval_reward": float(best_model_callback.best_mean_reward),
        "baseline_beaten": bool(best_model_callback.baseline_beaten),
        "baseline_beaten_step": best_model_callback.baseline_beaten_step,
        "final_resolution_rate": (
            float(metrics_callback.history[-1]["resolution_rate"]) if metrics_callback.history else 0.0
        ),
        "n_envs": int(n_envs),
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics_eval_freq": int(metrics_eval_freq),
        "heartbeat_freq_steps": int(heartbeat_freq_steps),
    }

    summary_path = phase_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_callback.save_outputs(summary)

    train_env.close()
    eval_env.close()
    return summary
