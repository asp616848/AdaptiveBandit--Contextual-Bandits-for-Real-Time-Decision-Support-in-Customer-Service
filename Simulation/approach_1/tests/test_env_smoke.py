from __future__ import annotations

from src.env.simple_support_env import SimpleSupportEnv, TaskDefinition


def _build_env() -> SimpleSupportEnv:
    tasks = [
        TaskDefinition(
            subflow="recover_username",
            canonical_actions=["pull_up_account", "verify_identity", "send_link"],
            support_count=10,
            empirical_success_rate=0.8,
        )
    ]
    reward_cfg = {
        "correct_step_reward": 1.0,
        "wrong_action_penalty": -1.0,
        "completion_bonus": 5.0,
        "ask_info_penalty": -0.1,
        "escalate_penalty": -2.0,
        "close_early_penalty": -3.0,
        "timeout_penalty": -5.0,
        "timeout_slack": 2,
    }
    return SimpleSupportEnv(tasks=tasks, reward_config=reward_cfg, seed=7)


def test_correct_path_completes_episode() -> None:
    env = _build_env()
    obs, info = env.reset(seed=7)

    seq = info["required_actions"]
    action_map = info["action_mapping"]

    done = False
    total_reward = 0.0
    for a in seq:
        obs, reward, done, _, step_info = env.step(action_map[a])
        total_reward += reward

    assert done is True
    assert step_info["terminal_reason"] == "completed"
    assert total_reward > 0.0


def test_escalate_ends_episode() -> None:
    env = _build_env()
    env.reset(seed=11)
    obs, reward, done, _, step_info = env.step(env.ESCALATE)

    assert done is True
    assert step_info["terminal_reason"] == "escalate"
    assert reward < 0.0
