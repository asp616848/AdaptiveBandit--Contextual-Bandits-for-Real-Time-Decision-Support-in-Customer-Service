from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.getenv("ROOT_DIR", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation_4.env.agent_response_generator import AgentResponseGenerator
from Simulation_4.env.support_env import SupportEnv
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.training.nlp_observation import NLPObservationWrapper
from Simulation_4.training.reward_shaping import RewardShapedWrapper, RewardShaper


OUT_DIR = ROOT / "rl_training_server" / "runs" / "smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("SUPPORT_SIM_LLM_BACKEND", "local")
os.environ.setdefault("SUPPORT_SIM_LOCAL_MODEL_PATH", str(ROOT / "Qwen2.5-7B-Instruct-merged"))
os.environ.setdefault("SUPPORT_SIM_LLM_MODEL", "local-qwen")
os.environ.setdefault("SUPPORT_SIM_AGENT_MODEL", os.environ["SUPPORT_SIM_LLM_MODEL"])
os.environ.setdefault("SUPPORT_SIM_INTENT_MODEL", os.environ["SUPPORT_SIM_LLM_MODEL"])


def main() -> None:
    env = SupportEnv(str(ROOT / "Simulation_4" / "artifacts"), nlg_enabled=True)
    env = NLPObservationWrapper(
        env,
        intent_model=os.environ["SUPPORT_SIM_INTENT_MODEL"],
        agent_response_generator=AgentResponseGenerator(model=os.environ["SUPPORT_SIM_AGENT_MODEL"]),
    )
    env = RewardShapedWrapper(env, RewardShaper(enabled=True, strict_potential=True))
    env = ActionMaskedEnv(env)

    obs, info = env.reset(seed=123)
    steps = []
    done = False
    total_reward = 0.0

    for _ in range(6):
        if done:
            break
        mask = env.action_masks().tolist()
        action = 0 if mask[0] else next(i for i, ok in enumerate(mask) if ok)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        transition = dict(info.get("last_transition_outcome", {}) or {})
        steps.append(
            {
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "terminal_type": transition.get("terminal_type"),
                "customer_utterance": transition.get("customer_utterance"),
                "obs_shape": list(getattr(obs, "shape", [])),
                "mask": mask,
            }
        )

    report = {
        "ok": True,
        "backend": os.environ["SUPPORT_SIM_LLM_BACKEND"],
        "model_path": os.environ["SUPPORT_SIM_LOCAL_MODEL_PATH"],
        "customer_model": os.environ["SUPPORT_SIM_LLM_MODEL"],
        "intent_model": os.environ["SUPPORT_SIM_INTENT_MODEL"],
        "agent_model": os.environ["SUPPORT_SIM_AGENT_MODEL"],
        "initial_subflow": info.get("subflow"),
        "total_reward": total_reward,
        "steps": steps,
    }
    out_path = OUT_DIR / f"simulator_nlp_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")
    env.close()


if __name__ == "__main__":
    main()
