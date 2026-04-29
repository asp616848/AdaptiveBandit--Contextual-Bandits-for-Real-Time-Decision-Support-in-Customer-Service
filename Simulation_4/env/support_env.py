from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from .nlg_layer import NLGLayer
from .reward_engine import RewardEngine
from .slot_tracker import SlotTracker
from .state_engine import StateEngine
try:
    from Simulation_4.rag.lumo_rag import LumoRAG
except Exception:
    LumoRAG = None


class SupportEnv(gym.Env):
    """Phase 7 support simulator environment with optional customer NLG."""

    metadata = {"render_modes": []}

    ACTION_NAMES = {
        0: "AskInfo",
        1: "ProvideSolution",
        2: "AffectiveRepair",
        3: "Escalate",
        4: "Close",
    }

    def __init__(
        self,
        artifacts_root: str,
        nlg_enabled: bool = False,
        subflow_filter: list[str] | None = None,
    ):
        super().__init__()
        self.T_max = 20
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

        self.artifacts_root = Path(artifacts_root).expanduser().resolve()
        self.repo_root = self.artifacts_root.parent.parent if self.artifacts_root.name == "artifacts" else self.artifacts_root.parent

        self.transition_params = self._load_json(
            self._resolve_artifact(
                [
                    "phase 3/transition_calibration.json",
                    "phase3/transition_calibration.json",
                    "abcd/transition_calibration.json",
                ]
            )
        )
        self.psuccess_params = self._load_json(
            self._resolve_artifact(
                [
                    "phase 4/psuccess_model.json",
                    "phase4/psuccess_model.json",
                    "abcd/psuccess_model.json",
                ]
            )
        )
        self.persona_profiles = self._load_json(
            self._resolve_artifact(
                [
                    "phase 5/persona_profiles.json",
                    "phase5/persona_profiles.json",
                    "abcd/persona_profiles.json",
                ]
            )
        )
        self.reward_model = self._load_json(
            self._resolve_artifact(["phase 6/reward_model.json", "phase6/reward_model.json"])
        )
        self.tier_config = self._load_json(
            self._resolve_artifact(["phase 6/tier_config.json", "phase6/tier_config.json"])
        )
        self.subflow_stats = self._load_subflow_stats()

        self.state_engine = StateEngine(self.transition_params, self.psuccess_params)
        reward_payload = self._build_reward_payload(self.reward_model, self.tier_config)
        self.reward_engine = RewardEngine(reward_payload, self.tier_config)
        self.nlg_enabled = bool(nlg_enabled)
        self.nlg_layer = NLGLayer(enabled=self.nlg_enabled)
        rag_dir = self.artifacts_root.parent / "rag"
        if LumoRAG is not None:
            self.lumo_rag = LumoRAG(
                rag_dir=str(rag_dir),
                index_path=str(rag_dir / "index"),
                enabled=True,
            )
        else:
            self.lumo_rag = None

        self.subflow_weights = self._build_subflow_weights(self.subflow_stats, subflow_filter)
        self.subflow_list = list(self.subflow_weights.keys())
        self.subflow_to_idx = {s: i for i, s in enumerate(self.subflow_list)}

        self.subflow_mean_values = {
            str(row["subflow"]): float(row["mean_action_count"])
            for _, row in self.subflow_stats.iterrows()
        }
        self.state_engine.subflow_mean_values = dict(self.subflow_mean_values)
        mean_actions = float(self.subflow_stats["mean_action_count"].mean())
        self.subflow_difficulty = {
            str(row["subflow"]): float(np.clip(float(row["mean_action_count"]) / max(mean_actions, 1e-9), 0.0, 1.0))
            for _, row in self.subflow_stats.iterrows()
        }

        self.member_level_config = self.tier_config.get("member_level_config", {})
        self.business_tier_probabilities = self.tier_config.get("business_tier_probabilities", {})
        self.tier_to_idx = {"Free": 0, "Pro": 1, "Business": 2, "Enterprise": 3}

        self.personas = list(self.persona_profiles.get("personas", []))
        self.persona_lookup = {p["label"]: p for p in self.personas}
        self.pi_persona = self.persona_profiles.get("pi_persona", {})

        self.abcd_data_path = self._resolve_data_file(["abcd_v1.1.json"])
        self.scenario_index = self._build_scenario_index(self.abcd_data_path)

        self.rng: np.random.Generator | None = None
        self.rng_py: random.Random | None = None
        self.state: dict[str, Any] = {}
        self.slot_tracker: SlotTracker | None = None
        self.rag_context: dict[str, Any] = {}
        self.system_prompt: str | None = None
        self.conversation_history: list[dict[str, str]] = []
        self.last_transition_outcome: dict[str, Any] = {}

    def _resolve_artifact(self, candidates: list[str]) -> Path:
        for rel in candidates:
            p = self.artifacts_root / rel
            if p.exists():
                return p
        raise FileNotFoundError(f"Artifact not found in {self.artifacts_root}: {candidates}")

    def _resolve_data_file(self, candidates: list[str]) -> Path:
        roots = [self.repo_root, self.repo_root.parent]
        for root in roots:
            for rel in candidates:
                p = root / rel
                if p.exists():
                    return p
        # Fall back to the first candidate path. _build_scenario_index handles
        # missing files and returns an empty index.
        return roots[0] / candidates[0]

    def _load_subflow_stats(self) -> pd.DataFrame:
        candidates = ["phase 1/extract3_subflow_stats.csv", "phase1/extract3_subflow_stats.csv"]
        try:
            return pd.read_csv(self._resolve_artifact(candidates))
        except FileNotFoundError:
            subflow_offsets = self.psuccess_params.get("subflow_offsets", {})
            subflows = sorted(str(k) for k in subflow_offsets.keys())
            if not subflows:
                subflows = ["generic_support"]

            offsets = np.array([float(subflow_offsets.get(sf, 0.0)) for sf in subflows], dtype=float)
            if len(offsets) <= 1 or float(np.max(offsets) - np.min(offsets)) < 1e-9:
                norm = np.full_like(offsets, 0.5)
            else:
                norm = (offsets - float(np.min(offsets))) / (float(np.max(offsets)) - float(np.min(offsets)))

            mean_action_count = 2.5 + 4.5 * (1.0 - norm)
            resolution_rate = np.clip(0.25 + 0.60 * norm, 0.05, 0.95)
            escalation_rate = np.clip(0.30 - 0.20 * norm, 0.02, 0.40)

            fallback_df = pd.DataFrame(
                {
                    "subflow": subflows,
                    "mean_turns": np.clip(mean_action_count * 3.2, 8.0, 28.0),
                    "mean_action_count": mean_action_count,
                    "resolution_rate": resolution_rate,
                    "escalation_rate": escalation_rate,
                    "conversation_count": np.full(len(subflows), 100, dtype=int),
                }
            )

            out_dir = self.artifacts_root / "phase 1"
            out_dir.mkdir(parents=True, exist_ok=True)
            fallback_df.to_csv(out_dir / "extract3_subflow_stats.csv", index=False)
            return fallback_df

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_reward_payload(self, reward_model: dict[str, Any], tier_config: dict[str, Any]) -> dict[str, Any]:
        payload = dict(reward_model)
        payload.setdefault("parameters", {})
        payload["parameters"]["escalation_costs"] = tier_config.get("escalation_cost_by_tier", {})
        payload["parameters"]["lambda_turn"] = 0.05
        payload["parameters"]["eta_success"] = 8.0
        payload["parameters"]["escalation_penalty"] = 4.0
        payload["parameters"]["dropout_penalty"] = 6.0
        payload["parameters"]["unresolved_close_penalty"] = 3.0
        payload["parameters"]["escalation_bonus_enterprise"] = 0.0
        payload["parameters"]["reward_min"] = -10.0
        payload["parameters"]["reward_max"] = 10.0

        # Phase 6 churn coefficients were calibrated to this selected set.
        payload["churn_model"] = {
            "coefficients": {"c0": -3.8, "cf": 3.0, "cs": 0.1, "ct": 0.08, "ctau": -1.5}
        }
        return payload

    def _build_subflow_weights(self, df: pd.DataFrame, subflow_filter: list[str] | None) -> dict[str, float]:
        work = df.copy()
        if subflow_filter:
            allowed = set(subflow_filter)
            work = work[work["subflow"].isin(allowed)]
            if work.empty:
                raise ValueError("subflow_filter did not match any subflows")

        counts = work[["subflow", "conversation_count"]].copy()
        total = float(counts["conversation_count"].sum())
        return {str(row["subflow"]): float(row["conversation_count"] / total) for _, row in counts.iterrows()}

    def _beta_params_from_mean_std(self, mean: float, std: float) -> tuple[float, float]:
        mean = float(np.clip(mean, 1e-3, 1.0 - 1e-3))
        var = float(max(std * std, 1e-4))
        max_var = mean * (1.0 - mean) - 1e-6
        var = min(var, max(max_var, 1e-5))
        k = mean * (1.0 - mean) / var - 1.0
        alpha = max(0.2, mean * k)
        beta = max(0.2, (1.0 - mean) * k)
        return alpha, beta

    def _sample_persona(self) -> tuple[str, float, float, float]:
        labels = list(self.pi_persona.keys())
        probs = np.array([self.pi_persona[l] for l in labels], dtype=float)
        probs = probs / probs.sum()
        idx = int(self.rng.choice(len(labels), p=probs))
        label = labels[idx]

        profile = self.persona_lookup[label]
        rho_a, rho_b = self._beta_params_from_mean_std(profile["rho_mean"], profile["rho_std"])
        sigma_a, sigma_b = self._beta_params_from_mean_std(profile["sigma_mean"], profile["sigma_std"])
        tau_a, tau_b = self._beta_params_from_mean_std(profile["tau_mean"], profile["tau_std"])

        rho = float(self.rng.beta(rho_a, rho_b))
        sigma = float(self.rng.beta(sigma_a, sigma_b))
        tau = float(self.rng.beta(tau_a, tau_b))
        return label, rho, sigma, tau

    def _sample_tier(self) -> tuple[str, str, int]:
        levels = []
        probs = []
        for level, cfg in self.member_level_config.items():
            if bool(cfg.get("included_in_calibration", False)):
                levels.append(level)
                probs.append(float(cfg["init_probability"]))
        p = np.array(probs, dtype=float)
        p = p / p.sum()
        level_idx = int(self.rng.choice(len(levels), p=p))
        level = levels[level_idx]
        tier = str(self.member_level_config[level]["business_tier"])
        tier_idx = self.tier_to_idx.get(tier, 0)
        return level, tier, tier_idx

    def _build_scenario_index(self, abcd_path: Path) -> dict[str, list[dict[str, Any]]]:
        try:
            with abcd_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return {}

        records: list[dict[str, Any]]
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            for candidate in ["data", "records", "examples", "conversations", "dataset"]:
                if isinstance(payload.get(candidate), list):
                    records = payload[candidate]
                    break
            else:
                records = []
        else:
            records = []

        out: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            subflow = self._extract_subflow(rec)
            scenario = self._extract_scenario(rec)
            if subflow is None or scenario is None:
                continue
            out.setdefault(subflow, []).append(scenario)

        return out

    def _extract_subflow(self, record: dict[str, Any]) -> str | None:
        keys = ["subflow", "sub_flow", "subflow_name", "scenario_subflow", "intent", "flow"]
        for key in keys:
            val = record.get(key)
            if isinstance(val, str) and val:
                return val

        scenario = record.get("scenario")
        if isinstance(scenario, dict):
            for key in keys:
                val = scenario.get(key)
                if isinstance(val, str) and val:
                    return val
        return None

    def _extract_scenario(self, record: dict[str, Any]) -> dict[str, Any] | None:
        scenario = record.get("scenario")
        if isinstance(scenario, dict):
            return scenario

        personal = record.get("personal") if isinstance(record.get("personal"), dict) else {}
        order = record.get("order") if isinstance(record.get("order"), dict) else {}
        product = record.get("product") if isinstance(record.get("product"), dict) else {}

        merged = {}
        if personal:
            merged["personal"] = personal
        if order:
            merged["order"] = order
        if product:
            merged["product"] = product

        return merged if merged else None

    def _sample_scenario(self, subflow: str) -> dict[str, Any]:
        candidates = self.scenario_index.get(subflow, [])
        if candidates:
            idx = int(self.rng.integers(0, len(candidates)))
            return candidates[idx]
        return {
            "personal": {"name": "Customer", "email": "unknown@example.com"},
            "order": {"order_id": "N/A", "subflow": subflow},
            "product": {},
        }

    def _scenario_to_text(self, scenario: dict[str, Any]) -> str:
        parts = []
        for top_key in ["personal", "order", "product"]:
            block = scenario.get(top_key)
            if isinstance(block, dict):
                attrs = ", ".join(f"{k}={v}" for k, v in block.items() if v is not None)
                if attrs:
                    parts.append(f"{top_key}: {attrs}")
        return "\n".join(parts) if parts else "No additional scenario context provided."

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None or self.rng is None:
            self.rng = np.random.default_rng(seed)
        if seed is not None or self.rng_py is None:
            self.rng_py = random.Random(seed)

        subflows = list(self.subflow_weights.keys())
        probs = np.array([self.subflow_weights[s] for s in subflows], dtype=float)
        probs = probs / probs.sum()
        subflow_idx = int(self.rng.choice(len(subflows), p=probs))
        subflow = subflows[subflow_idx]

        member_level, tier, tier_idx = self._sample_tier()
        persona_label, rho, sigma, tau = self._sample_persona()

        value_weight = float(self.rng.beta(2.0, 2.0))
        information = float(self.rng.beta(1.2, 6.0))
        frustration = float(self.rng.beta(1.5, 8.0))

        progress_noise = float(self.rng.normal(0.0, 0.03))
        progress = float(np.clip(0.3 * information + progress_noise, 0.0, 1.0))

        self.state = {
            "subflow": subflow,
            "subflow_idx": subflow_idx,
            "tier": tier,
            "tier_idx": tier_idx,
            "member_level": member_level,
            "difficulty": float(self.subflow_difficulty.get(subflow, 0.5)),
            "value_weight": value_weight,
            "persona_label": persona_label,
            "rho": rho,
            "sigma": sigma,
            "tau": tau,
            "information": information,
            "progress": progress,
            "frustration": frustration,
            "failed_streak": 0,
            "turn_count": 0,
            "resolved": 0,
            "escalated": 0,
            "dropped_off": 0,
            "done": False,
        }

        self.last_transition_outcome = {}
        self.conversation_history = []
        self.slot_tracker = None
        self.rag_context = {}
        self.system_prompt = None

        if self.lumo_rag is not None and self.lumo_rag.enabled:
            rag_context = self.lumo_rag.get_episode_context(
                abcd_subflow=subflow,
                tier=tier,
                rng=self.rng_py,
            )
            self.slot_tracker = SlotTracker(
                slots=rag_context["slot_list"],
                subflow=subflow,
            )
            self.rag_context = rag_context
            self.state["lumo_label"] = rag_context["lumo_label"]
            self.state["display_name"] = rag_context["display_name"]

        if self.nlg_enabled:
            scenario_context = self._scenario_to_text(self.rag_context.get("scenario", {}))
            policy_context = self.rag_context.get("policy_context", "")
            display_name = self.rag_context.get("display_name", subflow.replace("_", " ").title())
            self.system_prompt = self.nlg_layer.build_system_prompt(
                persona_label=persona_label,
                rho=rho,
                sigma=sigma,
                tau=tau,
                scenario_context=scenario_context,
                subflow=subflow,
                policy_context=policy_context,
                display_name=display_name,
                identity=self.rag_context.get("scenario", {}).get("identity", {}),
                free_info=self.rag_context.get("scenario", {}).get("free_info", {}),
            )

        return self._get_obs(), self._get_info()

    def _dispatch_transition(self, action_name: str) -> dict[str, Any]:
        if action_name == "AskInfo":
            self.state, outcome = self.state_engine.transition_ask_info(self.state, self.rng)
            slots_to_reveal = int(outcome.get("slots_to_reveal", 0) or 0)
            if slots_to_reveal > 0 and self.slot_tracker is not None:
                revealed = self.slot_tracker.reveal_next(n=slots_to_reveal)
                outcome["slots_revealed"] = revealed
                outcome["revealed_slots"] = revealed
            else:
                outcome["slots_revealed"] = []
                outcome["revealed_slots"] = []
                if self.slot_tracker is not None:
                    self.slot_tracker.clear_last_revealed()
            return outcome

        if action_name == "ProvideSolution":
            self.state, outcome = self.state_engine.transition_provide_solution(self.state, self.rng)
            return outcome

        if action_name == "AffectiveRepair":
            self.state, outcome = self.state_engine.transition_affective_repair(self.state, self.rng)
            return outcome

        if action_name == "Escalate":
            self.state, outcome = self.state_engine.transition_escalate(self.state)
            return outcome

        if action_name == "Close":
            self.state, outcome = self.state_engine.transition_close(self.state, self.rng)
            return outcome

        raise ValueError(f"Unsupported action_name={action_name}")

    def step(self, action: int, agent_text: str | None = None):
        assert action in range(5), f"Invalid action {action}"
        action_name = self.ACTION_NAMES[action]

        reward = self.reward_engine.per_turn_reward()
        transition_outcome = self._dispatch_transition(action_name)
        if action_name == "Close" and not bool(transition_outcome.get("resolved", False)):
            transition_outcome["terminal_type"] = "unresolved_close"
            transition_outcome["outcome"] = "unresolved_close"

        if self.nlg_enabled:
            agent_message = (agent_text or "").strip() or f"Agent action: {action_name}"
            self.conversation_history.append({"role": "user", "content": agent_message})
            turn_prompt = self.nlg_layer.build_turn_prompt(
                action_type=action_name,
                agent_text=agent_message,
                transition_outcome=transition_outcome,
                frustration=float(self.state["frustration"]),
                slot_tracker=self.slot_tracker,
                scenario_slots=self.rag_context.get("scenario", {}).get("slots", {}) if self.rag_context else {},
                conversation_history=self.conversation_history,
            )
            utterance = self.nlg_layer.generate_utterance(
                system_prompt=self.system_prompt or "",
                conversation_history=self.conversation_history,
                turn_prompt=turn_prompt,
            )
            self.conversation_history.append({"role": "assistant", "content": utterance})
            transition_outcome["customer_utterance"] = utterance

        if not bool(self.state["done"]):
            p_dropout = self.state_engine.compute_p_dropout(self.state)
            if self.rng.random() < p_dropout:
                self.state, drop_outcome = self.state_engine.transition_autonomous_dropout(self.state)
                transition_outcome.update(drop_outcome)

        self.state, timeout_outcome = self.state_engine.advance_turn_and_apply_timeout(self.state, self.T_max)
        transition_outcome.update(timeout_outcome)

        terminal_reward = 0.0
        if bool(self.state["done"]):
            terminal_type = transition_outcome.get("terminal_type", "timeout")
            terminal_reward = self.reward_engine.terminal_reward(
                outcome=terminal_type,
                state=self.state,
                tier=str(self.state["tier"]),
                value_weight=float(self.state["value_weight"]),
            )
            reward += terminal_reward

        self.last_transition_outcome = dict(transition_outcome)
        self.last_transition_outcome["per_turn_reward"] = self.reward_engine.per_turn_reward()
        self.last_transition_outcome["terminal_reward"] = terminal_reward

        return self._get_obs(), reward, bool(self.state["done"]), False, self._get_info()

    def _get_obs(self) -> np.ndarray:
        subflow_id_norm = float(self.state["subflow_idx"]) / max(len(self.subflow_list) - 1, 1)
        tier_id_norm = float(self.state["tier_idx"]) / 3.0
        return np.array(
            [
                subflow_id_norm,
                tier_id_norm,
                float(np.clip(self.state["difficulty"], 0.0, 1.0)),
                float(np.clip(self.state["information"], 0.0, 1.0)),
                float(np.clip(self.state["progress"], 0.0, 1.0)),
                float(np.clip(self.state["frustration"], 0.0, 1.0)),
                float(np.clip(self.state["failed_streak"] / 5.0, 0.0, 1.0)),
                float(np.clip(self.state["turn_count"] / self.T_max, 0.0, 1.0)),
                float(self.state["resolved"]),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        info = {
            "subflow": self.state.get("subflow"),
            "tier": self.state.get("tier"),
            "persona": self.state.get("persona_label"),
            "rho": self.state.get("rho"),
            "sigma": self.state.get("sigma"),
            "tau": self.state.get("tau"),
            "p_success": self.state_engine.compute_p_success(self.state),
            "p_dropout": self.state_engine.compute_p_dropout(self.state),
            "p_churn": self.reward_engine.compute_p_churn(
                float(self.state.get("frustration", 0.0)),
                int(self.state.get("failed_streak", 0)),
                int(self.state.get("turn_count", 0)),
                float(self.state.get("tau", 0.5)),
            ),
            "conversation_history": self.conversation_history if self.nlg_enabled else [],
            "last_transition_outcome": self.last_transition_outcome,
            "transition_outcome": self.last_transition_outcome,
        }
        if self.slot_tracker is not None:
            info["revealed_context"] = self.slot_tracker.get_revealed_context()
            info["information_proxy_slots"] = self.slot_tracker.information_proxy(
                self.subflow_mean_values.get(str(self.state["subflow"]), 1.0)
            )
        return info

