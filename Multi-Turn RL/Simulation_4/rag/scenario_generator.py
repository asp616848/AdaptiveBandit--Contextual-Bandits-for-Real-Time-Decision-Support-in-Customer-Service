from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class ScenarioGenerator:
    def __init__(self, templates_path: str, subflow_mapping_path: str):
        with open(templates_path, encoding="utf-8") as f:
            self.templates = json.load(f)
        with open(subflow_mapping_path, encoding="utf-8") as f:
            self.subflow_mapping = json.load(f)

    def get_lumo_label(self, abcd_subflow: str) -> str:
        entry = self.subflow_mapping.get(abcd_subflow, {})
        return entry.get("lumo_label", abcd_subflow)

    def get_display_name(self, abcd_subflow: str) -> str:
        entry = self.subflow_mapping.get(abcd_subflow, {})
        return entry.get("display_name", abcd_subflow.replace("_", " ").title())

    def get_variants(self, abcd_subflow: str) -> list[str]:
        entry = self.subflow_mapping.get(abcd_subflow, {})
        return entry.get("variants", [])

    def sample_scenario(
        self,
        abcd_subflow: str,
        tier: str | None = None,
        rng: random.Random | None = None,
    ) -> dict[str, Any]:
        lumo_label = self.get_lumo_label(abcd_subflow)
        issue_templates = self.templates.get(lumo_label, {})
        variants = issue_templates.get("variants", [])

        if not variants:
            return self._fallback_scenario(abcd_subflow, lumo_label, tier)

        rng = rng or random.Random()
        chosen = rng.choice(variants)

        scenario = {
            "variant_id": chosen["variant_id"],
            "lumo_label": lumo_label,
            "display_name": self.get_display_name(abcd_subflow),
            "description": chosen["description"],
            "identity": chosen.get("identity", {}).copy(),
            "slots": chosen.get("slots", {}).copy(),
            "free_info": chosen.get("free_info", {}).copy(),
            "abcd_subflow": abcd_subflow,
        }

        if tier:
            scenario["slots"]["plan"] = tier

        return scenario

    def get_slot_list(self, scenario: dict[str, Any]) -> list[tuple[str, str]]:
        identity = scenario.get("identity", {})
        slots = scenario.get("slots", {})
        free_info = scenario.get("free_info", {})

        ordered: list[tuple[str, str]] = []

        # Identity slots first - always available, reveal freely.
        identity_priority = [
            "customer_name",
            "customer_email",
            "company_name",
            "phone_last4",
            "card_last4",
            "billing_email",
        ]
        for k in identity_priority:
            if k in identity and identity[k] is not None:
                ordered.append((k, str(identity[k])))

        # Issue-specific slots - controlled by SlotTracker delta_i.
        slot_priority = [
            "workspace_name",
            "workspace_url",
            "plan",
            "member_role",
            "charge_amount",
            "charge_date",
            "seat_count",
            "invoice_number",
        ]
        for k in slot_priority:
            if k in slots and slots[k] is not None:
                ordered.append((k, str(slots[k])))

        # Remaining slot keys not in priority list.
        for k, v in slots.items():
            if k not in slot_priority and k != "issue_detail" and v is not None:
                ordered.append((k, str(v)))

        # issue_detail last in slots - most specific diagnostic info.
        if "issue_detail" in slots and slots["issue_detail"]:
            ordered.append(("issue_detail", str(slots["issue_detail"])))

        # free_info appended last - improvisation material for NLG.
        for k, v in free_info.items():
            if v:
                ordered.append((f"free_info_{k}", str(v)))

        return ordered

    def _fallback_scenario(self, abcd_subflow: str, lumo_label: str, tier: str | None) -> dict[str, Any]:
        return {
            "variant_id": f"{lumo_label}_fallback",
            "lumo_label": lumo_label,
            "display_name": self.get_display_name(abcd_subflow),
            "description": f"Generic {lumo_label} scenario",
            "identity": {
                "customer_name": "Alex Customer",
                "customer_email": "alex@example.com",
                "company_name": "ExampleCo",
                "billing_email": "billing@exampleco.com",
            },
            "slots": {
                "workspace_name": "ExampleCo Workspace",
                "workspace_url": "lumo.com/exampleco",
                "plan": tier or "Pro",
                "issue_detail": f"Having trouble with {lumo_label.replace('_', ' ')}",
            },
            "free_info": {
                "how_discovered": "Issue appeared during normal use",
                "prior_attempts": "Tried basic troubleshooting",
                "additional_context": "Needs quick resolution",
            },
            "abcd_subflow": abcd_subflow,
        }
