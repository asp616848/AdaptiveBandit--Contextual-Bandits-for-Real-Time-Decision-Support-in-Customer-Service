from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from simulation.rag.lumo_rag import LumoRAG
from simulation.validation.init import project_root, safe_check


RETRIEVAL_SPOT_CHECKS = [
    ("reset_2fa", "Security"),
    ("manage_dispute_bill", "Billing"),
    ("slow_speed", "Known Issues"),
    ("refund_initiate", "Billing"),
    ("recover_password", "Security"),
]


def run_level4(artifacts_root: str) -> dict[str, dict[str, Any]]:
    _ = artifacts_root
    root = project_root()
    rag_dir = root / "simulation" / "rag"

    rag = LumoRAG(
        rag_dir=str(rag_dir),
        index_path=str(rag_dir / "index"),
        enabled=True,
    )

    mapping = json.loads((rag_dir / "subflow_mapping.json").read_text(encoding="utf-8"))
    templates = json.loads((rag_dir / "scenario_templates.json").read_text(encoding="utf-8"))

    results: dict[str, dict[str, Any]] = {}

    def check_subflow_coverage() -> dict[str, Any]:
        coverage_rows: list[dict[str, Any]] = []
        full_coverage = 0
        fallback_count = 0

        for sf in sorted(mapping.keys()):
            ctx = rag.get_episode_context(sf, tier="Business", rng=random.Random(4242))
            scenario = ctx.get("scenario", {}) if isinstance(ctx, dict) else {}
            variant_id = str(scenario.get("variant_id", ""))
            used_fallback = variant_id.endswith("_fallback")
            has_mapping = str(ctx.get("lumo_label", "")) != sf
            has_slots = bool(ctx.get("slot_list"))
            has_policy = bool(ctx.get("policy_chunks"))
            ok = has_mapping and has_slots and has_policy and (not used_fallback)
            if ok:
                full_coverage += 1
            if used_fallback:
                fallback_count += 1
            coverage_rows.append(
                {
                    "subflow": sf,
                    "lumo_label": ctx.get("lumo_label", ""),
                    "variant_id": variant_id,
                    "has_mapping": has_mapping,
                    "slot_list_non_empty": has_slots,
                    "policy_chunks_non_empty": has_policy,
                    "used_fallback": used_fallback,
                    "passed": ok,
                }
            )

        passed = all(r["passed"] for r in coverage_rows)
        return {
            "passed": passed,
            "value": {
                "n_subflows": len(coverage_rows),
                "full_template_coverage_count": full_coverage,
                "fallback_count": fallback_count,
                "coverage_table": coverage_rows,
            },
            "threshold": "all subflows mapped, non-empty slot_list and policy_chunks, no fallback variant",
        }

    def check_identity_slot_completeness() -> dict[str, Any]:
        missing: list[dict[str, str]] = []
        total_variants = 0
        for lumo_label, block in templates.items():
            variants = block.get("variants", []) if isinstance(block, dict) else []
            for var in variants:
                total_variants += 1
                vid = str(var.get("variant_id", "unknown"))
                identity = var.get("identity", {}) if isinstance(var.get("identity", {}), dict) else {}
                slots = var.get("slots", {}) if isinstance(var.get("slots", {}), dict) else {}
                free_info = var.get("free_info", {}) if isinstance(var.get("free_info", {}), dict) else {}

                if not identity or not identity.get("customer_name") or not identity.get("billing_email"):
                    missing.append({"variant_id": vid, "field": "identity.customer_name/billing_email"})
                if not slots or not slots.get("issue_detail"):
                    missing.append({"variant_id": vid, "field": "slots.issue_detail"})
                if not free_info or not free_info.get("how_discovered"):
                    missing.append({"variant_id": vid, "field": "free_info.how_discovered"})

        return {
            "passed": len(missing) == 0,
            "value": {"total_variants": total_variants, "missing_fields": missing},
            "threshold": "every variant has identity(customer_name,billing_email), slots(issue_detail), free_info(how_discovered)",
        }

    def check_scenario_grounding_quality() -> dict[str, Any]:
        subflows = sorted(mapping.keys())
        rng = random.Random(9911)
        issues: list[dict[str, Any]] = []

        for i in range(50):
            sf = subflows[i % len(subflows)]
            ctx = rag.get_episode_context(sf, tier="Business", rng=rng)
            scenario = ctx.get("scenario", {})
            slots = scenario.get("slots", {}) if isinstance(scenario.get("slots", {}), dict) else {}
            identity = scenario.get("identity", {}) if isinstance(scenario.get("identity", {}), dict) else {}
            issue_detail = str(slots.get("issue_detail", ""))
            customer_name = str(identity.get("customer_name", ""))

            if len(issue_detail) <= 50:
                issues.append({"subflow": sf, "reason": "issue_detail_too_short", "value": issue_detail})
            if "Customer reports" in issue_detail:
                issues.append({"subflow": sf, "reason": "placeholder_customer_reports", "value": issue_detail})
            if "$X" in issue_detail or "[amount]" in issue_detail:
                issues.append({"subflow": sf, "reason": "placeholder_amount", "value": issue_detail})
            if customer_name in {"Customer", "Alex Customer"}:
                issues.append({"subflow": sf, "reason": "fallback_customer_name", "value": customer_name})

        return {
            "passed": len(issues) == 0,
            "value": {"sample_size": 50, "issues": issues},
            "threshold": "issue_detail > 50 chars, no placeholders, no fallback names",
        }

    def check_doc_type_isolation() -> dict[str, Any]:
        subflows = sorted(mapping.keys())
        sample = subflows[:20]

        policy_has_playbooks = []
        playbook_wrong_type = []
        for sf in sample:
            ctx = rag.get_episode_context(sf, tier="Business", rng=random.Random(1234))
            for ch in ctx.get("policy_chunks", []):
                if ch.get("doc_type") == "playbooks":
                    policy_has_playbooks.append({"subflow": sf, "section": ch.get("section_title", "")})

            playbook_chunks = rag.retrieve_agent_playbook(sf, top_k=3)
            for ch in playbook_chunks:
                if ch.get("doc_type") != "playbooks":
                    playbook_wrong_type.append({"subflow": sf, "doc_type": ch.get("doc_type")})

        return {
            "passed": (len(policy_has_playbooks) == 0) and (len(playbook_wrong_type) == 0),
            "value": {
                "policy_chunks_with_playbooks": policy_has_playbooks,
                "non_playbook_in_playbook_retrieval": playbook_wrong_type,
            },
            "threshold": "policy_chunks exclude playbooks; retrieve_agent_playbook returns only playbooks",
        }

    def check_retrieval_spot_checks() -> dict[str, Any]:
        rows = []
        for sf, expected in RETRIEVAL_SPOT_CHECKS:
            ctx = rag.get_episode_context(sf, tier="Business", rng=random.Random(37))
            chunks = ctx.get("policy_chunks", [])
            top_title = str(chunks[0].get("section_title", "")) if chunks else ""
            ok = expected.lower() in top_title.lower()
            rows.append({
                "subflow": sf,
                "expected_keyword": expected,
                "top_section_title": top_title,
                "passed": ok,
            })

        return {
            "passed": all(r["passed"] for r in rows),
            "value": {"spot_checks": rows},
            "threshold": "top section title contains expected keyword",
        }

    results["check1_subflow_coverage"] = safe_check("check1_subflow_coverage", check_subflow_coverage)
    results["check2_identity_slot_completeness"] = safe_check("check2_identity_slot_completeness", check_identity_slot_completeness)
    results["check3_scenario_grounding_quality"] = safe_check("check3_scenario_grounding_quality", check_scenario_grounding_quality)
    results["check4_doc_type_isolation"] = safe_check("check4_doc_type_isolation", check_doc_type_isolation)
    results["check5_retrieval_spot_checks"] = safe_check("check5_retrieval_spot_checks", check_retrieval_spot_checks)

    return results
