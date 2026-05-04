from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulation_4.rag.lumo_rag import LumoRAG


def _print_retrieval_tests(rag: LumoRAG) -> None:
    tests = [
        "refund outside 30 days courtesy credit",
        "2fa lost phone cannot login",
        "cancel subscription and export data",
        "charged twice duplicate billing",
    ]
    print("[Retrieval] Top policy sections")
    for q in tests:
        rows = rag.retriever.retrieve(q, top_k=2, doc_type_filter="support_policies")
        titles = [r.get("section_title", "?") for r in rows]
        print(f"  query={q!r} -> {titles}")


def _smoke_tests(rag: LumoRAG) -> list[tuple[str, bool, str]]:
    rng = random.Random(7)
    ctx = rag.get_episode_context("manage_dispute_bill", tier="Business", rng=rng)

    checks: list[tuple[str, bool, str]] = []
    checks.append(("context_has_required_keys", all(k in ctx for k in ["scenario", "slot_list", "policy_chunks", "lumo_label", "display_name", "policy_context"]), "missing required context keys"))
    checks.append(("slot_list_non_empty", len(ctx.get("slot_list", [])) > 0, "slot_list is empty"))
    checks.append(("policy_chunks_non_empty", len(ctx.get("policy_chunks", [])) > 0, "policy_chunks is empty"))
    checks.append(("label_maps_to_billing_dispute", ctx.get("lumo_label") == "billing_dispute", f"unexpected label: {ctx.get('lumo_label')}"))
    checks.append(("display_name_present", bool(ctx.get("display_name")), "display_name missing"))
    checks.append(("policy_context_non_empty", bool(str(ctx.get("policy_context", "")).strip()), "policy_context empty"))
    return checks


def main() -> None:
    rag_dir = PROJECT_ROOT / "Simulation_4" / "rag"
    index_dir = rag_dir / "index"

    rag = LumoRAG(rag_dir=str(rag_dir), index_path=str(index_dir), enabled=True)

    _print_retrieval_tests(rag)

    checks = _smoke_tests(rag)
    print("[Smoke] Results")
    for name, ok, detail in checks:
        if ok:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name} ({detail})")

    ctx = rag.get_episode_context("manage_dispute_bill", tier="Business", rng=random.Random(7))
    print("[SampleContext] manage_dispute_bill")
    print(json.dumps(ctx, indent=2))


if __name__ == "__main__":
    main()
