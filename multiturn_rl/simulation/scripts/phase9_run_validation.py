from __future__ import annotations

"""
Usage: python simulation/scripts/phase9_run_validation.py

Runs all Phase 9 validation checks and writes:
- Console output: live PASS/FAIL for each check
- simulation/artifacts/phase9/validation_report.json: machine-readable results
- simulation/artifacts/phase9/validation_summary.txt: human-readable summary

Exit code 0 if all checks pass, 1 if any check fails.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.validation.init import phase9_artifacts_root
from simulation.validation.level1_statistical import run_level1
from simulation.validation.level2_transition import run_level2
from simulation.validation.level3_rl_signal import run_level3
from simulation.validation.level4_rag_coverage import run_level4


INFORMATIONAL_CHECK_KEYS = {"summary_level1_metrics", "policy_metrics"}


def _level_error_result(level_name: str, exc: Exception) -> dict[str, dict[str, Any]]:
    return {
        f"{level_name}_fatal_error": {
            "passed": False,
            "value": f"{type(exc).__name__}: {exc}",
            "threshold": "No exception",
        }
    }


def run_all_validation(
    artifacts_root: str,
    n_episodes_level1: int = 5000,
    n_episodes_level2: int = 2000,
    n_episodes_level3: int = 3000,
) -> dict[str, Any]:
    _ = artifacts_root
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "level1_statistical": {},
        "level2_transition": {},
        "level3_rl_signal": {},
        "level4_rag_coverage": {},
        "overall_pass": False,
        "total_checks": 0,
        "passed_checks": 0,
        "failed_checks": 0,
    }

    print("\n" + "=" * 65)
    print("  PHASE 9 VALIDATION SUITE")
    print("=" * 65)

    print("\n[Level 1] Statistical Realism...")
    try:
        results["level1_statistical"] = run_level1(artifacts_root, n_episodes_level1)
    except Exception as exc:
        results["level1_statistical"] = _level_error_result("level1", exc)

    print("\n[Level 2] Transition Logic Sanity...")
    try:
        results["level2_transition"] = run_level2(artifacts_root, n_episodes_level2)
    except Exception as exc:
        results["level2_transition"] = _level_error_result("level2", exc)

    print("\n[Level 3] RL Signal Quality...")
    try:
        results["level3_rl_signal"] = run_level3(artifacts_root, n_episodes_level3)
    except Exception as exc:
        results["level3_rl_signal"] = _level_error_result("level3", exc)

    print("\n[Level 4] RAG Coverage...")
    try:
        results["level4_rag_coverage"] = run_level4(artifacts_root)
    except Exception as exc:
        results["level4_rag_coverage"] = _level_error_result("level4", exc)

    all_checks: list[bool] = []
    for level_results in [
        results["level1_statistical"],
        results["level2_transition"],
        results["level3_rl_signal"],
        results["level4_rag_coverage"],
    ]:
        for check_name, check_result in level_results.items():
            if check_name in INFORMATIONAL_CHECK_KEYS:
                continue
            all_checks.append(bool(check_result.get("passed", False)))

    results["total_checks"] = len(all_checks)
    results["passed_checks"] = int(sum(all_checks))
    results["failed_checks"] = int(len(all_checks) - sum(all_checks))
    results["overall_pass"] = bool(all(all_checks))

    return results


def print_summary(results: dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("  VALIDATION SUMMARY")
    print("=" * 65)

    for level_name, level_results in [
        ("Level 1 — Statistical Realism", results["level1_statistical"]),
        ("Level 2 — Transition Logic", results["level2_transition"]),
        ("Level 3 — RL Signal Quality", results["level3_rl_signal"]),
        ("Level 4 — RAG Coverage", results["level4_rag_coverage"]),
    ]:
        print(f"\n  {level_name}:")
        for check_name, check_result in level_results.items():
            status = "PASS" if check_result.get("passed") else "FAIL"
            value = check_result.get("value", "")
            threshold = check_result.get("threshold", "")
            print(f"    [{status}] {check_name}")
            if not check_result.get("passed"):
                print(f"           Got: {value}  Expected: {threshold}")

    print(f"\n  Total: {results['passed_checks']}/{results['total_checks']} checks passed")
    overall = "ALL CHECKS PASSED" if results["overall_pass"] else f"{results['failed_checks']} CHECKS FAILED"
    print(f"  Result: {overall}")
    print("=" * 65)


def _make_summary_text(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("PHASE 9 VALIDATION SUMMARY")
    lines.append(f"Timestamp: {results.get('timestamp', '')}")
    lines.append("")
    for level_key in ["level1_statistical", "level2_transition", "level3_rl_signal", "level4_rag_coverage"]:
        lines.append(level_key)
        level_results = results.get(level_key, {})
        for check_name, check_result in level_results.items():
            status = "PASS" if check_result.get("passed") else "FAIL"
            lines.append(f"- [{status}] {check_name}")
        lines.append("")
    lines.append(f"Total checks: {results.get('total_checks', 0)}")
    lines.append(f"Passed checks: {results.get('passed_checks', 0)}")
    lines.append(f"Failed checks: {results.get('failed_checks', 0)}")
    lines.append(f"Overall pass: {results.get('overall_pass', False)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    start = time.time()
    out_dir = phase9_artifacts_root()

    results = run_all_validation(
        artifacts_root=str(out_dir),
        n_episodes_level1=5000,
        n_episodes_level2=2000,
        n_episodes_level3=3000,
    )
    elapsed = time.time() - start
    results["runtime_seconds"] = float(elapsed)

    report_path = out_dir / "validation_report.json"
    summary_path = out_dir / "validation_summary.txt"

    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_path.write_text(_make_summary_text(results), encoding="utf-8")

    print_summary(results)
    print(f"\nWrote report: {report_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Runtime: {elapsed:.2f} seconds")

    return 0 if results.get("overall_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
