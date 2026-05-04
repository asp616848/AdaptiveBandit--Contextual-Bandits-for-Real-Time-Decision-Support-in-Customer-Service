from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.getenv("ROOT_DIR", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation_4.env.local_qwen_client import get_local_qwen_client


OUT_DIR = ROOT / "rl_training_server" / "runs" / "smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(os.getenv("SUPPORT_SIM_LOCAL_MODEL_PATH", ROOT / "Qwen2.5-7B-Instruct-merged"))
BEHAVIOR_RE = re.compile(r"<behavior>\s*(\{.*?\})\s*</behavior>", re.DOTALL)


def parse_behavior(text: str) -> dict[str, Any]:
    match = BEHAVIOR_RE.search(text)
    if not match:
        return {"parse_ok": False, "error": "missing behavior tag"}
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return {"parse_ok": False, "error": str(exc)}
    return {
        "parse_ok": True,
        "info_provided": int(data.get("info_provided", 0)),
        "frustration_signal": int(data.get("frustration_signal", 0)),
        "wants_escalation": int(data.get("wants_escalation", 0)),
    }


PROMPTS = [
    (
        "ask_info",
        "Return strictly:\n"
        "<response>...</response>\n"
        "<behavior>{\"info_provided\":0,\"frustration_signal\":0,\"wants_escalation\":0}</behavior>\n"
        "state={\"subflow\":\"recover_password\",\"turn_index\":2,\"failed_attempts\":0,\"info_collected_ratio\":0.0}\n"
        "agent_action=AskInfo\n"
        "assistant:\n",
    ),
    (
        "failed_solution",
        "Return strictly:\n"
        "<response>...</response>\n"
        "<behavior>{\"info_provided\":0,\"frustration_signal\":0,\"wants_escalation\":0}</behavior>\n"
        "state={\"subflow\":\"billing_dispute\",\"turn_index\":6,\"failed_attempts\":2,\"info_collected_ratio\":0.5}\n"
        "agent_action=ProvideSolution\n"
        "The previous agent solution did not work.\n"
        "assistant:\n",
    ),
    (
        "escalation_pressure",
        "Return strictly:\n"
        "<response>...</response>\n"
        "<behavior>{\"info_provided\":0,\"frustration_signal\":0,\"wants_escalation\":0}</behavior>\n"
        "state={\"subflow\":\"unauthorized_charge\",\"turn_index\":9,\"failed_attempts\":4,\"info_collected_ratio\":0.7}\n"
        "agent_action=Other\n"
        "The customer has repeated the problem several times.\n"
        "assistant:\n",
    ),
]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Local Qwen model folder not found: {MODEL_PATH}")

    os.environ["SUPPORT_SIM_LLM_BACKEND"] = "local"
    os.environ["SUPPORT_SIM_LOCAL_MODEL_PATH"] = str(MODEL_PATH)
    client = get_local_qwen_client(str(MODEL_PATH))

    out_path = OUT_DIR / f"local_qwen_behavior_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    rows = []
    for name, prompt in PROMPTS:
        started = time.time()
        try:
            text = client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an ABCD customer simulator. Return only the requested "
                            "customer response and behavior tags."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=180,
                temperature=0.2,
            )
            parsed = parse_behavior(text)
            ok = bool(parsed.get("parse_ok"))
            error = None
        except Exception as exc:
            text = ""
            parsed = {}
            ok = False
            error = str(exc)
        row = {
            "case": name,
            "ok": ok,
            "error": error,
            "latency_seconds": round(time.time() - started, 3),
            "parsed_behavior": parsed,
            "raw": text,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

