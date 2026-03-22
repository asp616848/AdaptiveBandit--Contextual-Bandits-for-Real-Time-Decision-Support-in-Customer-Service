import json
from typing import Any, Dict


def safe_json_loads(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    if not text:
        return default
    payload = text.strip()
    if payload.startswith("```"):
        parts = payload.split("```")
        if len(parts) >= 2:
            payload = parts[1].replace("json", "", 1).strip()
    try:
        value = json.loads(payload)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    return default
