from __future__ import annotations

from typing import Any


class SlotTracker:
    """Tracks revealed/unrevealed concrete slots for one scenario episode."""

    _PREFERRED_ORDER = [
        "name",
        "order_id",
        "email",
        "phone",
        "product_type",
        "product_name",
        "dollar_amount",
        "amount",
        "address",
        "zip",
        "tracking_id",
    ]

    def __init__(self, slots: list[tuple[str, str]], subflow: str):
        self.subflow = subflow
        self.scenario = {"slots": {k: v for k, v in slots}}
        self.all_slots = self._order_slots([(str(k), str(v)) for k, v in slots])
        self.revealed_slots: list[tuple[str, Any]] = []
        self.unrevealed_slots: list[tuple[str, Any]] = list(self.all_slots)
        self.last_revealed: list[tuple[str, Any]] = []

    def _flatten_slots(self, payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
        slots: list[tuple[str, Any]] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_str = str(key)
                new_prefix = f"{prefix}.{key_str}" if prefix else key_str
                slots.extend(self._flatten_slots(value, new_prefix))
        elif isinstance(payload, list):
            for idx, item in enumerate(payload):
                slots.extend(self._flatten_slots(item, f"{prefix}[{idx}]"))
        else:
            if payload is None:
                return slots
            value = str(payload).strip()
            if value == "" or value.lower() == "null":
                return slots
            slots.append((prefix, payload))
        return slots

    def _order_slots(self, slots: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        if not slots:
            return []

        rank = {name: i for i, name in enumerate(self._PREFERRED_ORDER)}

        def slot_key(item: tuple[str, Any]) -> tuple[int, str]:
            slot_name = item[0]
            leaf = slot_name.split(".")[-1].split("[")[0]
            return (rank.get(leaf, len(rank) + 1), slot_name)

        return sorted(slots, key=slot_key)

    def reveal_next(self, n: int = 1) -> list[tuple[str, Any]]:
        if n <= 0 or not self.unrevealed_slots:
            self.last_revealed = []
            return []
        count = min(n, len(self.unrevealed_slots))
        newly = self.unrevealed_slots[:count]
        self.unrevealed_slots = self.unrevealed_slots[count:]
        self.revealed_slots.extend(newly)
        self.last_revealed = list(newly)
        return newly

    def get_last_revealed(self) -> list[tuple[str, Any]]:
        return list(getattr(self, "last_revealed", []))

    def clear_last_revealed(self):
        self.last_revealed = []

    def get_revealed_context(self) -> str:
        if not self.revealed_slots:
            return "No specific details provided yet."

        parts: list[str] = []
        for name, value in self.revealed_slots:
            value_str = str(value).strip()
            lowered = value_str.lower()
            if lowered in ("n/a", "none", "null", "unknown", ""):
                continue
            if value_str.startswith("{") or "=" in value_str:
                continue
            parts.append(f"{name}: {value}")

        if not parts:
            return "No specific details provided yet."

        return "Customer has provided: " + ", ".join(parts)

    def get_next_to_reveal(self) -> tuple[str, Any] | None:
        if not self.unrevealed_slots:
            return None
        return self.unrevealed_slots[0]

    def information_proxy(self, subflow_mean_total_values: float) -> float:
        denom = max(float(subflow_mean_total_values), 1.0)
        ratio = len(self.revealed_slots) / denom
        return float(max(0.0, min(1.0, ratio)))
