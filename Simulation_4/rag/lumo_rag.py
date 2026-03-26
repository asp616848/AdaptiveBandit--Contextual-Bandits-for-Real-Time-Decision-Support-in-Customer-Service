from __future__ import annotations

from pathlib import Path
from typing import Any

from .document_store import DocumentStore
from .retriever import Retriever
from .scenario_generator import ScenarioGenerator


class LumoRAG:
    def __init__(
        self,
        rag_dir: str,
        index_path: str | None = None,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        if not self.enabled:
            return

        self.rag_dir = Path(rag_dir).expanduser().resolve()
        documents_dir = self.rag_dir / "documents"
        templates_path = self.rag_dir / "scenario_templates.json"
        subflow_mapping_path = self.rag_dir / "subflow_mapping.json"

        self.document_store = DocumentStore(str(documents_dir))
        self.retriever = Retriever(
            document_store=self.document_store,
            index_path=index_path,
        )
        self.scenario_generator = ScenarioGenerator(
            templates_path=str(templates_path),
            subflow_mapping_path=str(subflow_mapping_path),
        )

    def get_episode_context(self, abcd_subflow: str, tier: str, rng=None) -> dict[str, Any]:
        if not self.enabled:
            return self._disabled_context(abcd_subflow, tier)

        scenario = self.scenario_generator.sample_scenario(
            abcd_subflow=abcd_subflow,
            tier=tier,
            rng=rng,
        )

        query_expansion = self.scenario_generator.subflow_mapping.get(
            abcd_subflow, {}
        ).get("query_expansion", "")

        query = f"{scenario['display_name']} {scenario['description']} {query_expansion}"
        policy_chunks = self.retriever.retrieve(
            query=query,
            top_k=2,
            doc_type_filter="support_policies",
        )

        slot_list = self.scenario_generator.get_slot_list(scenario)

        return {
            "scenario": scenario,
            "slot_list": slot_list,
            "policy_chunks": policy_chunks,
            "lumo_label": scenario["lumo_label"],
            "display_name": scenario["display_name"],
            "variant_descriptions": self.scenario_generator.get_variants(abcd_subflow),
            "policy_context": self._format_policy_context(policy_chunks),
        }

    def _format_policy_context(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return ""
        parts: list[str] = []
        for chunk in chunks:
            parts.append(f"[{chunk['section_title']}]\n{chunk['content'][:800]}")
        return "\n\n".join(parts)

    def _disabled_context(self, abcd_subflow: str, tier: str) -> dict[str, Any]:
        return {
            "scenario": {"slots": {}, "description": abcd_subflow},
            "slot_list": [],
            "policy_chunks": [],
            "lumo_label": abcd_subflow,
            "display_name": abcd_subflow.replace("_", " ").title(),
            "variant_descriptions": [],
            "policy_context": "",
        }

    def retrieve_agent_playbook(self, abcd_subflow: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        entry = self.scenario_generator.subflow_mapping.get(abcd_subflow, {})
        query_expansion = entry.get("query_expansion", "")
        display_name = entry.get("display_name", abcd_subflow.replace("_", " "))
        query = f"{display_name} {query_expansion}".strip()
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            doc_type_filter="playbooks",
        )
