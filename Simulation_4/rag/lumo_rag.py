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
            top_k=3,
            doc_type_filter="support_policies",
        )
        product_chunks = self.retriever.retrieve(
            query=query,
            top_k=2,
            doc_type_filter="product_overview",
        )
        playbook_chunks = self.retriever.retrieve(
            query=query,
            top_k=2,
            doc_type_filter="playbooks",
        )
        if not playbook_chunks:
            # Fallback for older indexes where playbook retrieval works better with subflow phrasing.
            playbook_chunks = self.retrieve_agent_playbook(abcd_subflow, top_k=2)

        llm_chunks = self._merge_and_rank_chunks(policy_chunks, product_chunks, playbook_chunks)

        slot_list = self.scenario_generator.get_slot_list(scenario)

        return {
            "scenario": scenario,
            "slot_list": slot_list,
            "policy_chunks": policy_chunks,
            "product_chunks": product_chunks,
            "playbook_chunks": playbook_chunks,
            "llm_chunks": llm_chunks,
            "lumo_label": scenario["lumo_label"],
            "display_name": scenario["display_name"],
            "variant_descriptions": self.scenario_generator.get_variants(abcd_subflow),
            "policy_context": self._format_policy_context(llm_chunks),
        }

    def _merge_and_rank_chunks(self, *chunk_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_total = 7
        by_id: dict[str, dict[str, Any]] = {}
        top_per_source: list[dict[str, Any]] = []

        for chunk_list in chunk_lists:
            if chunk_list:
                top_per_source.append(dict(chunk_list[0]))
            for chunk in chunk_list:
                chunk_id = str(chunk.get("chunk_id", ""))
                if not chunk_id:
                    continue
                score = float(chunk.get("similarity_score", 0.0))
                prev = by_id.get(chunk_id)
                if prev is None or score > float(prev.get("similarity_score", 0.0)):
                    by_id[chunk_id] = dict(chunk)

        selected: dict[str, dict[str, Any]] = {}

        # Ensure context diversity: keep at least one strongest chunk per source list when available.
        for chunk in top_per_source:
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id and chunk_id not in selected:
                selected[chunk_id] = chunk
            if len(selected) >= max_total:
                break

        remaining = sorted(by_id.values(), key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)
        for chunk in remaining:
            if len(selected) >= max_total:
                break
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id and chunk_id not in selected:
                selected[chunk_id] = chunk

        out = list(selected.values())
        out.sort(key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)
        return out

    def _format_policy_context(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return ""
        doc_type_title = {
            "support_policies": "Support Policies",
            "product_overview": "Company/Product Overview",
            "playbooks": "Agent Playbook",
        }
        parts: list[str] = []
        for chunk in chunks:
            section_title = str(chunk.get("section_title", "Untitled"))
            doc_type = str(chunk.get("doc_type", "other"))
            source_title = doc_type_title.get(doc_type, doc_type.replace("_", " ").title())
            parts.append(f"[{source_title}] {section_title}\n{str(chunk.get('content', ''))[:800]}")
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
