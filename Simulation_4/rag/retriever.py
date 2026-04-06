from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from .document_store import DocumentStore


class Retriever:
    def __init__(
        self,
        document_store: DocumentStore,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str | None = None,
    ):
        self.document_store = document_store
        self.model_name = model_name
        self.index: Any | None = None
        self._embeddings: np.ndarray | None = None
        self.chunks: list[dict[str, Any]] = []
        self.model = SentenceTransformer(self.model_name)

        if index_path and (Path(index_path) / "faiss.index").exists():
            self.load_index(index_path)
        else:
            self._build_index()

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return embeddings / norms

    def _build_index(self) -> None:
        self.chunks = self.document_store.get_all_chunks()
        corpus = [c["content"] for c in self.chunks]
        if not corpus:
            if faiss is not None:
                self.index = faiss.IndexFlatIP(384)
            else:
                self.index = None
                self._embeddings = np.zeros((0, 384), dtype=np.float32)
            return

        emb = self.model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
        emb = self._normalize(emb.astype(np.float32))

        if faiss is not None:
            dim = emb.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(emb)
            self._embeddings = None
        else:
            self.index = None
            self._embeddings = emb

    def save_index(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if self.index is None and self._embeddings is None:
            raise RuntimeError("No index to save")

        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(out / "faiss.index"))
        elif self._embeddings is not None:
            np.save(out / "embeddings.npy", self._embeddings)
        (out / "chunks.json").write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")
        (out / "config.json").write_text(json.dumps({"model_name": self.model_name}, indent=2), encoding="utf-8")

    def load_index(self, path: str) -> None:
        src = Path(path)
        if faiss is not None and (src / "faiss.index").exists():
            self.index = faiss.read_index(str(src / "faiss.index"))
            self._embeddings = None
        elif (src / "embeddings.npy").exists():
            self.index = None
            self._embeddings = np.load(src / "embeddings.npy")
        else:
            self.index = None
            self._embeddings = None
        self.chunks = json.loads((src / "chunks.json").read_text(encoding="utf-8"))
        config = json.loads((src / "config.json").read_text(encoding="utf-8"))
        saved_model = config.get("model_name", self.model_name)
        if saved_model != self.model_name:
            self.model_name = saved_model
            self.model = SentenceTransformer(self.model_name)

        if self.index is None and self._embeddings is None and self.chunks:
            # Backward compatibility for index folders that only contain faiss.index when faiss isn't available.
            self._build_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        doc_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.chunks:
            return []

        q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        q = self._normalize(q)

        search_k = min(max(top_k * 6, top_k), len(self.chunks))
        if self.index is not None:
            scores, indices = self.index.search(q, search_k)
            row_scores = scores[0]
            row_indices = indices[0]
        else:
            if self._embeddings is None:
                self._build_index()
            if self._embeddings is None or len(self._embeddings) == 0:
                return []
            all_scores = (self._embeddings @ q[0]).astype(np.float32)
            row_indices = np.argsort(-all_scores)[:search_k]
            row_scores = all_scores[row_indices]

        out: list[dict[str, Any]] = []
        for score, idx in zip(row_scores, row_indices):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            if doc_type_filter and chunk.get("doc_type") != doc_type_filter:
                continue
            chunk["similarity_score"] = float(score)
            out.append(chunk)
            if len(out) >= top_k:
                break

        out.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return out

    def retrieve_for_subflow(self, subflow: str, top_k: int = 2) -> list[dict[str, Any]]:
        mapping_path = Path(__file__).resolve().parent / "subflow_mapping.json"
        mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
        entry = mapping.get(subflow, {})
        query = entry.get("query_expansion", subflow.replace("_", " "))
        return self.retrieve(query=query, top_k=top_k, doc_type_filter="support_policies")
