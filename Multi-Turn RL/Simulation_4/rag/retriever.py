from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

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
        self.model: Any | None = None
        self._model_init_error: str | None = None
        self._lexical_cache: list[set[str]] = []

        try:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed")
            self.model = SentenceTransformer(self.model_name)
        except Exception as exc:
            # Offline-safe fallback: keep running with lexical retrieval.
            self.model = None
            self._model_init_error = str(exc)

        if index_path and (Path(index_path) / "faiss.index").exists():
            self.load_index(index_path)
        else:
            self._build_index()

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return embeddings / norms

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def _ensure_lexical_cache(self) -> None:
        if self._lexical_cache and len(self._lexical_cache) == len(self.chunks):
            return
        self._lexical_cache = [self._tokenize(c.get("content", "")) for c in self.chunks]

    def _build_index(self) -> None:
        self.chunks = self.document_store.get_all_chunks()
        self._lexical_cache = []
        corpus = [c["content"] for c in self.chunks]
        if not corpus:
            if faiss is not None:
                self.index = faiss.IndexFlatIP(384)
            else:
                self.index = None
                self._embeddings = np.zeros((0, 384), dtype=np.float32)
            return

        if self.model is None:
            self.index = None
            self._embeddings = None
            self._ensure_lexical_cache()
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
            try:
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers is not installed")
                self.model = SentenceTransformer(self.model_name)
                self._model_init_error = None
            except Exception as exc:
                self.model = None
                self._model_init_error = str(exc)

        if self.index is None and self._embeddings is None and self.chunks:
            # Backward compatibility for index folders that only contain faiss.index when faiss isn't available.
            self._build_index()

    def _retrieve_lexical(
        self,
        query: str,
        top_k: int,
        doc_type_filter: str | None,
    ) -> list[dict[str, Any]]:
        self._ensure_lexical_cache()
        q_tokens = self._tokenize(query)
        if not q_tokens:
            q_tokens = set(query.lower().split())

        scored: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self.chunks):
            if doc_type_filter and chunk.get("doc_type") != doc_type_filter:
                continue
            c_tokens = self._lexical_cache[idx]
            overlap = len(q_tokens & c_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(q_tokens), 1)
            scored.append((float(score), idx))

        if not scored:
            # Deterministic fallback so caller still receives context.
            fallback: list[dict[str, Any]] = []
            for chunk in self.chunks:
                if doc_type_filter and chunk.get("doc_type") != doc_type_filter:
                    continue
                item = dict(chunk)
                item["similarity_score"] = 0.0
                fallback.append(item)
                if len(fallback) >= top_k:
                    break
            return fallback

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, idx in scored[:top_k]:
            item = dict(self.chunks[idx])
            item["similarity_score"] = score
            out.append(item)
        return out

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        doc_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.chunks:
            return []

        if self.model is None:
            return self._retrieve_lexical(query=query, top_k=top_k, doc_type_filter=doc_type_filter)

        try:
            q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            q = self._normalize(q)
        except Exception:
            self.model = None
            return self._retrieve_lexical(query=query, top_k=top_k, doc_type_filter=doc_type_filter)

        if doc_type_filter:
            # When filtering by doc_type, over-fetch from the full corpus first;
            # otherwise relevant filtered chunks can be missed in the initial top slice.
            search_k = len(self.chunks)
        else:
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
