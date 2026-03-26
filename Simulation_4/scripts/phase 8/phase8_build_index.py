from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Simulation_4.rag.document_store import DocumentStore
from Simulation_4.rag.retriever import Retriever


def main() -> None:
    rag_dir = PROJECT_ROOT / "Simulation_4" / "rag"
    documents_dir = rag_dir / "documents"
    index_dir = rag_dir / "index"

    print("[Phase8] Building RAG index...")
    store = DocumentStore(str(documents_dir))
    retriever = Retriever(document_store=store)
    retriever.save_index(str(index_dir))

    chunks = store.get_all_chunks()
    print(f"[Phase8] Index saved: {index_dir}")
    print(f"[Phase8] Chunk count: {len(chunks)}")
    if chunks:
        sizes = [c["content_length"] for c in chunks]
        print(f"[Phase8] Chunk size stats -> min={min(sizes)} max={max(sizes)} mean={int(sum(sizes)/len(sizes))}")


if __name__ == "__main__":
    main()
