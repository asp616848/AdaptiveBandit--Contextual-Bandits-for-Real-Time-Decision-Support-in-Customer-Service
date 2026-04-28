from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class DocumentStore:
    def __init__(self, documents_dir: str):
        self.documents_dir = Path(documents_dir).expanduser().resolve()
        if not self.documents_dir.exists():
            raise FileNotFoundError(f"documents_dir not found: {self.documents_dir}")

        self._chunks: list[dict[str, Any]] = []
        self._per_doc_counts: dict[str, int] = {}

        for filepath in sorted(self.documents_dir.glob("*.md")):
            doc_chunks = self._chunk_document(str(filepath))
            self._chunks.extend(doc_chunks)
            self._per_doc_counts[filepath.name] = len(doc_chunks)

        self._print_summary()

    def _doc_type_from_filename(self, filename: str) -> str:
        lowered = filename.lower()
        if "doc1" in lowered or "product" in lowered or "company" in lowered:
            return "product_overview"
        if "doc2" in lowered or "policy" in lowered:
            return "support_policies"
        if "playbook" in lowered or "doc3" in lowered:
            return "playbooks"
        return "other"

    def _clean_content_for_embedding(self, text: str) -> str:
        text = text.replace("**", "").replace("*", "")
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _parse_section_number(self, title: str, fallback_idx: int) -> str:
        m = re.match(r"^(\d+(?:\.\d+)*)", title.strip())
        if m:
            return m.group(1)
        return str(fallback_idx)

    def _split_large_chunk(self, base_chunk: dict[str, Any], max_chars: int = 1000, soft_mark: int = 700) -> list[dict[str, Any]]:
        content = base_chunk["content"]
        if len(content) <= max_chars:
            return [base_chunk]

        out: list[dict[str, Any]] = []
        remaining = content
        part = 1
        while len(remaining) > max_chars:
            cut = remaining.find("\n\n", soft_mark)
            if cut == -1 or cut > max_chars:
                cut = max_chars
            chunk_text = remaining[:cut].strip()
            remaining = remaining[cut:].strip()

            new_chunk = dict(base_chunk)
            if part > 1:
                new_chunk["section_title"] = f"{base_chunk['section_title']} (continued)"
            new_chunk["chunk_id"] = f"{base_chunk['chunk_id']}_part{part}"
            new_chunk["content"] = chunk_text
            new_chunk["content_length"] = len(chunk_text)
            out.append(new_chunk)
            part += 1

        if remaining:
            new_chunk = dict(base_chunk)
            if part > 1:
                new_chunk["section_title"] = f"{base_chunk['section_title']} (continued)"
            new_chunk["chunk_id"] = f"{base_chunk['chunk_id']}_part{part}"
            new_chunk["content"] = remaining
            new_chunk["content_length"] = len(remaining)
            out.append(new_chunk)

        return out

    def _chunk_document(self, filepath: str) -> list[dict[str, Any]]:
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        sections: list[tuple[str, str]] = []
        current_title: str | None = None
        current_lines: list[str] = []

        for line in lines:
            if line.startswith("## "):
                if current_title is not None:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = line[3:].strip()
                current_lines = []
                continue
            if current_title is not None:
                current_lines.append(line)

        if current_title is not None:
            sections.append((current_title, "\n".join(current_lines).strip()))

        doc_type = self._doc_type_from_filename(path.name)

        merged_sections: list[tuple[str, str]] = []
        i = 0
        while i < len(sections):
            title, content = sections[i]
            content = self._clean_content_for_embedding(content)
            if len(content) < 150 and i + 1 < len(sections):
                next_title, next_content = sections[i + 1]
                next_content = self._clean_content_for_embedding(next_content)
                merged_sections.append((f"{title} + {next_title}", f"{content}\n\n{next_content}".strip()))
                i += 2
            else:
                merged_sections.append((title, content))
                i += 1

        chunks: list[dict[str, Any]] = []
        for idx, (section_title, content) in enumerate(merged_sections, start=1):
            section_number = self._parse_section_number(section_title, idx)
            chunk = {
                "chunk_id": f"{path.stem}_section_{idx}_{section_number.replace('.', '_')}",
                "doc_filename": path.name,
                "doc_type": doc_type,
                "section_title": section_title,
                "section_number": section_number,
                "content": content,
                "content_length": len(content),
            }
            for split_chunk in self._split_large_chunk(chunk, max_chars=1000, soft_mark=700):
                chunks.append(split_chunk)

        return chunks

    def _print_summary(self):
        sizes = [c["content_length"] for c in self._chunks]
        total = len(self._chunks)
        mean_size = int(sum(sizes) / total) if total else 0
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0

        print("DocumentStore loaded:")
        for filename, count in self._per_doc_counts.items():
            print(f"  {filename}: {count} chunks")
        print(f"  Total chunks: {total}")
        print(f"  Mean chunk size: {mean_size} characters")
        print(f"  Min chunk size: {min_size} characters")
        print(f"  Max chunk size: {max_size} characters")

    def get_all_chunks(self) -> list[dict[str, Any]]:
        return list(self._chunks)

    def get_chunks_by_doc_type(self, doc_type: str) -> list[dict[str, Any]]:
        return [c for c in self._chunks if c.get("doc_type") == doc_type]

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        for chunk in self._chunks:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None
