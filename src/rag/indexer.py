import re
from pathlib import Path
from typing import Any, Dict, List
import yaml
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    chunk_id: str
    filename: str
    title: str
    heading: str
    content: str
    metadata: Dict[str, Any]


class KnowledgeBaseIndexer:
    def __init__(self, kb_dir: Path | str = "knowledge-base"):
        self.kb_dir = Path(kb_dir)

    def parse_front_matter(self, raw_text: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML front matter delimited by '---' and return metadata + body."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, raw_text, re.DOTALL)
        if match:
            front_yaml, body = match.groups()
            try:
                metadata = yaml.safe_load(front_yaml) or {}
            except Exception:
                metadata = {}
            return metadata, body
        return {}, raw_text

    def is_eligible_document(self, metadata: Dict[str, Any]) -> bool:
        """
        Deterministic filter to drop:
        - Internal drafts / migration scratchpads (audience == internal or customer_answering is False)
        - Superseded / legacy policies (status == superseded or policy_authority == none)
        """
        if metadata.get("customer_answering") is False:
            return False
        if metadata.get("status") in ["superseded", "draft", "deprecated"]:
            return False
        if metadata.get("audience") in ["internal"]:
            return False
        if metadata.get("policy_authority") in ["none"]:
            return False
        return True

    def chunk_markdown_by_headings(self, body: str, filename: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Splits markdown by level 1-3 headings while preserving heading context."""
        lines = body.split("\n")
        chunks: List[DocumentChunk] = []
        current_heading = metadata.get("title", filename)
        current_lines: List[str] = []
        chunk_idx = 0

        heading_pattern = re.compile(r"^(#{1,3})\s+(.*)$")

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                # Flush existing chunk
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{filename}_{chunk_idx}",
                            filename=filename,
                            title=metadata.get("title", filename),
                            heading=current_heading,
                            content=content,
                            metadata=metadata,
                        )
                    )
                    chunk_idx += 1
                    current_lines = []
                current_heading = match.group(2).strip()
            else:
                current_lines.append(line)

        # Flush final chunk
        final_content = "\n".join(current_lines).strip()
        if final_content:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{filename}_{chunk_idx}",
                    filename=filename,
                    title=metadata.get("title", filename),
                    heading=current_heading,
                    content=final_content,
                    metadata=metadata,
                )
            )

        return chunks

    def load_indexable_chunks(self) -> List[DocumentChunk]:
        """Loads and processes all valid active policy/product documents."""
        if not self.kb_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found at {self.kb_dir}")

        all_chunks: List[DocumentChunk] = []
        for file_path in sorted(self.kb_dir.glob("*.md")):
            raw_text = file_path.read_text(encoding="utf-8")
            metadata, body = self.parse_front_matter(raw_text)

            # Enforce document precedence filter
            if not self.is_eligible_document(metadata):
                continue

            chunks = self.chunk_markdown_by_headings(body, file_path.name, metadata)
            all_chunks.extend(chunks)

        return all_chunks