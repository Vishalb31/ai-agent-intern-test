# Aster & Row — Reliable AI Customer Support Agent

An evaluation-driven, privacy-preserving AI customer support agent for Aster & Row built with Python, OpenAI (`gpt-4o` and `text-embedding-3-small`), Pydantic, and vector similarity search.

---
## Demo

[![Watch Demo Walkthrough](https://img.youtube.com/vi/6wcFhtPIMIQ/0.jpg)](https://youtu.be/6wcFhtPIMIQ)

*Click the thumbnail above to watch the video demonstration walkthrough on YouTube.*
---

## Tech Stack & Architecture

- **LLM**: OpenAI `gpt-4o` (low temperature, strict JSON structured schema enforcement)
- **Embeddings**: `text-embedding-3-small` (in-memory NumPy cosine similarity)
- **Data Validation & Privacy**: Pydantic v2 schemas for strict tool outputs and sanitized prompt payloads
- **Testing**: `pytest` and custom evaluation runner (`evaluation/evaluate.py`)

### System Architecture

1. **Ingestion & Document Precedence (`src/rag/indexer.py`)**:
   - Parses YAML front-matter from all Markdown files in `knowledge-base/`.
   - Filters out legacy or non-authoritative documents: any document marked `status: superseded`, `audience: internal`, or `policy_authority: none` (e.g., `02-returns-policy-legacy.md`, `14-internal-content-migration-notes.md`) is excluded from indexing.
   - Splits documents by Markdown headers (`#`, `##`, `###`) to preserve context, attaching metadata (`filename` and `heading`) to each chunk.

2. **Retrieval (`src/rag/retriever.py`)**:
   - Computes normalized vector embeddings and queries chunks using dot-product cosine similarity.
   - Preserves exact source tags (`[filename#heading]`) required for customer citations.

3. **Tool Execution & Data Sanitization (`src/tools/order_tool.py`)**:
   - Normalizes order IDs using regex (stripping casing, spaces, and punctuation).
   - Enforces an allowlist projection (`SafeOrderResult`): strips customer names, emails, physical shipping addresses, warehouse tags, and internal risk scores before context is passed to the LLM.
   - Enforces status precedence: if `status` is `cancelled` or `returned`, any stale `estimated_delivery` dates or tracking fields are suppressed.

4. **Multi-Turn Orchestration & Defensive Synthesis (`src/agent/core.py`)**:
   - Maintains conversation history across turns with sliding-window memory and query contextualization for follow-up questions.
   - Wraps retrieved passages in `<retrieved_context>` tags to isolate internal text from execution instructions.
   - Programs deterministic fallback triggers for human handoff when items conflict, information is absent, or order statuses are flagged as `exception`.

---

## Setup and Run Instructions

### 1. Prerequisites & Environment Setup
```bash
git clone <your-repo-url>
cd ai-agent-intern-test
conda create -n aster-agent python=3.11 -y
conda activate aster-agent
pip install -r requirements.txt