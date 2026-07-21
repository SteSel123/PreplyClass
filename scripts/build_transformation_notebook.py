"""Generate the modular chunking + embeddings lecture notebook."""
import json
from pathlib import Path


def md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


cells = []

cells.append(
    md(
        """
# Chunking and Embeddings for RAG

**Lecture goal:** Build the **Transform** pipeline that turns extracted text artifacts into searchable **chunks** and **embedding vectors**.

## What you will learn

1. Why raw page text must be cleaned and split before indexing
2. How to chunk text with overlap while keeping page numbers
3. How to generate embeddings with the **Google Gemini embedding API**
4. How to wire it all into one pipeline function

**Prerequisite:** Run Notebook 1 (`rag-pdf-extraction-lecture.ipynb`) first and complete through **Step 8 (Save text artifacts)**.

---

## Pipeline at a glance

```mermaid
flowchart LR
    A[Load text artifacts] --> B[Clean text]
    B --> C[Chunk with overlap]
    C --> D[Embed chunks]
    D --> E[Save chunks and vectors]
    E --> F[Demo similarity search]
```

**Teaching tip:** Each step below is one small function. At the end, `run_transformation_pipeline()` calls them in order.
"""
    )
)

cells.append(
    md(
        """
## Part 1 - Why Transform?

After **Extract**, you have page-level text artifacts. That is still too large and noisy for vector search.

| Problem | Transform fix |
|---------|---------------|
| Pages are long | Split into ~500 character **chunks** |
| Chunks lose context at boundaries | Use **overlap** between chunks |
| Search needs numbers, not strings | Convert each chunk to an **embedding vector** |
| Answers need citations | Keep **page_number** on every chunk |

```mermaid
flowchart TD
    A[EXTRACT done] --> B[TRANSFORM]
    B --> C[Clean text]
    C --> D[Chunk text]
    D --> E[Generate embeddings]
    E --> F[Ready for vector DB]

    style B fill:#4a90d9,color:#fff
```

**Today we build the Transform stage** as a pipeline of small functions.
"""
    )
)

cells.append(
    md(
        """
## Part 2 - Setup

Run the install cell once, then set your API key, then run the config cell.

**API key setup (pick one):**
- Export in terminal: `export GOOGLE_API_KEY=your_key`
- Or create `rag/.env` with `GOOGLE_API_KEY=your_key` (`.env` is gitignored)

| Package | Role in pipeline |
|---------|------------------|
| `google-genai` | Google Gemini embedding API |
| `python-dotenv` | Load `GOOGLE_API_KEY` from `.env` |
| `numpy` | Store and compare vectors |
| `pandas` | Inspect chunks in a table |
"""
    )
)

cells.append(code("%pip install -q google-genai python-dotenv pandas numpy"))

cells.append(
    code(
        """
# --- Pipeline configuration ---

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from rag/.env if present
_config_root = Path.cwd()
if _config_root.name == "notebooks":
    _config_root = _config_root.parent
load_dotenv(_config_root / ".env")

# Must match ARXIV_ID from Notebook 1 Step 8
SOURCE_ID = "1706.03762"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Google embedding settings
# gemini-embedding-001 supports Matryoshka dims such as 768, 1536, or 3072
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
EMBED_BATCH_SIZE = 50

print("Config loaded.")
print(f"  Source ID: {SOURCE_ID}")
print(f"  Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
print(f"  Embedding model: {EMBEDDING_MODEL}")
print(f"  Embedding dimension: {EMBEDDING_DIMENSION}")
print(f"  GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")
"""
    )
)

cells.append(
    md(
        """
---

## Step 1 - Load text artifacts from Notebook 1

**Goal:** Read the JSON file saved at the end of Notebook 1.

**Function:** `load_text_artifacts(source_id) -> dict`
"""
    )
)

cells.append(
    code(
        """
def get_project_root() -> Path:
    \"\"\"Find the rag/ project root whether cwd is rag/ or rag/notebooks/.\"\"\"
    root = Path.cwd()
    return root.parent if root.name == "notebooks" else root


def load_text_artifacts(source_id: str) -> dict:
    \"\"\"Load page-level text artifacts produced by Notebook 1.\"\"\"
    path = get_project_root() / "data" / "extracted" / f"{source_id}_text_artifacts.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Notebook 1 through Step 8 first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"[Load] {payload['record_count']} text records from {path.name}")
    return payload


# --- Run Step 1 ---
artifact_payload = load_text_artifacts(SOURCE_ID)
text_records = artifact_payload["records"]
"""
    )
)

cells.append(
    md(
        """
## Step 2 - Define the output: `TextChunk`

Each chunk is a searchable unit with metadata for citations.

| Field | Meaning |
|-------|---------|
| `chunk_id` | Unique ID for this chunk |
| `text` | Chunk content |
| `page_number` | Source page from the PDF |
| `char_start` / `char_end` | Position inside the source page text |
| `metadata` | Extra context (source id, artifact index) |
"""
    )
)

cells.append(
    code(
        """
@dataclass
class TextChunk:
    \"\"\"One searchable text chunk ready for embedding.\"\"\"

    chunk_id: str
    text: str
    page_number: int | None
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def preview(self, max_chars: int = 80) -> str:
        snippet = self.text[:max_chars].replace("\\n", " ")
        return f"p{self.page_number}: {snippet}..."


print("TextChunk defined.")
"""
    )
)

cells.append(
    md(
        """
## Step 3 - Clean text

**Goal:** Normalize whitespace and remove characters that break downstream tools.

**Function:** `clean_text(text) -> str`
"""
    )
)

cells.append(
    code(
        """
def clean_text(text: str) -> str:
    \"\"\"Normalize page text before chunking.\"\"\"
    text = text.replace("\\x00", " ")
    text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    text = re.sub(r"[ \\t]+", " ", text)
    text = re.sub(r"\\n{3,}", "\\n\\n", text)
    return text.strip()


# --- Demo Step 3 ---
if text_records:
    raw = text_records[0]["text"][:200]
    cleaned = clean_text(text_records[0]["text"])[:200]
    print("Before:", repr(raw[:80]))
    print("After: ", repr(cleaned[:80]))
"""
    )
)

cells.append(
    md(
        """
## Step 4 - Chunk text with overlap

**Goal:** Split long page text into smaller overlapping windows.

Why overlap? A sentence split across two chunks still appears in full in at least one chunk.

**Functions:**
- `chunk_page_text(...)` - chunk one page
- `chunk_all_records(...)` - chunk every loaded record
"""
    )
)

cells.append(
    code(
        """
def chunk_page_text(
    text: str,
    *,
    page_number: int | None,
    source_id: str,
    artifact_index: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    \"\"\"Split one page of text into overlapping character chunks.\"\"\"
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=piece,
                    page_number=page_number,
                    char_start=start,
                    char_end=end,
                    metadata={
                        "source_id": source_id,
                        "artifact_index": artifact_index,
                    },
                )
            )
        if end >= len(cleaned):
            break
        start += chunk_size - overlap

    return chunks


def chunk_all_records(records: list[dict], source_id: str) -> list[TextChunk]:
    \"\"\"Chunk every text record loaded from Notebook 1.\"\"\"
    all_chunks: list[TextChunk] = []
    for record in records:
        all_chunks.extend(
            chunk_page_text(
                record["text"],
                page_number=record.get("page_number"),
                source_id=source_id,
                artifact_index=record.get("artifact_index", -1),
            )
        )
    print(f"[Chunk] Created {len(all_chunks)} chunks from {len(records)} pages")
    return all_chunks


# --- Demo Step 4 ---
chunks = chunk_all_records(text_records, SOURCE_ID)
print("Sample chunk:", chunks[0].preview() if chunks else "No chunks")
"""
    )
)

cells.append(
    md(
        """
## Step 5 - Generate embeddings

**Goal:** Convert each chunk into a dense vector for similarity search.

We use **Google `gemini-embedding-001`** via the Gemini API:
- **768 dimensions** (good balance for RAG; you can also try `1536` or `3072`)
- **`RETRIEVAL_DOCUMENT`** for chunks, **`RETRIEVAL_QUERY`** for search questions

**Functions:**
- `configure_google_client()` - set API key once
- `embed_texts(texts, task_type)` - batch encode via Google API
"""
    )
)

cells.append(
    code(
        """
_google_client: genai.Client | None = None


def load_google_api_key() -> str:
    \"\"\"Read GOOGLE_API_KEY from environment or .env file.\"\"\"
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key.strip()
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. Export it or add it to rag/.env"
    )


def configure_google_client() -> genai.Client:
    \"\"\"Create the Google GenAI client (reused across calls).\"\"\"
    global _google_client
    if _google_client is None:
        _google_client = genai.Client(api_key=load_google_api_key())
        print("[Embed] Google API ready")
        print(f"[Embed] Model: {EMBEDDING_MODEL}, dim: {EMBEDDING_DIMENSION}")
    return _google_client


def embed_texts(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    \"\"\"Embed strings via Google gemini-embedding-001.\"\"\"
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    client = configure_google_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )
        vectors.extend([item.values for item in response.embeddings])
        print(f"[Embed] Encoded {min(start + len(batch), len(texts))}/{len(texts)}")

    return np.asarray(vectors, dtype=np.float32)


# --- Demo Step 5 ---
chunk_texts = [chunk.text for chunk in chunks]
embeddings = embed_texts(chunk_texts, task_type="RETRIEVAL_DOCUMENT")
print(f"[Embed] Shape: {embeddings.shape}")
"""
    )
)

cells.append(
    md(
        """
## Step 6 - Similarity search helper

**Goal:** Given a question, find the most similar chunks by cosine similarity.

**Functions:**
- `cosine_similarity(query_vec, matrix)` 
- `search_chunks(query, chunks, embeddings, top_k)`
"""
    )
)

cells.append(
    code(
        """
def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    \"\"\"Cosine similarity between one query vector and many chunk vectors.\"\"\"
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return matrix_norm @ query_norm


def search_chunks(
    query: str,
    chunks: list[TextChunk],
    embeddings: np.ndarray,
    top_k: int = 3,
) -> pd.DataFrame:
    \"\"\"Return the top-k most similar chunks for a natural-language query.\"\"\"
    query_vector = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    scores = cosine_similarity(query_vector, embeddings)
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    rows = []
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = chunks[int(idx)]
        rows.append(
            {
                "rank": rank,
                "score": round(float(scores[idx]), 4),
                "page": chunk.page_number,
                "preview": chunk.preview(120),
            }
        )
    return pd.DataFrame(rows)


print("Search helpers defined.")
"""
    )
)

cells.append(
    md(
        """
## Step 7 - Wire the pipeline together

**Goal:** One orchestrator function for the full Transform stage.

```text
run_transformation_pipeline(source_id)
  |-- load_text_artifacts()
  |-- chunk_all_records()
  |-- configure_google_client()
  |-- embed_texts()
  +-- return (chunks, embeddings, stats)
```
"""
    )
)

cells.append(
    code(
        """
def run_transformation_pipeline(
    source_id: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[list[TextChunk], np.ndarray, dict]:
    \"\"\"Run the full Transform pipeline: load, chunk, embed.\"\"\"
    payload = load_text_artifacts(source_id)
    records = payload["records"]

    chunks = chunk_all_records(records, source_id)
    configure_google_client()
    vectors = embed_texts([chunk.text for chunk in chunks], task_type="RETRIEVAL_DOCUMENT")

    stats = {
        "source_id": source_id,
        "pages_loaded": len(records),
        "chunks_created": len(chunks),
        "embedding_dim": int(vectors.shape[1]) if vectors.size else 0,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "model_name": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
    }
    print("[Pipeline] Transform complete.")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return chunks, vectors, stats


print("Pipeline runner defined: run_transformation_pipeline()")
"""
    )
)

cells.append(
    md(
        """
## Step 8 - Run the pipeline
"""
    )
)

cells.append(
    code(
        """
print("[Pipeline] Starting transformation...")
chunks, embeddings, transform_stats = run_transformation_pipeline(SOURCE_ID)
"""
    )
)

cells.append(
    md(
        """
## Step 9 - Save outputs and inspect results
"""
    )
)

cells.append(
    code(
        """
def save_transformation_output(
    chunks: list[TextChunk],
    embeddings: np.ndarray,
    source_id: str,
    stats: dict[str, Any],
) -> tuple[Path, Path, Path]:
    \"\"\"Save chunks JSON and embedding matrix for the next pipeline stage.\"\"\"
    output_dir = get_project_root() / "data" / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / f"{source_id}_chunks.json"
    vectors_path = output_dir / f"{source_id}_embeddings.npy"
    stats_path = output_dir / f"{source_id}_transform_stats.json"

    chunk_records = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "page_number": chunk.page_number,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    chunks_path.write_text(json.dumps(chunk_records, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(vectors_path, embeddings)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[Save] Chunks -> {chunks_path}")
    print(f"[Save] Embeddings -> {vectors_path}")
    print(f"[Save] Stats -> {stats_path}")
    return chunks_path, vectors_path, stats_path


def chunks_to_dataframe(chunks: list[TextChunk]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": chunk.chunk_id[:8],
                "page": chunk.page_number,
                "chars": len(chunk.text),
                "preview": chunk.preview(100),
            }
            for chunk in chunks
        ]
    )


# --- Run Step 9 ---
save_transformation_output(chunks, embeddings, SOURCE_ID, transform_stats)
df_chunks = chunks_to_dataframe(chunks)
print(f"\\nTotal chunks: {len(df_chunks)}")
df_chunks.head(10)
"""
    )
)

cells.append(
    code(
        """
# Demo: ask a question and retrieve the best matching chunks
demo_query = "What is the attention mechanism?"
search_chunks(demo_query, chunks, embeddings, top_k=3)
"""
    )
)

cells.append(
    md(
        """
---

## What happens next?

Transform is **stage 2**. Open **Notebook 3** (`rag-vector-store-lecture.ipynb`) to load chunks and embeddings into **Chroma**.

Next steps in a full RAG system:

```mermaid
flowchart LR
    A[Chunks and embeddings] --> B[Store in vector DB]
    B --> C[User query]
    C --> D[Retrieve top chunks]
    D --> E[Send to LLM]
    E --> F[Grounded answer]
```

### Key takeaways

1. **Transform turns page text into search units** - chunks small enough to retrieve precisely.
2. **Overlap reduces boundary errors** - important sentences are not cut in half without backup.
3. **Embeddings capture meaning** - similar questions match relevant chunks even without exact keywords.
4. **Keep page numbers on chunks** - citations still work after splitting.
5. **Pipeline pattern again** - small functions plus one runner, same style as Notebook 1.

### Try it yourself

- Change `demo_query` and re-run the search cell
- Try `CHUNK_SIZE = 300` and `CHUNK_OVERLAP = 50`, then re-run from Step 4
- Use a different `SOURCE_ID` after running Notebook 1 on another arXiv paper

### Further reading

- [Google Gemini embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings)
- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- RAG overview: search "Retrieval Augmented Generation Lewis 2020"
"""
    )
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (pdf-rag-pipeline)",
            "language": "python",
            "name": "pdf-rag-pipeline",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "rag-chunking-embeddings-lecture.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"Wrote {out} with {len(cells)} cells")
