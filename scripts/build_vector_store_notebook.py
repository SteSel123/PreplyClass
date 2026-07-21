"""Generate the modular vector store (Chroma) lecture notebook."""
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
# Vector Store Indexing for RAG (Chroma)

**Lecture goal:** Load chunks and embeddings from Notebook 2 into a **vector database** so we can search them at query time.

## What you will learn

1. Why we need a vector database in RAG
2. Why **Chroma** is a good choice for local development and teaching
3. How to load precomputed embeddings (no re-embedding on insert)
4. How to query the database and inspect retrieved chunks

**Prerequisite:** Run Notebook 2 through **Step 9** so these files exist:
- `data/extracted/{SOURCE_ID}_chunks.json`
- `data/extracted/{SOURCE_ID}_embeddings.npy`

---

## Pipeline at a glance

```mermaid
flowchart LR
    A[Load chunks JSON] --> B[Load embeddings NPY]
    B --> C[Connect to Chroma]
    C --> D[Create collection]
    D --> E[Insert vectors and metadata]
    E --> F[Query with question embedding]
    F --> G[Top matching chunks]
```

**Teaching tip:** Same pattern as earlier notebooks - one function per step, then one pipeline runner.
"""
    )
)

cells.append(
    md(
        """
## Part 1 - Why a vector database?

After **Transform**, you have:
- Text chunks on disk (`*.json`)
- Embedding vectors on disk (`*.npy`)

That is enough for a demo search in memory, but a real RAG app needs a **vector database** to:
- Persist indexes across restarts
- Search quickly as data grows
- Attach metadata (page numbers, source id) to each vector

```mermaid
flowchart TD
    A[Transform done] --> B[LOAD into vector DB]
    B --> C[Query time search]
    C --> D[Send chunks to LLM]

    style B fill:#4a90d9,color:#fff
```

### Why Chroma?

| Option | Good for teaching? | Notes |
|--------|-------------------|-------|
| **Chroma** | Yes | Local, free, persistent, simple Python API |
| Pinecone | Production | Hosted, needs account |
| Weaviate | Production | More setup |
| FAISS | Research | Library only, no metadata DB |

We use **Chroma** with a local folder (`data/chroma/`). No server install required.
"""
    )
)

cells.append(
    md(
        """
## Part 2 - Setup

Run the install cell once, then the config cell.

| Package | Role |
|---------|------|
| `chromadb` | Local vector database |
| `google-genai` | Embed search queries (same model as Notebook 2) |
| `python-dotenv` | Load `GOOGLE_API_KEY` from `.env` |
"""
    )
)

cells.append(code("%pip install -q chromadb google-genai python-dotenv pandas numpy"))

cells.append(
    code(
        """
# --- Pipeline configuration ---

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

_config_root = Path.cwd()
if _config_root.name == "notebooks":
    _config_root = _config_root.parent
load_dotenv(_config_root / ".env")

# Must match SOURCE_ID from Notebook 2
SOURCE_ID = "1706.03762"

# Must match Notebook 2 embedding settings
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768

# Chroma settings
CHROMA_PATH = _config_root / "data" / "chroma"
COLLECTION_NAME = f"pdf_{SOURCE_ID.replace('.', '_')}"
TOP_K = 3

print("Config loaded.")
print(f"  Source ID: {SOURCE_ID}")
print(f"  Chroma path: {CHROMA_PATH}")
print(f"  Collection: {COLLECTION_NAME}")
print(f"  GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")
"""
    )
)

cells.append(
    md(
        """
---

## Step 1 - Load chunks and embeddings from disk

**Goal:** Read the files saved by Notebook 2.

**Functions:**
- `load_saved_chunks(source_id)`
- `load_saved_embeddings(source_id)`
"""
    )
)

cells.append(
    code(
        """
def get_project_root() -> Path:
    root = Path.cwd()
    return root.parent if root.name == "notebooks" else root


def extracted_paths(source_id: str) -> tuple[Path, Path]:
    base = get_project_root() / "data" / "extracted"
    chunks_path = base / f"{source_id}_chunks.json"
    vectors_path = base / f"{source_id}_embeddings.npy"
    return chunks_path, vectors_path


def load_saved_chunks(source_id: str) -> list[dict[str, Any]]:
    chunks_path, _ = extracted_paths(source_id)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing {chunks_path}. Run Notebook 2 through Step 9 first."
        )
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"[Load] {len(chunks)} chunks from {chunks_path.name}")
    return chunks


def load_saved_embeddings(source_id: str) -> np.ndarray:
    _, vectors_path = extracted_paths(source_id)
    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Missing {vectors_path}. Run Notebook 2 through Step 9 first."
        )
    vectors = np.load(vectors_path)
    print(f"[Load] Embeddings shape {vectors.shape} from {vectors_path.name}")
    return vectors


# --- Run Step 1 ---
chunks = load_saved_chunks(SOURCE_ID)
embeddings = load_saved_embeddings(SOURCE_ID)

if len(chunks) != len(embeddings):
    raise ValueError(
        f"Chunk/embedding count mismatch: {len(chunks)} chunks vs {len(embeddings)} vectors"
    )
"""
    )
)

cells.append(
    md(
        """
## Step 2 - Connect to Chroma

**Goal:** Open a persistent local Chroma database.

**Function:** `connect_chroma(path) -> Client`
"""
    )
)

cells.append(
    code(
        """
def connect_chroma(path: Path = CHROMA_PATH) -> chromadb.PersistentClient:
    \"\"\"Create a persistent Chroma client stored on disk.\"\"\"
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    print(f"[Chroma] Connected at {path}")
    return client


# --- Run Step 2 ---
chroma_client = connect_chroma()
"""
    )
)

cells.append(
    md(
        """
## Step 3 - Create a collection and load vectors

**Goal:** Insert chunks with their **precomputed** embeddings.

We pass embeddings directly so Chroma does not re-embed with a different model.

**Functions:**
- `get_or_reset_collection(client, name)`
- `index_chunks(collection, chunks, embeddings)`
"""
    )
)

cells.append(
    code(
        """
def get_or_reset_collection(
    client: chromadb.PersistentClient,
    name: str,
    *,
    reset: bool = True,
) -> chromadb.Collection:
    \"\"\"Create a fresh collection (delete old one if reset=True).\"\"\"
    if reset:
        try:
            client.delete_collection(name)
            print(f"[Chroma] Deleted existing collection: {name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[Chroma] Collection ready: {name}")
    return collection


def index_chunks(
    collection: chromadb.Collection,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    batch_size: int = 100,
) -> int:
    \"\"\"Insert chunk text, metadata, and precomputed embeddings.\"\"\"
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_vectors = embeddings[start : start + batch_size]

        collection.add(
            ids=[item["chunk_id"] for item in batch_chunks],
            documents=[item["text"] for item in batch_chunks],
            embeddings=batch_vectors.tolist(),
            metadatas=[
                {
                    "page_number": int(item.get("page_number") or -1),
                    "source_id": str(item.get("metadata", {}).get("source_id", SOURCE_ID)),
                    "artifact_index": int(item.get("metadata", {}).get("artifact_index", -1)),
                }
                for item in batch_chunks
            ],
        )
        total += len(batch_chunks)
        print(f"[Chroma] Indexed {total}/{len(chunks)}")

    return total


# --- Run Step 3 ---
collection = get_or_reset_collection(chroma_client, COLLECTION_NAME, reset=True)
indexed_count = index_chunks(collection, chunks, embeddings)
print(f"[Chroma] Indexed {indexed_count} vectors")
"""
    )
)

cells.append(
    md(
        """
## Step 4 - Embed a search query

At query time we embed the **question** with the same Google model and task type as Notebook 2.

**Functions:**
- `configure_google_client()`
- `embed_query(text)`
"""
    )
)

cells.append(
    code(
        """
_google_client: genai.Client | None = None


def load_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key.strip()
    raise EnvironmentError("GOOGLE_API_KEY not found. Add it to rag/.env")


def configure_google_client() -> genai.Client:
    global _google_client
    if _google_client is None:
        _google_client = genai.Client(api_key=load_google_api_key())
        print(f"[Embed] Query client ready ({EMBEDDING_MODEL}, dim={EMBEDDING_DIMENSION})")
    return _google_client


def embed_query(text: str) -> list[float]:
    \"\"\"Embed a user question for retrieval search.\"\"\"
    client = configure_google_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    return response.embeddings[0].values


print("Query embedding helpers defined.")
"""
    )
)

cells.append(
    md(
        """
## Step 5 - Search the vector database

**Goal:** Retrieve the top-k most similar chunks for a question.

**Function:** `search_collection(collection, query, top_k)`
"""
    )
)

cells.append(
    code(
        """
def search_collection(
    collection: chromadb.Collection,
    query: str,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    \"\"\"Query Chroma and return ranked chunks as a table.\"\"\"
    query_vector = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    rows = []
    for rank, (doc, meta, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        preview = doc[:120].replace("\\n", " ")
        rows.append(
            {
                "rank": rank,
                "distance": round(float(distance), 4),
                "page": meta.get("page_number"),
                "preview": preview,
            }
        )

    return pd.DataFrame(rows)


print("Search helper defined.")
"""
    )
)

cells.append(
    md(
        """
## Step 6 - Wire the pipeline together

```text
run_indexing_pipeline(source_id)
  |-- load_saved_chunks()
  |-- load_saved_embeddings()
  |-- connect_chroma()
  |-- index_chunks()
  +-- return (client, collection, stats)
```
"""
    )
)

cells.append(
    code(
        """
def run_indexing_pipeline(
    source_id: str,
    *,
    reset_collection: bool = True,
) -> tuple[chromadb.PersistentClient, chromadb.Collection, dict]:
    \"\"\"Load files from disk and index them into Chroma.\"\"\"
    loaded_chunks = load_saved_chunks(source_id)
    loaded_embeddings = load_saved_embeddings(source_id)

    if len(loaded_chunks) != len(loaded_embeddings):
        raise ValueError("Chunk and embedding counts do not match")

    client = connect_chroma()
    coll = get_or_reset_collection(client, COLLECTION_NAME, reset=reset_collection)
    count = index_chunks(coll, loaded_chunks, loaded_embeddings)

    stats = {
        "source_id": source_id,
        "collection_name": COLLECTION_NAME,
        "vectors_indexed": count,
        "embedding_dim": int(loaded_embeddings.shape[1]),
        "chroma_path": str(CHROMA_PATH),
    }
    print("[Pipeline] Indexing complete.")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return client, coll, stats


print("Pipeline runner defined: run_indexing_pipeline()")
"""
    )
)

cells.append(
    md(
        """
## Step 7 - Run the pipeline and search
"""
    )
)

cells.append(
    code(
        """
chroma_client, collection, index_stats = run_indexing_pipeline(SOURCE_ID)
"""
    )
)

cells.append(
    code(
        """
demo_query = "What is the attention mechanism?"
print(f"Query: {demo_query}\\n")
search_collection(collection, demo_query, top_k=TOP_K)
"""
    )
)

cells.append(
    md(
        """
---

## What happens next?

The vector database is ready. The final RAG step is **Generation**:

```mermaid
flowchart LR
    A[User question] --> B[Embed query]
    B --> C[Search Chroma]
    C --> D[Top chunks]
    D --> E[LLM prompt]
    E --> F[Grounded answer with citations]
```

### Key takeaways

1. **Chroma stores vectors + text + metadata together** - page numbers survive indexing.
2. **Use precomputed embeddings on insert** - keeps the same model as Notebook 2.
3. **Query embeddings use RETRIEVAL_QUERY** - paired with RETRIEVAL_DOCUMENT at index time.
4. **PersistentClient** saves data under `data/chroma/` - survives notebook restarts.
5. **Same pipeline pattern** - small functions, one runner.

### Try it yourself

- Change `demo_query` and re-run the search cell
- Set `reset=False` in `get_or_reset_collection()` to append instead of rebuild
- Index a second PDF with a different `SOURCE_ID` and separate collection name

### Further reading

- [Chroma documentation](https://docs.trychroma.com/)
- [Google Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
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
    out = Path(__file__).resolve().parents[1] / "notebooks" / "rag-vector-store-lecture.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"Wrote {out} with {len(cells)} cells")
