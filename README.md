# PDF RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) pipeline for PDF documents.

## Project Structure

```
rag/
├── notebooks/
│   ├── rag-pdf-extraction-lecture.ipynb      # Step 1: Extract text, tables, images
│   ├── rag-chunking-embeddings-lecture.ipynb # Step 2: Chunk and embed text
│   └── rag-vector-store-lecture.ipynb        # Step 3: Load into Chroma
├── scripts/
│   ├── build_extraction_notebook.py
│   ├── build_transformation_notebook.py
│   └── build_vector_store_notebook.py
├── data/
│   ├── pdfs/                                 # Downloaded PDFs (gitignored)
│   ├── extracted/                            # Chunks and embeddings (gitignored)
│   └── chroma/                               # Chroma vector DB (gitignored)
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

1. Create virtual environment with `uv`:
```bash
cd rag
uv venv
source .venv/bin/activate  # on Mac/Linux
# or on Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv sync
```

3. Register the project kernel (one-time):
```bash
uv run python -m ipykernel install --user --name pdf-rag-pipeline --display-name "Python (pdf-rag-pipeline)"
```

4. Start Jupyter Lab and open the notebook:
```bash
uv run jupyter lab notebooks/rag-pdf-extraction-lecture.ipynb
```

5. If prompted, select kernel **Python (pdf-rag-pipeline)**, then run cells in order

## Notebook 2 - Chunking and Embeddings (Transform)

1. Complete **Notebook 1 through Step 8** (saves `data/extracted/{ARXIV_ID}_text_artifacts.json`)
2. Open the second notebook:
```bash
uv run jupyter lab notebooks/rag-chunking-embeddings-lecture.ipynb
```
3. Set `SOURCE_ID` in the config cell to match `ARXIV_ID` from Notebook 1
4. Set your Google API key (required for embeddings):
```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key
```
5. Run all cells in order

Outputs saved to `data/extracted/`:
- `{SOURCE_ID}_chunks.json` - searchable text chunks with page numbers
- `{SOURCE_ID}_embeddings.npy` - embedding vectors
- `{SOURCE_ID}_transform_stats.json` - pipeline statistics

## Notebook 3 - Vector Store (Load into Chroma)

1. Complete **Notebook 2 through Step 9** (saves chunks + embeddings)
2. Open the third notebook:
```bash
uv run jupyter lab notebooks/rag-vector-store-lecture.ipynb
```
3. Set `SOURCE_ID` to match Notebook 2
4. Ensure `GOOGLE_API_KEY` is in `.env` (needed for query embedding)
5. Run all cells in order

Chroma persists to `data/chroma/`. Each source gets its own collection (e.g. `pdf_1706_03762`).

## Extraction Tools

- **pdfplumber**: Best-in-class table extraction with structure preservation
- **PyMuPDF (fitz)**: Fast image extraction and PDF handling
- **Pillow**: Image processing and encoding

## Phase Overview

| Phase | Purpose | Output |
|-------|---------|--------|
| 1: Indexing | Extract text, tables, images from PDFs | Structured data files |
| 2: Chunking | Split content into overlapping chunks | Chunk metadata |
| 3: Embedding | Convert chunks to vectors | Vector embeddings |
| 4: Storage | Store in vector database | Vector DB ready |
| 5: Querying | Convert user query to vector | Retrieved chunks |
| 6: Generation | LLM answers with citations | Final response |