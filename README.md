# PDF RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) pipeline for PDF documents.

## Project Structure

```
rag/
├── notebooks/
│   └── rag-pdf-extraction-lecture.ipynb  # Phase 1: Extract text, tables, images
├── scripts/
│   └── build_extraction_notebook.py      # Regenerate the lecture notebook
├── data/
│   └── pdfs/                             # Downloaded PDFs (gitignored)
├── pyproject.toml                        # Project configuration
├── uv.lock                               # Locked dependencies
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