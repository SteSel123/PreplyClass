"""Generate the modular PDF extraction lecture notebook."""
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
# PDF Extraction Pipeline for RAG

**Lecture goal:** Build a small, readable **Extract** pipeline that turns a PDF into typed **artifacts** ready for the next RAG stages.

## What you will learn

1. How RAG uses an ETL pipeline
2. How to break extraction into **modular functions**
3. How to run those functions as a **pipeline**: Acquire -> Classify -> Extract -> Inspect

This notebook is **standalone**. Each section introduces one function (or one small group), then we wire them together at the end.

---

## Pipeline at a glance

```mermaid
flowchart LR
    A[Acquire PDF] --> B[Classify PDF]
    B --> C{Slide deck?}
    C -->|Yes| D[Render pages]
    C -->|No| E[Text images tables]
    A --> F[Annotations]
    D --> G[Combine artifacts]
    E --> G
    F --> G
    G --> H[Inspect results]
```

**Teaching tip:** Read top to bottom. Run each cell in order. By the end, you will have one function - `run_pdf_extraction_pipeline()` - that calls everything else.
"""
    )
)

cells.append(
    md(
        """
## Part 1 - Why Extract?

### What is RAG?

LLMs are good at language, but they do not know your private documents. **RAG** (Retrieval Augmented Generation) retrieves relevant snippets at question time and sends them to the model.

```mermaid
flowchart LR
    A[User question] --> B[Search vector DB]
    B --> C[Relevant chunks]
    C --> D[LLM with chunks]
    D --> E[Grounded answer]
```

```mermaid
flowchart TD
    A[PDF file bytes] --> B[EXTRACT]
    B --> C[PROCESS]
    C --> D[CHUNK]
    D --> E[INDEX]
    E --> F[Vector database]

    style B fill:#4a90d9,color:#fff
    style C fill:#ddd
    style D fill:#ddd
    style E fill:#ddd
```

Before search works, documents go through **ETL**:

| Stage | PDF example |
|-------|-------------|
| **Extract** | Pull text, images, tables from PDF |
| **Process** | Clean text, caption images |
| **Chunk** | Split into ~500-char pieces |
| **Index** | Embed and store in vector DB |

**Today we build only the Extract stage**, as a pipeline of small functions.
"""
    )
)

cells.append(
    md(
        """
## Part 2 - Setup

Run the install cell once, then the imports + config cell.

| Package | Role in pipeline |
|---------|------------------|
| `pymupdf` | Open PDFs, read text/images/tables |
| `pdf-comments-extractor` | Read reviewer comments |
| `requests` | Download PDF from arXiv |
| `pandas` | Display artifact summary table |
| `Pillow` | Show images inline |
"""
    )
)

cells.append(code("%pip install -q pymupdf pdf-comments-extractor requests pandas Pillow ipython"))

cells.append(
    code(
        """
# --- Pipeline configuration ---
# One place to tune behavior. In production this would live in a config file.

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import files as pkg_files
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF
import pandas as pd
import requests
from IPython.display import display
from PIL import Image

# --- Classification thresholds ---
TEXT_RATIO_THRESHOLD = 0.75
LANDSCAPE_RATIO_THRESHOLD = 0.6
CLASSIFICATION_SAMPLE_PAGES = 6
MIN_TEXT_CHARS_FOR_TEXT_PAGE = 80

# --- Table extraction ---
TABLE_STRATEGY = "lines_strict"
TABLE_ZOOM = 2.0

# --- Input: change this to try another paper ---
ARXIV_ID = "1706.03762"

print("Config loaded.")
print(f"  Paper: arXiv:{ARXIV_ID}")
"""
    )
)

cells.append(
    md(
        """
---

## Step 1 - Acquire the PDF

**Goal:** Get PDF bytes and save them to disk so later steps can reuse the file.

**Function:** `acquire_pdf(arxiv_id) -> (pdf_bytes, pdf_path)`

This is the first stage of our pipeline: **input in, saved file out**.
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


def acquire_pdf(arxiv_id: str) -> tuple[bytes, Path]:
    \"\"\"Download a PDF from arXiv, save it locally, return bytes + path.\"\"\"
    project_root = get_project_root()
    pdf_dir = project_root / "data" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{arxiv_id}.pdf"

    if pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()
        print(f"[Acquire] Loaded existing file: {pdf_path}")
    else:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"[Acquire] Downloading: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        pdf_bytes = response.content
        pdf_path.write_bytes(pdf_bytes)
        print(f"[Acquire] Saved to: {pdf_path}")

    print(f"[Acquire] Size: {len(pdf_bytes):,} bytes ({len(pdf_bytes)/1024:.1f} KB)")
    return pdf_bytes, pdf_path


# --- Run Step 1 ---
pdf_bytes, pdf_path = acquire_pdf(ARXIV_ID)
"""
    )
)

cells.append(
    md(
        """
## Step 2 - Define the output: `ExtractArtifact`

Every pipeline stage should produce a **clear, typed output**.

We use a dataclass called `ExtractArtifact` - one labeled box per piece of content pulled from the PDF.

| Field | Meaning |
|-------|---------|
| `extract_type` | Broad category: `document`, `image`, `tabular` |
| `artifact_kind` | Specific kind: `document`, `image`, `table`, `page_render` |
| `text` | Text content |
| `image_bytes` | Raw image bytes |
| `page_number` | Source page (for citations later) |
| `metadata` | Extra context |
"""
    )
)

cells.append(
    code(
        """
ArtifactKind = Literal["document", "image", "table", "page_render"]
ExtractType = Literal["document", "image", "tabular"]


@dataclass
class ExtractArtifact:
    \"\"\"One typed piece extracted from a PDF.\"\"\"

    extract_type: ExtractType
    artifact_kind: ArtifactKind
    text: str | None = None
    image_bytes: bytes | None = None
    mime_type: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.text:
            preview = self.text[:80].replace("\\n", " ")
            return f"text: {preview}..."
        if self.image_bytes:
            return f"image: {len(self.image_bytes):,} bytes"
        return "empty artifact"


print("ExtractArtifact defined.")
"""
    )
)

cells.append(
    md(
        """
## Step 3 - Classify the PDF

**Goal:** Decide which extraction path to use.

- **Document path** -> extract text, embedded images, and tables page by page
- **Presentation path** -> render each page as a PNG (for slide decks with little text)

```mermaid
flowchart TD
    A[Open PDF with PyMuPDF] --> B{Presentation-like?}
    B -->|Yes| C[Render each page as PNG]
    B -->|No| D[For each page]
    D --> E[Extract page text]
    D --> F[Extract embedded images]
    D --> G[Detect and crop tables]
    A --> H[Extract PDF annotations]
    C --> I[List of ExtractArtifacts]
    E --> I
    F --> I
    G --> I
    H --> I
```

**Functions:**
- `iter_sample_pages(doc)` - sample pages for fast classification
- `classify_pdf(doc) -> "document" | "presentation"`
"""
    )
)

cells.append(
    code(
        """
def iter_sample_pages(doc: fitz.Document) -> Iterable[fitz.Page]:
    \"\"\"Yield a spread of sample pages for fast classification.\"\"\"
    total = len(doc)
    sample_size = max(CLASSIFICATION_SAMPLE_PAGES, 1)
    if total <= sample_size:
        yield from doc
        return
    step = max(total // sample_size, 1)
    for idx in range(0, total, step):
        yield doc[idx]
        if idx >= step * (sample_size - 1):
            break


def sampled_pages_have_tables(doc: fitz.Document) -> bool:
    \"\"\"Return True if sampled pages contain at least one table.\"\"\"
    for page in iter_sample_pages(doc):
        try:
            tables = page.find_tables(strategy=TABLE_STRATEGY)
        except Exception:
            tables = None
        if tables and getattr(tables, "tables", None) and len(tables.tables) > 0:
            return True
    return False


def classify_pdf(doc: fitz.Document) -> Literal["document", "presentation"]:
    \"\"\"Classify PDF as a regular document or a slide-deck-style presentation.\"\"\"
    if sampled_pages_have_tables(doc):
        return "document"

    sampled = text_pages = landscape_pages = 0
    for page in iter_sample_pages(doc):
        sampled += 1
        text = (page.get_text() or "").strip()
        if len(text) >= MIN_TEXT_CHARS_FOR_TEXT_PAGE:
            text_pages += 1
        rect = page.rect
        if rect.width > rect.height:
            landscape_pages += 1

    if sampled == 0:
        return "document"

    text_ratio = text_pages / sampled
    landscape_ratio = landscape_pages / sampled
    is_presentation = (
        text_ratio < TEXT_RATIO_THRESHOLD
        or landscape_ratio >= LANDSCAPE_RATIO_THRESHOLD
    )
    return "presentation" if is_presentation else "document"


# --- Demo Step 3 ---
with fitz.open(stream=pdf_bytes, filetype="pdf") as demo_doc:
    pdf_type = classify_pdf(demo_doc)
    print(f"[Classify] Result: {pdf_type}")
"""
    )
)

cells.append(
    md(
        """
## Step 4 - Extraction functions (one job each)

Each function below handles **one type of content**. This keeps the code easy to read, test, and teach.

| Function | Extracts |
|----------|----------|
| `extract_page_text(...)` | Selectable text on one page |
| `extract_page_images(...)` | Embedded images on one page |
| `extract_page_tables(...)` | Table regions cropped as PNG |
| `extract_presentation_pages(...)` | Full-page renders for slide decks |
| `extract_annotations(...)` | PDF reviewer comments |
"""
    )
)

cells.append(
    code(
        """
def extract_page_text(
    page: fitz.Page,
    page_number: int,
    total_pages: int,
) -> ExtractArtifact | None:
    \"\"\"Extract text from a single page.\"\"\"
    text = (page.get_text() or "").strip()
    if not text:
        return None
    return ExtractArtifact(
        extract_type="document",
        artifact_kind="document",
        text=text.replace("\\x00", ""),
        page_number=page_number,
        metadata={"total_pages": total_pages, "page_number": page_number},
    )


def extract_page_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_number: int,
    page_text: str,
) -> list[ExtractArtifact]:
    \"\"\"Extract embedded images from a single page.\"\"\"
    artifacts: list[ExtractArtifact] = []
    page_text_context = page_text[:500] if page_text else ""

    for image_info in page.get_images(full=True) or []:
        xref = int(image_info[0])
        image_dict = doc.extract_image(xref)
        image_bytes = image_dict.get("image")
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            continue
        ext = str(image_dict.get("ext") or "").lower()
        mime_type = (
            f"image/{ext}"
            if ext in {"png", "jpeg", "jpg", "webp", "gif"}
            else "image/jpeg"
        )
        artifacts.append(
            ExtractArtifact(
                extract_type="image",
                artifact_kind="image",
                image_bytes=bytes(image_bytes),
                mime_type=mime_type,
                page_number=page_number,
                metadata={"xref": xref, "page_text_context": page_text_context},
            )
        )
    return artifacts


def extract_page_tables(
    page: fitz.Page,
    page_number: int,
    page_text: str,
) -> list[ExtractArtifact]:
    \"\"\"Detect tables on one page and crop each as a PNG artifact.\"\"\"
    artifacts: list[ExtractArtifact] = []
    page_text_context = page_text[:500] if page_text else ""

    try:
        tables = page.find_tables(strategy=TABLE_STRATEGY)
    except Exception:
        tables = None

    for table_index, table in enumerate((tables.tables if tables else []), start=1):
        clip = fitz.Rect(table.bbox)
        mat = fitz.Matrix(TABLE_ZOOM, TABLE_ZOOM)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        artifacts.append(
            ExtractArtifact(
                extract_type="image",
                artifact_kind="table",
                image_bytes=pix.tobytes("png"),
                mime_type="image/png",
                page_number=page_number,
                metadata={
                    "table_index": table_index,
                    "page_text_context": page_text_context,
                },
            )
        )
    return artifacts


print("Document extraction helpers defined.")
"""
    )
)

cells.append(
    code(
        """
def extract_presentation_pages(
    doc: fitz.Document,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[ExtractArtifact]:
    \"\"\"Render every page as PNG - used for slide-deck PDFs.\"\"\"
    artifacts: list[ExtractArtifact] = []
    total_pages = len(doc)

    for page_index, page in enumerate(doc):
        pix = page.get_pixmap()
        artifacts.append(
            ExtractArtifact(
                extract_type="image",
                artifact_kind="page_render",
                image_bytes=pix.tobytes("png"),
                mime_type="image/png",
                page_number=page_index + 1,
                metadata={"total_pages": total_pages},
            )
        )
        if on_progress:
            on_progress(page_index + 1, total_pages)

    return artifacts


def extract_document_pages(
    doc: fitz.Document,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ExtractArtifact], dict[str, int]]:
    \"\"\"Extract text, images, and tables from every page.\"\"\"
    artifacts: list[ExtractArtifact] = []
    stats = {"pdf_images_seen": 0, "pdf_tables_seen": 0, "pdf_document_extracts": 0}
    total_pages = len(doc)

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        text = (page.get_text() or "").strip()

        text_artifact = extract_page_text(page, page_number, total_pages)
        if text_artifact:
            artifacts.append(text_artifact)
            stats["pdf_document_extracts"] += 1

        image_artifacts = extract_page_images(doc, page, page_number, text)
        stats["pdf_images_seen"] += len(page.get_images(full=True) or [])
        artifacts.extend(image_artifacts)

        table_artifacts = extract_page_tables(page, page_number, text)
        stats["pdf_tables_seen"] += len(table_artifacts)
        artifacts.extend(table_artifacts)

        if on_progress:
            on_progress(page_number, total_pages)

    return artifacts, stats


print("Page-loop extraction helpers defined.")
"""
    )
)

cells.append(
    code(
        """
def load_pdf_comment_extractor():
    \"\"\"Load PDFCommentExtractor without importing the broken package __init__.\"\"\"
    for path in pkg_files("pdf-comments-extractor"):
        if str(path).endswith("comment_extractor.py"):
            spec = importlib.util.spec_from_file_location(
                "_pdf_comments_comment_extractor",
                path.locate() if hasattr(path, "locate") else str(path),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.PDFCommentExtractor
    raise ImportError("pdf-comments-extractor comment module not found")


def extract_annotations(pdf_path: Path | str) -> tuple[list[ExtractArtifact], dict[str, int]]:
    \"\"\"Extract PDF comment annotations (optional - skips if library unavailable).\"\"\"
    stats = {
        "pdf_annotations_seen": 0,
        "pdf_annotations_indexed": 0,
        "pdf_annotations_skipped_empty": 0,
    }
    artifacts: list[ExtractArtifact] = []

    try:
        PDFCommentExtractor = load_pdf_comment_extractor()
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        print("[Annotations] Library unavailable - skipping.")
        return artifacts, stats

    source_path = Path(pdf_path)
    if not source_path.exists():
        print("[Annotations] PDF file not found - skipping.")
        return artifacts, stats

    with PDFCommentExtractor(str(source_path)) as extractor:
        comments = extractor.extract_all_comments()

    if not isinstance(comments, list):
        return artifacts, stats

    for raw in comments:
        if not isinstance(raw, dict):
            continue
        stats["pdf_annotations_seen"] += 1
        page = raw.get("page")
        author = str(raw.get("author") or "").strip() or "Unknown"
        comment = str(raw.get("comment") or "").strip()
        if not comment:
            stats["pdf_annotations_skipped_empty"] += 1
            continue

        artifacts.append(
            ExtractArtifact(
                extract_type="document",
                artifact_kind="document",
                text=f"Page {page} | Author: {author} | Comment: {comment}",
                page_number=int(page) if page is not None else None,
                metadata={"author": author, "comment": comment},
            )
        )
        stats["pdf_annotations_indexed"] += 1

    return artifacts, stats


print("Annotation extractor defined.")
"""
    )
)

cells.append(
    md(
        """
## Step 5 - Wire the pipeline together

**Goal:** One orchestrator function that calls every step in order.

```text
run_pdf_extraction_pipeline(pdf_bytes, pdf_path)
  |-- open PDF
  |-- classify_pdf()
  |-- extract_presentation_pages()  OR  extract_document_pages()
  |-- extract_annotations()
  +-- return (artifacts, stats)
```

This is the pattern used in production: **small functions + one pipeline runner**.
"""
    )
)

cells.append(
    code(
        """
def run_pdf_extraction_pipeline(
    pdf_bytes: bytes,
    pdf_path: Path | str,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ExtractArtifact], dict[str, int]]:
    \"\"\"Run the full Extract pipeline: classify, extract, combine.\"\"\"
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        pdf_type = classify_pdf(doc)
        print(f"[Pipeline] PDF type: {pdf_type}")

        stats: dict[str, int] = {
            "pages_total": len(doc),
            "pdf_page_renders": 0,
            "pdf_images_seen": 0,
            "pdf_tables_seen": 0,
            "pdf_document_extracts": 0,
            "pdf_annotations_seen": 0,
            "pdf_annotations_indexed": 0,
            "pdf_annotations_skipped_empty": 0,
        }

        if pdf_type == "presentation":
            artifacts = extract_presentation_pages(doc, on_progress=on_progress)
            stats["pdf_page_renders"] = len(artifacts)
        else:
            artifacts, page_stats = extract_document_pages(doc, on_progress=on_progress)
            stats.update(page_stats)

        annotation_artifacts, annotation_stats = extract_annotations(pdf_path)
        stats.update(annotation_stats)

        return [*artifacts, *annotation_artifacts], stats
    finally:
        doc.close()


print("Pipeline runner defined: run_pdf_extraction_pipeline()")
"""
    )
)

cells.append(
    md(
        """
## Step 6 - Run the pipeline

Three lines: define progress callback, run pipeline, print stats.
"""
    )
)

cells.append(
    code(
        """
def show_progress(current: int, total: int) -> None:
    if current == total or current % max(total // 5, 1) == 0:
        print(f"  [Progress] page {current} / {total}")


print("[Pipeline] Starting extraction...")
artifacts, stats = run_pdf_extraction_pipeline(
    pdf_bytes,
    pdf_path,
    on_progress=show_progress,
)

print("\\n[Pipeline] Done!")
print("\\n--- Stats ---")
for key, value in stats.items():
    print(f"  {key}: {value}")
"""
    )
)

cells.append(
    md(
        """
## Step 7 - Inspect the results

Helper functions to explore what the pipeline produced.
"""
    )
)

cells.append(
    code(
        """
def artifacts_to_dataframe(artifacts: list[ExtractArtifact]) -> pd.DataFrame:
    \"\"\"Convert artifacts into a summary table for inspection.\"\"\"
    rows = []
    for i, artifact in enumerate(artifacts):
        rows.append(
            {
                "#": i,
                "kind": artifact.artifact_kind,
                "type": artifact.extract_type,
                "page": artifact.page_number,
                "chars": len(artifact.text) if artifact.text else None,
                "image_kb": (
                    round(len(artifact.image_bytes) / 1024, 1)
                    if artifact.image_bytes
                    else None
                ),
                "preview": artifact.summary(),
            }
        )
    return pd.DataFrame(rows)


def show_text_sample(artifacts: list[ExtractArtifact], max_chars: int = 1200) -> None:
    \"\"\"Print the first text artifact.\"\"\"
    text_artifacts = [a for a in artifacts if a.artifact_kind == "document" and a.text]
    if not text_artifacts:
        print("No text artifacts found.")
        return
    sample = text_artifacts[0]
    print(f"Page {sample.page_number} text ({len(sample.text)} chars):")
    print("-" * 60)
    print(sample.text[:max_chars])
    if len(sample.text) > max_chars:
        print("\\n... (truncated)")


def show_image_sample(artifacts: list[ExtractArtifact]) -> None:
    \"\"\"Display the first image/table/page_render artifact.\"\"\"
    image_artifacts = [
        a for a in artifacts
        if a.image_bytes and a.artifact_kind in {"image", "table", "page_render"}
    ]
    if not image_artifacts:
        print("No image artifacts found.")
        return
    sample = image_artifacts[0]
    print(f"Showing: kind={sample.artifact_kind}, page={sample.page_number}")
    display(Image.open(BytesIO(sample.image_bytes)))


# --- Run Step 7 ---
df = artifacts_to_dataframe(artifacts)
print(f"Total artifacts: {len(df)}")
print(f"Breakdown by kind:\\n{df['kind'].value_counts().to_string()}")
df.head(15)
"""
    )
)

cells.append(code("show_text_sample(artifacts)"))
cells.append(code("show_image_sample(artifacts)"))

cells.append(
    md(
        """
## Step 8 - Save text artifacts for Notebook 2

**Goal:** Hand off text artifacts to the next lecture notebook (chunking and embeddings).

Notebook 2 reads from `data/extracted/{SOURCE_ID}_text_artifacts.json`. The `SOURCE_ID` must match `ARXIV_ID` from the config cell above.
"""
    )
)

cells.append(
    code(
        """
import json


def save_text_artifacts(
    artifacts: list[ExtractArtifact],
    source_id: str,
) -> Path:
    \"\"\"Save page-level text artifacts as JSON for the transformation notebook.\"\"\"
    output_dir = get_project_root() / "data" / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_id}_text_artifacts.json"

    records = []
    for index, artifact in enumerate(artifacts):
        if artifact.artifact_kind != "document" or not artifact.text:
            continue
        records.append(
            {
                "artifact_index": index,
                "page_number": artifact.page_number,
                "text": artifact.text,
                "metadata": artifact.metadata,
            }
        )

    payload = {
        "source_id": source_id,
        "pdf_path": str(pdf_path),
        "record_count": len(records),
        "records": records,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Save] Wrote {len(records)} text records to {output_path}")
    return output_path


# --- Run Step 8 ---
saved_path = save_text_artifacts(artifacts, ARXIV_ID)
"""
    )
)

cells.append(
    md(
        """
---

## What happens next?

Open **Notebook 2** (`rag-chunking-embeddings-lecture.ipynb`) to run the **Transform** stage: clean text, chunk, and generate embeddings.

Extract is **stage 1** only. The rest of the RAG pipeline:

```mermaid
flowchart LR
    A[Text artifacts] --> B[Clean text]
    C[Image artifacts] --> D[Caption and OCR]
    B --> E[Chunk]
    D --> E
    E --> F[Embed]
    F --> G[Vector DB]
    G --> H[RAG at query time]
```

### Key takeaways

1. **Break extraction into functions** - one job per function, easy to test and teach.
2. **Use a pipeline runner** - `run_pdf_extraction_pipeline()` orchestrates the steps.
3. **Typed artifacts** - not one big string; each type gets different treatment later.
4. **Page numbers are preserved** - enables citations like "see Page 4".
5. **Classify first** - slide decks need a different extraction path.

### Try it yourself

- Change `ARXIV_ID` in the config cell and re-run from Step 1
- Add a `print()` inside `classify_pdf()` to see the ratios it computes
- Try a slide-deck PDF and watch the pipeline switch to `extract_presentation_pages()`

### Further reading

- [PyMuPDF documentation](https://pymupdf.readthedocs.io/)
- [arXiv API](https://info.arxiv.org/help/api/index.html)
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
    out = Path(__file__).resolve().parents[1] / "notebooks" / "rag-pdf-extraction-lecture.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"Wrote {out} with {len(cells)} cells")
