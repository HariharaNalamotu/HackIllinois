"""
Agentic chunker — uses OpenAI to inspect a sample of your input file and
decide which chunking strategy is most appropriate, then applies it.

Supported input formats:
  .txt  .md  .pdf  .html  .htm  .docx  .csv  .json  .jsonl

Usage:
  python chunk.py <path_to_file> [--out chunks.json]

Output:
  JSON file containing a list of text chunk strings.
"""

import os
import sys
import re
import json
import csv
import textwrap
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import openai

# ── optional deps (graceful errors if missing) ──────────────────────────────
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    NLTK_OK = True
except ImportError:
    NLTK_OK = False

try:
    from pypdf import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False


client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SAMPLE_CHARS = 3000   # characters sent to OpenAI for inspection


# ============================================================================
# FILE READERS
# ============================================================================

def read_file(path: Path) -> str:
    """Return the raw text content of any supported file."""
    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        if not PDF_OK:
            raise ImportError("pypdf not installed — run: pip install pypdf")
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext in (".html", ".htm"):
        if not BS4_OK:
            raise ImportError("beautifulsoup4 not installed — run: pip install beautifulsoup4")
        soup = BeautifulSoup(path.read_bytes(), "lxml")
        return soup.get_text(separator="\n")

    if ext == ".docx":
        if not DOCX_OK:
            raise ImportError("python-docx not installed — run: pip install python-docx")
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    if ext == ".csv":
        rows = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(rows)

    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2)

    if ext == ".jsonl":
        lines = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.dumps(json.loads(line)))
        return "\n".join(lines)

    # Fallback — try plain text
    return path.read_text(encoding="utf-8", errors="replace")


# ============================================================================
# CHUNKING METHODS
# ============================================================================

def chunk_by_sentence(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """Split on sentence boundaries, group N sentences per chunk."""
    if NLTK_OK:
        sentences = sent_tokenize(text)
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_by_paragraph(text: str, min_words: int = 10) -> list[str]:
    """Split on blank lines (paragraph breaks)."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if p and len(p.split()) >= min_words:
            chunks.append(p)
    return chunks


def chunk_sliding_window(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Overlapping fixed-size word windows."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_fixed_size(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    """Non-overlapping fixed word-count chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap if overlap else chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_by_markdown_headers(text: str) -> list[str]:
    """
    Split on Markdown headers (# / ## / ###).
    Each chunk = header + its body.
    """
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    chunks = [s.strip() for s in sections if s.strip()]
    return chunks


def chunk_by_html_sections(text: str) -> list[str]:
    """
    For HTML-derived text: split on section-like breaks
    (h1–h4 remnants, horizontal rules, etc.).
    """
    parts = re.split(r"\n(?=[A-Z][^\n]{0,80}\n[-=]{3,})", text)
    if len(parts) == 1:
        parts = re.split(r"\n{3,}", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    LangChain-style recursive splitting.
    Tries separators in order: paragraph → newline → sentence → space.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: list[str]) -> list[str]:
        if not seps or len(t) <= chunk_size:
            return [t] if t.strip() else []
        sep = seps[0]
        parts = t.split(sep)
        result, current = [], ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size:
                    result.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part.strip()
        if current:
            result.append(current)
        # Apply overlap
        if overlap and len(result) > 1:
            overlapped = [result[0]]
            for i in range(1, len(result)):
                tail_words = overlapped[-1].split()[-overlap:]
                overlapped.append(" ".join(tail_words) + " " + result[i])
            return overlapped
        return result

    return _split(text, separators)


def chunk_by_code_blocks(text: str) -> list[str]:
    """
    For source code or mixed code+prose: split on function/class definitions
    and markdown code fences.
    """
    # Markdown fenced code blocks
    fences = re.split(r"```[\w]*\n", text)
    if len(fences) > 1:
        return [f.strip() for f in fences if f.strip()]
    # Python / JS / TS function/class boundaries
    parts = re.split(r"(?=\n(?:def |class |function |const |async function ))", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_by_csv_rows(text: str, rows_per_chunk: int = 10) -> list[str]:
    """Group CSV rows into chunks of N rows each (preserving the header)."""
    lines = text.strip().splitlines()
    if not lines:
        return []
    header = lines[0]
    data_lines = lines[1:]
    chunks = []
    for i in range(0, len(data_lines), rows_per_chunk):
        group = data_lines[i : i + rows_per_chunk]
        chunks.append(header + "\n" + "\n".join(group))
    return chunks


def chunk_by_json_objects(text: str) -> list[str]:
    """Each top-level JSON object or array element becomes its own chunk."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [json.dumps(item, ensure_ascii=False) for item in data]
        if isinstance(data, dict):
            return [json.dumps({k: v}, ensure_ascii=False) for k, v in data.items()]
    except json.JSONDecodeError:
        # Might be JSONL
        chunks = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    chunks.append(json.dumps(json.loads(line), ensure_ascii=False))
                except json.JSONDecodeError:
                    chunks.append(line)
        return chunks
    return [text]


def chunk_by_page(text: str, page_sep: str = "\x0c") -> list[str]:
    """Split on form-feed characters (PDF page breaks)."""
    pages = text.split(page_sep)
    return [p.strip() for p in pages if p.strip()]


# ── registry ────────────────────────────────────────────────────────────────
CHUNKERS = {
    "sentence":          chunk_by_sentence,
    "paragraph":         chunk_by_paragraph,
    "sliding_window":    chunk_sliding_window,
    "fixed_size":        chunk_fixed_size,
    "markdown_headers":  chunk_by_markdown_headers,
    "html_sections":     chunk_by_html_sections,
    "recursive":         chunk_recursive,
    "code_blocks":       chunk_by_code_blocks,
    "csv_rows":          chunk_by_csv_rows,
    "json_objects":      chunk_by_json_objects,
    "page":              chunk_by_page,
}


# ============================================================================
# OPENAI AGENT — decides which chunker to use
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sentence",
            "description": (
                "Split text into groups of sentences using NLTK sentence tokenisation. "
                "Best for: conversational text, Q&A pairs, short factual statements, "
                "interview transcripts, or any content where individual sentences are "
                "self-contained units of meaning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sentences_per_chunk": {
                        "type": "integer",
                        "description": "How many sentences to group per chunk (1–5). Use 1 for dense technical text, 2–3 for prose.",
                        "default": 2,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "paragraph",
            "description": (
                "Split on blank lines (double newlines). "
                "Best for: blog posts, articles, essays, documentation, README files, "
                "or any text naturally organised into paragraphs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_words": {
                        "type": "integer",
                        "description": "Minimum word count to keep a paragraph (default 10).",
                        "default": 10,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sliding_window",
            "description": (
                "Overlapping fixed-size word windows. "
                "Best for: long dense documents where context must bleed across chunk boundaries "
                "(legal contracts, scientific papers, technical manuals). "
                "Overlap prevents losing meaning at split points."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_size": {"type": "integer", "description": "Words per chunk.", "default": 100},
                    "overlap":    {"type": "integer", "description": "Overlapping words between consecutive chunks.", "default": 20},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fixed_size",
            "description": (
                "Non-overlapping fixed word-count chunks with no regard for sentence boundaries. "
                "Best for: homogeneous corpora where you need uniform chunk sizes, "
                "or when content has no natural structure (OCR output, raw logs)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_size": {"type": "integer", "description": "Words per chunk.", "default": 200},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "markdown_headers",
            "description": (
                "Split on Markdown headers (#, ##, ###). Each section header + its body = one chunk. "
                "Best for: .md files, wikis, documentation sites, Notion exports, GitHub READMEs."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "html_sections",
            "description": (
                "Split HTML-derived plain text on heading-like visual breaks. "
                "Best for: web-scraped content, HTML exports, email newsletters."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recursive",
            "description": (
                "Recursive character-based splitting (LangChain-style). "
                "Tries paragraph → newline → sentence → space until chunks fit the size limit. "
                "Best for: mixed-format text where structure is inconsistent — "
                "PDFs with columns, scraped pages, forum posts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_size": {"type": "integer", "description": "Max characters per chunk.", "default": 500},
                    "overlap":    {"type": "integer", "description": "Character overlap between chunks.", "default": 50},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_blocks",
            "description": (
                "Split on function/class definitions or Markdown code fences. "
                "Best for: source code files (.py, .js, .ts), Jupyter notebooks exported to text, "
                "or documentation that mixes prose with code snippets."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "csv_rows",
            "description": (
                "Group CSV rows into chunks of N rows, preserving the header in each chunk. "
                "Best for: .csv files, tabular data, spreadsheet exports."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rows_per_chunk": {"type": "integer", "description": "Number of data rows per chunk.", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "json_objects",
            "description": (
                "Each top-level JSON object or array element becomes one chunk. "
                "Best for: .json arrays of records, .jsonl files, API response dumps."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "page",
            "description": (
                "Split on PDF page-break characters (form-feed \\x0c). "
                "Best for: PDFs where each page is a self-contained unit "
                "(slide decks, forms, scanned documents)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def agent_choose_chunker(file_path: Path, sample: str) -> tuple[str, dict]:
    """
    Send a sample of the file to OpenAI and let it call one of the chunking
    tools. Returns (method_name, kwargs).
    """
    system_prompt = textwrap.dedent("""
        You are a text-processing expert. You will be shown the file name and a
        sample of its content. Your job is to call exactly one of the provided
        chunking tools — the one best suited to the structure and format of the
        content. Choose parameters that will produce clean, semantically coherent
        chunks of roughly 50–200 words each. Do not explain yourself; just call
        the tool.
    """).strip()

    user_message = (
        f"File: {file_path.name}\n"
        f"Extension: {file_path.suffix}\n\n"
        f"--- CONTENT SAMPLE ---\n{sample}\n--- END SAMPLE ---"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        tools=TOOLS,
        tool_choice="required",
    )

    tool_call = response.choices[0].message.tool_calls[0]
    method    = tool_call.function.name
    kwargs    = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
    return method, kwargs


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="Path to the input file")
    parser.add_argument("--out", default="chunks.json", help="Output JSON file (default: chunks.json)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found — {path}")
        sys.exit(1)

    print(f"Reading '{path}' ...")
    text = read_file(path)
    print(f"  {len(text):,} characters extracted")

    sample = text[:SAMPLE_CHARS]
    print(f"\nAsking OpenAI which chunking method to use (sample: {len(sample)} chars) ...")
    method, kwargs = agent_choose_chunker(path, sample)

    print(f"  → OpenAI chose: '{method}' with params {kwargs}")

    chunker = CHUNKERS[method]
    # Pass only the kwargs the function actually accepts
    import inspect
    sig = inspect.signature(chunker)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    chunks = chunker(text, **valid_kwargs)

    # Filter empty / too-short chunks
    chunks = [c.strip() for c in chunks if len(c.strip().split()) >= 5]

    out_path = Path(args.out)
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone — {len(chunks)} chunks saved to '{out_path}'")
    print(f"  Avg chunk length: {sum(len(c.split()) for c in chunks) // max(len(chunks), 1)} words")

    # Preview first 2 chunks
    print("\n--- Preview (first 2 chunks) ---")
    for i, chunk in enumerate(chunks[:2], 1):
        preview = chunk[:200] + ("..." if len(chunk) > 200 else "")
        print(f"\n[{i}] {preview}")


if __name__ == "__main__":
    main()
