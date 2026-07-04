"""
RAG Document Chatbot — Single-File Version
===========================================
All modules combined into one file for easy GitHub deployment.

Deploy to Streamlit Community Cloud:
  1. Push this file + requirements.txt + packages.txt + .streamlit/config.toml to GitHub
  2. Go to share.streamlit.io → New app → set Main file path: app.py
  3. (Optional) Add GROQ_API_KEY in Settings → Secrets for fast generative answers

Run locally:
  pip install -r requirements.txt
  streamlit run app.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import streamlit as st

# ── PyMuPDF ───────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ── OCR fallback ──────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── FAISS ─────────────────────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ── Sentence Transformers ─────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INGESTION
# PyMuPDF parsing, OCR fallback, text cleaning, sliding-window chunker
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum native characters per page before we trigger OCR
MIN_NATIVE_CHARS: int = 50

# Chunk size in approximate tokens (1 token ≈ 4 chars)
CHUNK_SIZE_TOKENS: int = 500
OVERLAP_FRACTION: float = 0.15
CHARS_PER_TOKEN: int = 4
CHUNK_SIZE_CHARS: int = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN   # 2000
OVERLAP_CHARS: int = int(CHUNK_SIZE_CHARS * OVERLAP_FRACTION)  # 300


def _clean_text(raw: str) -> str:
    """Normalise extracted text: fix whitespace, strip lone page numbers."""
    text = raw.replace("\xa0", " ").replace("\t", " ")
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"[ivxlcdmIVXLCDM\d]{1,6}", stripped):
            continue  # skip page-number-only lines
        cleaned.append(stripped)
    text = " ".join(cleaned)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _ocr_page(page: Any) -> str:
    """Render a PDF page to image and run Tesseract OCR on it."""
    if not OCR_AVAILABLE or not PYMUPDF_AVAILABLE:
        return ""
    try:
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    except Exception:
        return ""


def _parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from every page of a PDF with OCR fallback for scanned pages.
    Returns list of {filename, page_number, text} dicts.
    """
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF (fitz) is required. Run: pip install pymupdf")

    filename = Path(pdf_path).name
    pages: List[Dict[str, Any]] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open PDF '{pdf_path}': {exc}") from exc

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = _clean_text(page.get_text("text"))

        # OCR fallback for scanned / image-only pages
        if len(text) < MIN_NATIVE_CHARS:
            ocr = _ocr_page(page)
            if ocr:
                text = _clean_text(ocr)

        if not text:
            continue

        pages.append({
            "filename": filename,
            "page_number": page_num + 1,
            "text": text,
        })

    doc.close()
    return pages


def _chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sliding-window chunker: splits page text into fixed-size chunks with overlap.
    Each chunk inherits the page number where it started.
    """
    if not pages:
        return []

    chunks: List[Dict[str, Any]] = []
    stride = CHUNK_SIZE_CHARS - OVERLAP_CHARS
    filename = pages[0]["filename"]

    # Build a flat text buffer with character-offset → page-number map
    flat_text = ""
    offset_to_page: List[Tuple[int, int]] = []
    for page in pages:
        offset_to_page.append((len(flat_text), page["page_number"]))
        flat_text += page["text"] + "\n\n"

    def page_at_offset(offset: int) -> int:
        pg = offset_to_page[0][1]
        for start, pg_num in offset_to_page:
            if start > offset:
                break
            pg = pg_num
        return pg

    chunk_index = 0
    pos = 0
    total_len = len(flat_text)

    while pos < total_len:
        end = min(pos + CHUNK_SIZE_CHARS, total_len)
        chunk_text = flat_text[pos:end].strip()
        if len(chunk_text) > 20:
            chunks.append({
                "filename": filename,
                "page_number": page_at_offset(pos),
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            chunk_index += 1
        pos += stride

    return chunks


def ingest_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Full ingestion pipeline: parse → clean → chunk a single PDF."""
    pages = _parse_pdf(pdf_path)
    return _chunk_pages(pages) if pages else []


def get_pdf_page_count(pdf_path: str) -> int:
    """Return page count without full ingestion."""
    if not PYMUPDF_AVAILABLE:
        return 0
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VECTOR STORE
# FAISS HNSW index, sentence-transformers embeddings, thread-safe, disk-persistent
# ═══════════════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 200
HNSW_EF_SEARCH: int = 100
RERANK_FACTOR: int = 3

STORE_DIR: Path = Path("rag_store")
INDEX_PATH: Path = STORE_DIR / "faiss.index"
META_PATH: Path = STORE_DIR / "metadata.json"

# Module-level singletons — protected by _lock
_vs_lock = threading.RLock()
_model: Optional[Any] = None
_index: Optional[Any] = None
_metadata: List[Dict[str, Any]] = []
_doc_ids: Dict[str, List[int]] = {}


def _assert_vs_deps() -> None:
    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss-cpu is required. Run: pip install faiss-cpu")
    if not ST_AVAILABLE:
        raise RuntimeError("sentence-transformers is required. Run: pip install sentence-transformers")


def _get_model() -> Any:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _build_faiss_index() -> Any:
    _assert_vs_deps()
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    return index


def load_store() -> bool:
    """Load persisted FAISS index and metadata from disk. Returns True if found."""
    global _index, _metadata, _doc_ids
    _assert_vs_deps()
    with _vs_lock:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            return False
        try:
            _index = faiss.read_index(str(INDEX_PATH))
            _index.hnsw.efSearch = HNSW_EF_SEARCH
            with open(META_PATH, "r", encoding="utf-8") as f:
                store = json.load(f)
            _metadata = store.get("metadata", [])
            _doc_ids = store.get("doc_ids", {})
            return True
        except Exception:
            _index = None
            _metadata = []
            _doc_ids = {}
            return False


def _save_store() -> None:
    """Persist the current index and metadata to disk."""
    global _index, _metadata, _doc_ids
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with _vs_lock:
        if _index is not None:
            faiss.write_index(_index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"metadata": _metadata, "doc_ids": _doc_ids}, f)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings into L2-normalised float32 vectors."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Embed and insert chunks into FAISS. Re-ingesting a file replaces it."""
    global _index, _metadata, _doc_ids
    if not chunks:
        return 0

    filename = chunks[0]["filename"]

    with _vs_lock:
        if _index is None:
            _index = _build_faiss_index()

        _remove_document_unsafe(filename)

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        start_idx = len(_metadata)
        _doc_ids[filename] = list(range(start_idx, start_idx + len(chunks)))
        _index.add(embeddings)

        for chunk in chunks:
            _metadata.append({
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk["text"],
            })

    _save_store()
    return len(chunks)


def _remove_document_unsafe(filename: str) -> None:
    """Remove all vectors for a document. Must be called with _vs_lock held."""
    global _index, _metadata, _doc_ids
    if filename not in _doc_ids:
        return

    surviving = [(i, m) for i, m in enumerate(_metadata) if m["filename"] != filename]
    if not surviving:
        _index = _build_faiss_index()
        _metadata = []
        _doc_ids = {}
        return

    _, keep_meta = zip(*surviving)
    embeddings = embed_texts([m["text"] for m in keep_meta])

    new_index = _build_faiss_index()
    new_index.add(embeddings)
    _metadata = list(keep_meta)
    _index = new_index

    _doc_ids = {}
    for pos, meta in enumerate(_metadata):
        _doc_ids.setdefault(meta["filename"], []).append(pos)
    _doc_ids.pop(filename, None)


def delete_document(filename: str) -> bool:
    """Remove all vectors for a document and persist. Returns True if found."""
    global _index, _metadata, _doc_ids
    with _vs_lock:
        if filename not in _doc_ids:
            return False
        _remove_document_unsafe(filename)
    _save_store()
    return True


def similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval: HNSW ANN (semantic) → lexical re-ranking.
    Returns top_k chunks with semantic_score, lexical_score, and combined score.
    """
    global _index, _metadata

    with _vs_lock:
        if _index is None or _index.ntotal == 0:
            return []

        n_candidates = min(top_k * RERANK_FACTOR, _index.ntotal)
        query_emb = embed_texts([query])
        distances, indices = _index.search(query_emb, n_candidates)

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(_metadata):
                continue
            sem_score = float(np.clip(dist, 0.0, 1.0))
            candidates.append({**_metadata[idx], "semantic_score": sem_score})

        query_tokens = {w.lower() for w in query.split() if len(w) >= 3}
        for c in candidates:
            if query_tokens:
                chunk_tokens = {w.lower() for w in c["text"].split() if len(w) >= 3}
                overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            else:
                overlap = 0.0
            c["lexical_score"] = float(overlap)
            c["score"] = 0.7 * c["semantic_score"] + 0.3 * overlap

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


def get_indexed_documents() -> Dict[str, int]:
    """Return {filename: chunk_count} for all indexed documents."""
    with _vs_lock:
        return {fn: len(positions) for fn, positions in _doc_ids.items()}


def get_total_vectors() -> int:
    """Return total number of vectors in the index."""
    with _vs_lock:
        return _index.ntotal if _index is not None else 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LLM SERVICE
# Groq → HuggingFace Inference API → Extractive fallback cascade
# ═══════════════════════════════════════════════════════════════════════════════

GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS: int = 1024
GROQ_TEMPERATURE: float = 0.1

HF_API_URL_TEMPLATE: str = "https://api-inference.huggingface.co/models/{model}"
HF_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
HF_MAX_NEW_TOKENS: int = 512
HF_TEMPERATURE: float = 0.1
HF_TIMEOUT: int = 30

MAX_CONTEXT_CHUNKS: int = 4
MAX_CONTEXT_CHARS: int = 3000


def _build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Construct the RAG prompt with numbered context passages."""
    context_parts = []
    total_chars = 0
    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):
        passage = (
            f"[{i}] SOURCE: {chunk['filename']} (Page {chunk['page_number']})\n"
            f"{chunk['text'][:MAX_CONTEXT_CHARS // MAX_CONTEXT_CHUNKS]}"
        )
        total_chars += len(passage)
        if total_chars > MAX_CONTEXT_CHARS:
            break
        context_parts.append(passage)

    return (
        "You are a precise document assistant. Answer the question using ONLY "
        "the provided context passages. When you cite information, include the "
        "source filename and page number using the format [filename, p.N]. "
        "If the context does not contain enough information, say exactly: "
        "'The provided documents do not contain enough information to answer this question.'\n\n"
        "--- CONTEXT ---\n"
        f"{chr(10).join(context_parts)}\n\n"
        "--- QUESTION ---\n"
        f"{query}\n\n"
        "--- ANSWER ---\n"
    )


def _call_groq(prompt: str) -> Optional[str]:
    """Call Groq chat completions API. Returns None if key missing or call fails."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": GROQ_MAX_TOKENS,
                "temperature": GROQ_TEMPERATURE,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _call_huggingface(prompt: str) -> Optional[str]:
    """Call HuggingFace Inference API (free, may cold-start). Returns None on failure."""
    hf_token = os.environ.get("HUGGINGFACE_API_TOKEN", "").strip()
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        resp = requests.post(
            HF_API_URL_TEMPLATE.format(model=HF_MODEL),
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": HF_MAX_NEW_TOKENS,
                    "temperature": HF_TEMPERATURE,
                    "return_full_text": False,
                },
            },
            timeout=HF_TIMEOUT,
        )
        if resp.status_code == 503:  # model loading
            return None
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        return None
    except Exception:
        return None


def _extractive_fallback(chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Return the top-scoring chunk verbatim — always works, zero network."""
    if not chunks:
        return "No relevant content was found in the indexed documents.", "extractive"
    top = chunks[0]
    answer = (
        f"Based on **{top['filename']}** (Page {top['page_number']}):\n\n"
        f"{top['text'][:800]}"
    )
    return answer, "extractive"


def _enforce_citations(answer: str, chunks: List[Dict[str, Any]]) -> str:
    """Append a Sources block if the LLM answer contains no page/file citations."""
    has_citation = bool(
        re.search(r"\.pdf", answer, re.IGNORECASE)
        or re.search(r"\bp\.?\s*\d+", answer, re.IGNORECASE)
        or re.search(r"page\s+\d+", answer, re.IGNORECASE)
        or re.search(r"\[\d+\]", answer)
    )
    if has_citation or not chunks:
        return answer
    seen: set = set()
    refs = []
    for chunk in chunks[:MAX_CONTEXT_CHUNKS]:
        key = (chunk["filename"], chunk["page_number"])
        if key not in seen:
            refs.append(f"- {chunk['filename']}, p. {chunk['page_number']}")
            seen.add(key)
    return f"{answer}\n\n**Sources:**\n" + "\n".join(refs)


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate an answer using the tiered backend cascade:
    Groq → HuggingFace → Extractive fallback.
    Returns dict with answer, backend, latency, prompt.
    """
    prompt = _build_prompt(query, chunks)
    start = time.perf_counter()

    answer = _call_groq(prompt)
    if answer:
        return {"answer": _enforce_citations(answer, chunks), "backend": "groq",
                "latency": time.perf_counter() - start, "prompt": prompt}

    answer = _call_huggingface(prompt)
    if answer:
        return {"answer": _enforce_citations(answer, chunks), "backend": "huggingface",
                "latency": time.perf_counter() - start, "prompt": prompt}

    answer, backend = _extractive_fallback(chunks)
    return {"answer": answer, "backend": backend,
            "latency": time.perf_counter() - start, "prompt": prompt}


def get_active_backend() -> str:
    """Return a label for the highest-priority available backend."""
    if os.environ.get("GROQ_API_KEY", "").strip():
        return f"Groq ({GROQ_MODEL})"
    if os.environ.get("HUGGINGFACE_API_TOKEN", "").strip():
        return f"HuggingFace ({HF_MODEL}) — authenticated"
    return f"HuggingFace ({HF_MODEL}) — public / Extractive fallback"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store_loaded" not in st.session_state:
    st.session_state.store_loaded = load_store()
if "ingestion_status" not in st.session_state:
    st.session_state.ingestion_status = {}

UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Chatbot")
    st.caption("Retrieval-Augmented Generation over your PDF library")
    st.divider()

    # System status
    st.subheader("⚙️ System Status")
    total_vectors = get_total_vectors()
    indexed_docs = get_indexed_documents()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Indexed Docs", len(indexed_docs))
    with col2:
        st.metric("Total Chunks", total_vectors)
    st.info(f"**LLM Backend:** {get_active_backend()}")
    st.divider()

    # PDF Upload
    st.subheader("📄 Add Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs. Click 'Ingest' after uploading.",
    )

    if uploaded_files:
        saved_files = []
        for uf in uploaded_files:
            # Sanitize filename — strip path components, replace unsafe chars
            safe_name = Path(uf.name).name
            safe_name = re.sub(r"[^\w.\- ]", "_", safe_name).strip() or "upload.pdf"

            dest = (UPLOAD_DIR / safe_name).resolve()
            if not str(dest).startswith(str(UPLOAD_DIR.resolve())):
                st.warning(f"⚠️ {uf.name} has an unsafe filename — skipped.")
                continue

            uf.seek(0)
            file_bytes = uf.read()
            if not file_bytes:
                st.warning(f"⚠️ {uf.name} appears to be empty — skipped.")
                continue

            dest.write_bytes(file_bytes)
            saved_files.append(safe_name)

        if st.button("🔄 Ingest Selected PDFs", use_container_width=True, type="primary"):
            for filename in saved_files:
                pdf_path = UPLOAD_DIR / filename
                with st.status(f"Ingesting {filename}…", expanded=False) as status_widget:
                    try:
                        st.write("Parsing pages…")
                        page_count = get_pdf_page_count(str(pdf_path))
                        st.write(f"Found {page_count} pages. Chunking…")
                        chunks = ingest_pdf(str(pdf_path))
                        n_chunks = len(chunks)

                        if n_chunks == 0:
                            st.session_state.ingestion_status[filename] = "⚠️ No text extracted"
                            status_widget.update(label=f"{filename}: no text", state="error")
                            continue

                        st.write(f"Embedding {n_chunks} chunks…")
                        inserted = upsert_chunks(chunks)
                        st.session_state.ingestion_status[filename] = f"✅ {inserted} chunks ({page_count} pages)"
                        status_widget.update(label=f"{filename}: done ({inserted} chunks)", state="complete")
                    except Exception as exc:
                        st.session_state.ingestion_status[filename] = f"❌ Error: {exc}"
                        status_widget.update(label=f"{filename}: failed", state="error")
            st.rerun()

    st.divider()

    # Indexed documents list
    st.subheader("📋 Indexed Documents")
    indexed_docs_live = get_indexed_documents()
    if not indexed_docs_live:
        st.caption("No documents indexed yet. Upload PDFs above to get started.")
    else:
        for fname, chunk_count in sorted(indexed_docs_live.items()):
            status_str = st.session_state.ingestion_status.get(fname, f"✅ {chunk_count} chunks")
            with st.expander(f"📄 {fname}", expanded=False):
                st.caption(status_str)
                if st.button("🗑️ Remove from index", key=f"del_{fname}"):
                    delete_document(fname)
                    st.session_state.ingestion_status.pop(fname, None)
                    st.rerun()

    st.divider()

    # Settings
    st.subheader("🔧 Retrieval Settings")
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=5,
                       help="More chunks = richer context but slower LLM calls.")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("📚 Document Q&A — RAG Chatbot")
st.caption("Ask questions about your uploaded PDFs. Every answer includes source citations.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _total = msg.get("total_latency", msg.get("latency", 0.0))
            _backend = msg.get("backend", "unknown")
            color = "green" if _total <= 5 else "orange"
            st.markdown(f"⏱️ **Response time:** :{color}[{_total:.2f}s] | 🤖 **Backend:** {_backend}")
            with st.expander("📌 Sources & Provenance", expanded=False):
                for i, chunk in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"**[{i}] {chunk['filename']}** — Page {chunk['page_number']} | "
                        f"Combined: `{int(chunk.get('score', 0)*100)}%` "
                        f"(Semantic: `{int(chunk.get('semantic_score', 0)*100)}%`, "
                        f"Lexical: `{int(chunk.get('lexical_score', 0)*100)}%`)"
                    )
                    st.text(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))
                    if i < len(msg["sources"]):
                        st.divider()

# Query input
has_docs = get_total_vectors() > 0
if not has_docs:
    st.warning("📂 No documents indexed yet. Use the sidebar to upload and ingest PDFs.")

query = st.chat_input("Ask a question about your documents…", disabled=not has_docs)

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    wall_start = time.perf_counter()

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            retrieval_start = time.perf_counter()
            sources = similarity_search(query, top_k=top_k)
            retrieval_time = time.perf_counter() - retrieval_start

            if not sources:
                answer = ("I could not find relevant content in the indexed documents. "
                          "Try rephrasing your question or uploading more PDFs.")
                backend = "none"
                gen_latency = 0.0
            else:
                result = generate_answer(query, sources)
                answer = result["answer"]
                backend = result["backend"]
                gen_latency = result["latency"]

        total_latency = time.perf_counter() - wall_start
        st.markdown(answer)

        if sources:
            color = "green" if total_latency <= 5 else "orange"
            st.markdown(
                f"⏱️ **Response time:** :{color}[{total_latency:.2f}s] "
                f"(retrieval: {retrieval_time:.2f}s, generation: {gen_latency:.2f}s) "
                f"| 🤖 **Backend:** {backend}"
            )
            with st.expander("📌 Sources & Provenance", expanded=True):
                for i, chunk in enumerate(sources, start=1):
                    st.markdown(
                        f"**[{i}] {chunk['filename']}** — Page {chunk['page_number']} | "
                        f"Combined: `{int(chunk.get('score', 0)*100)}%` "
                        f"(Semantic: `{int(chunk.get('semantic_score', 0)*100)}%`, "
                        f"Lexical: `{int(chunk.get('lexical_score', 0)*100)}%`)"
                    )
                    st.text(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))
                    if i < len(sources):
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "backend": backend,
        "latency": gen_latency,
        "total_latency": total_latency,
    })
