# rag-chatbot

A production-ready Retrieval-Augmented Generation (RAG) document chatbot using Streamlit, FAISS, and LLM APIs.

## Features

- **PDF Processing**: Native text extraction with OCR fallback for scanned documents
- **Vector Search**: FAISS HNSW index with semantic + lexical reranking
- **Multi-Backend LLM**: Groq → HuggingFace Inference API → Extractive fallback cascade
- **Thread-Safe**: Safe concurrent access to vector store and embeddings
- **Persistent Storage**: Save and load indexed documents across sessions
- **Source Citations**: Every answer includes page numbers and source files
- **Streamlit Cloud Ready**: One-click deployment with `packages.txt` + secrets

## Quick Start

### Prerequisites

- **Python**: 3.9 or higher
- **System Dependencies**: Tesseract OCR (for scanned PDF support)

### Local Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/hungrycoder09/rag-chatbot.git
cd rag-chatbot
```

#### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng libgl1-mesa-glx
```

**macOS (with Homebrew):**
```bash
brew install tesseract
```

**Windows (with Chocolatey):**
```bash
choco install tesseract
```

#### 4. (Optional) Configure LLM Backend

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys (optional — the app works without them):
```
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

Or set environment variables directly:
```bash
export GROQ_API_KEY="your-groq-key"
export HUGGINGFACE_API_TOKEN="your-hf-token"
streamlit run app.py
```

#### 5. Run the Application
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

## Usage

### Upload & Index Documents

1. **Upload PDFs** in the left sidebar under "📄 Add Documents"
2. Click **"🔄 Ingest Selected PDFs"** — the app will:
   - Extract text from each page
   - Run OCR on scanned pages if needed
   - Split into overlapping chunks
   - Embed with sentence-transformers
   - Add to FAISS index

3. Watch the ingestion progress — status shows "✅ {chunk_count} chunks ({page_count} pages)"

### Ask Questions

1. Type a question in the chat box at the bottom: *"What is mentioned about...?"*
2. The app will:
   - Search for relevant chunks (top-k retrieval)
   - Send context to the LLM
   - Return an answer with source citations

3. **Sources** are shown in an expandable "📌 Sources & Provenance" section with:
   - Document name and page number
   - Combined relevance score (semantic + lexical)
   - Snippet preview (first 300 chars)

### Retrieval Settings

In the sidebar under "🔧 Retrieval Settings":
- **Top-K chunks**: How many passages to retrieve (1–10)
  - Smaller = faster but less context
  - Larger = richer context but slower LLM calls

### Manage Indexed Documents

- View indexed documents in "📋 Indexed Documents"
- Expand any document to see its chunk count
- Click "🗑️ Remove from index" to delete and re-index
- Click "🗑️ Clear Chat History" to reset conversation

## Deployment

### Streamlit Community Cloud (Free, Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** and select your repo
4. Set **Main file path** to `app.py`
5. (Optional) Add secrets in **Settings → Secrets**:
   ```
   GROQ_API_KEY = "your-key-here"
   HUGGINGFACE_API_TOKEN = "your-token-here"
   ```

The app will auto-restart when you push updates to `main`.

### Local Docker Deployment

```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 8501:8501 \
  -e GROQ_API_KEY="your-key" \
  -e HUGGINGFACE_API_TOKEN="your-token" \
  rag-chatbot
```

## Configuration

### LLM Backends (in priority order)

1. **Groq** (Fastest, requires API key)
   - Model: `llama-3.1-8b-instant`
   - Max tokens: 1024
   - Get key: [console.groq.com/keys](https://console.groq.com/keys)

2. **HuggingFace Inference API** (Free, can cold-start)
   - Model: `mistralai/Mistral-7B-Instruct-v0.2`
   - Max tokens: 512
   - Get token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

3. **Extractive Fallback** (Always works, zero latency)
   - Returns top-scoring chunk verbatim
   - No API key needed

### Embedding Model

- **sentence-transformers/all-MiniLM-L6-v2**
- Dimension: 384
- Downloads on first use (~80 MB)

### Vector Store

- **FAISS HNSW** (Hierarchical Navigable Small World)
- Re-ranking factor: 3x candidates
- Persisted to `rag_store/` directory

### Chunking Strategy

- Chunk size: ~500 tokens (~2,000 characters)
- Overlap: 15% (300 characters)
- Min native text per page: 50 chars (triggers OCR fallback)

## Troubleshooting

### "ModuleNotFoundError: No module named X"

**Solution**: One or more dependencies failed to install. Reinstall:
```bash
pip install --upgrade -r requirements.txt
```

The app checks at startup and warns which modules are missing:
- `fitz` (PyMuPDF) — PDF parsing
- `pytesseract` — OCR fallback
- `faiss` — Vector search
- `sentence_transformers` — Embeddings

### "No module named 'tesseract_cmd'"

**Solution**: System Tesseract OCR not installed. On your OS:

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

Then on Windows or if installed in non-standard path, set:
```bash
export TESSDATA_PREFIX=/path/to/tessdata
```

### PDF Ingestion Shows "⚠️ No text extracted"

**Causes:**
- PDF is image-only (scanned) without OCR installed
- PDF is corrupted or encrypted

**Solutions:**
1. Install Tesseract OCR (see above)
2. Verify PDF is readable: `pdftotext filename.pdf`
3. Try re-uploading the PDF

### "Cannot add vectors: dimension mismatch"

**Solution**: This shouldn't happen with the default embedding model. If you modify `EMBEDDING_MODEL_NAME`, ensure:
```python
EMBEDDING_DIM = 384  # Match the new model's output dimension
```

Then delete the `rag_store/` directory and re-ingest:
```bash
rm -rf rag_store/
```

### App is slow / retrieval takes >10 seconds

**Causes:**
- Large number of chunks (1000+) in index
- HuggingFace model cold-starting
- Network latency to API

**Solutions:**
1. Use Groq (fastest): add `GROQ_API_KEY` to `.env`
2. Reduce `top_k` in sidebar
3. Clear unused documents from the index
4. Pre-warm HuggingFace by asking a question first

### "GROQ_API_KEY not recognized"

**Solution**: Ensure environment variable is set before starting the app:
```bash
export GROQ_API_KEY="your-key-here"
streamlit run app.py
```

Or add to `.env` and load with `python-dotenv`:
```bash
pip install python-dotenv
# Then the app will auto-load from .env
```

### Streamlit Cloud deployment fails

**Check:**
1. `requirements.txt` and `packages.txt` exist in repo root
2. `packages.txt` includes: `tesseract-ocr`, `tesseract-ocr-eng`, `libgl1-mesa-glx`
3. Secrets are added in **Settings → Secrets** (not `.env`)
4. Main file path is set to `app.py`

## Project Structure

```
rag-chatbot/
├── app.py                    # Single-file Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies (Streamlit Cloud)
├── .env.example              # Environment variable template
├── .streamlit/config.toml    # Streamlit configuration
└── README.md                 # This file
```

### app.py Sections

1. **SECTION 1 — INGESTION** (Lines 64–196)
   - PDF parsing (PyMuPDF + Tesseract OCR fallback)
   - Text cleaning and normalization
   - Sliding-window chunking with overlap

2. **SECTION 2 — VECTOR STORE** (Lines 212–421)
   - FAISS HNSW index (thread-safe, persistent)
   - Sentence-Transformers embeddings
   - Semantic + lexical re-ranking

3. **SECTION 3 — LLM SERVICE** (Lines 424–589)
   - Multi-backend cascade: Groq → HuggingFace → Extractive
   - RAG prompt construction with context passages
   - Citation enforcement

4. **SECTION 4 — STREAMLIT UI** (Lines 591–800)
   - Sidebar: upload, settings, indexed documents
   - Main panel: chat history, source provenance

## API Keys & Sign-ups

- **Groq**: [console.groq.com/keys](https://console.groq.com/keys) (free tier available)
- **HuggingFace**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free, no token needed for public models)

## Performance Notes

- **First run**: ~2–3 minutes (downloads embedding model, ~80 MB)
- **Subsequent runs**: <1 second startup
- **Ingestion**: ~5–10 pages/second on modern hardware
- **Retrieval**: <100ms with FAISS (Groq LLM: 1–3s, HuggingFace: 2–10s, Extractive: instant)

## License

MIT (or your preferred license)

## Contributing

Contributions welcome! Please ensure:
- Code follows the existing style
- All imports are in the try/except blocks (for graceful degradation)
- Changes are tested locally before pushing
