# bana275-rag
# RAG-Powered Medical Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built on the `sci.med` subset of the 20 Newsgroups dataset. The system answers medical-domain questions by retrieving relevant document chunks from a ChromaDB vector store and generating grounded responses via OpenAI's `gpt-4o-mini`, with conversation memory and cross-encoder reranking.

**Live Demo:** [Deployed via Gradio `share=True` on Google Colab]

---

## Features

- **Domain-specific RAG** over 100 `sci.med` newsgroup documents
- **Recursive text chunking** (chunk size 400, overlap 50) with email metadata preserved
- **ChromaDB vector store** with `all-MiniLM-L6-v2` sentence embeddings
- **Conversation memory** — rolling 10-turn window using a `deque` buffer
- **Follow-up handling** — LLM-based query rewriting for contextual follow-up questions
- **Cross-encoder reranking** — initial top-20 retrieval re-ranked to top-5 using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Source citations** — every answer includes document and chunk references
- **Gradio UI** — chat interface with clear button and per-session memory isolation

---

## Architecture

```
User Query
    │
    ▼
Query Rewriter (gpt-4o-mini)   ←── Conversation Memory (deque, maxlen=20)
    │
    ▼
ChromaDB Retrieval (top-20, all-MiniLM-L6-v2)
    │
    ▼
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) → top-5 chunks
    │
    ▼
Prompt Builder (context + history + query)
    │
    ▼
gpt-4o-mini → Answer + Citations
    │
    ▼
Gradio UI + Memory Update
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- OpenAI API key

### Install Dependencies

```bash
pip install langchain langchain-community chromadb sentence-transformers \
            google-generativeai tiktoken langchain-text-splitters openai gradio
```

### Configure API Key

In Google Colab, store your key as a secret named `OPENAI_API_KEY`. Locally:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"
```

### Run

Open `RAG_powered_chatbot.ipynb` in Google Colab and run all cells in order. The final cell launches the Gradio interface with a public share link.

---

## Project Structure

```
RAG_powered_chatbot.ipynb   # Main notebook (all code)
README.md                   # This file
chroma_db/                  # Persistent ChromaDB vector store (auto-created)
```

---

## Usage

Once the Gradio UI is running, type any medical-domain question. Example queries:

- *"Where can I find a comprehensive review of HIV treatments?"*
- *"What are common symptoms of chronic fatigue syndrome?"*
- *"Are you sure about that?"* ← follow-up query (automatically rewritten)

Each response includes cited sources in the format:
```
[Source 1] doc=42 chunk=2 subject=Re: AIDS treatment options
```

---

## Technical Details

### Document Processing

- **Dataset:** `sklearn.datasets.fetch_20newsgroups`, `sci.med` category, training split, first 100 documents
- **Chunking:** `RecursiveCharacterTextSplitter` with separators `['\n\n', '\n', '. ', ' ', '']`
- **Metadata stored:** `From`, `Subject`, `Organization`, `Lines`, `doc_id`, `chunk_idx`

### Retrieval

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector DB:** ChromaDB `PersistentClient`
- **Retrieval pipeline:** Semantic search (top-20) → CrossEncoder rerank (top-5)

### Generation

- **Model:** `gpt-4o-mini`, temperature=0
- **Prompt strategy:** System + retrieved context + conversation history + user query
- **Instruction:** Answer only from context; cite sources; acknowledge gaps

### Memory

- Stored as list of `{"role": ..., "content": ...}` dicts in a `deque(maxlen=20)`
- Per-session isolation via Gradio `gr.State`
- Query rewriting uses history to convert follow-ups into standalone retrieval queries

---

## Limitations

- Corpus is limited to 100 training documents from a 1990s newsgroup — not a medical authority
- `gpt-4.1-mini` used in initial API test cell; main pipeline uses `gpt-4o-mini`
- `google.generativeai` imported but unused (superseded by OpenAI client)
- No persistent memory across Colab sessions (ChromaDB reloads, but conversation history resets)

---

## License

For academic/educational use. Dataset from scikit-learn (BSD 3-Clause). Model weights from HuggingFace (Apache 2.0 / MIT).
