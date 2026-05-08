# Reddit RabbitHole RAG

Ask anything — get answered by real people who lived it.

A production-style RAG system built over 2,671 verified Reddit AMA (Ask Me Anything)
Q&A pairs from 451 threads. Ask a question, get an answer grounded in real human
experiences — attributed to the actual person who said it, with source links.

**Live demo:** [HuggingFace Spaces](https://huggingface.co/spaces/athxrvxaa/reddit-rabbithole-rag)

---

## What makes this non-trivial

Most RAG tutorials chunk a PDF and call it done. This project deals with messier problems:

**Thread-aware chunking**  
Reddit comments only make sense with their parent question. Standard text splitters
destroy this context. Custom OP detection logic identifies the AMA subject from thread
structure and extracts only their verified responses, filtering out bots, moderators,
and deleted accounts.

**Hybrid retrieval**  
Pure semantic search misses keyword-heavy queries. Pure BM25 misses semantic ones.
This system fuses both with weighted score combination (BM25 0.3 + dense 0.7),
handling the full spectrum of query types.

**Two-stage retrieval**  
Embedding similarity is fast but imprecise. A cross-encoder reranker
(ms-marco-MiniLM-L-6-v2) re-scores the top 10 candidates for precision before
anything reaches the LLM. Slower but significantly more accurate.

**Grounded generation**  
The LLM is explicitly constrained to retrieved context only. Every sentence in the
output is attributed to a real person by their AMA title. If the answer is not in
the corpus, the system says so rather than hallucinating.

---

## Evaluation

| Metric | Score |
|---|---|
| RAGAS Faithfulness | 0.81 |
| Corpus size | 2,671 Q&A pairs |
| AMA threads | 451 |

Faithfulness measures whether generated answers stay grounded in retrieved context.
Evaluated on a 20-question benchmark using Gemini 2.5 Flash as judge.

---

## Data Pipeline

- **Source:** r/IAmA via HuggingFaceGECLM/REDDIT_comments (25.8M comments streamed)
- **OP detection:** Author identified by top-level comment frequency + score in thread
- **Q&A extraction:** Parent-child comment matching, minimum length filters,
  question detection heuristics, score thresholds
- **Enrichment:** AMA post titles and metadata fetched via Arctic Shift API
  (open Reddit archive, no auth required)
- **Categories:** crime/law enforcement, gaming, identity, business, entertainment,
  education, healthcare, religion, politics

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector index | FAISS (IndexFlatIP, cosine similarity) |
| Keyword search | rank-bm25 (BM25Okapi) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generation | Gemini 2.5 Flash |
| Evaluation | RAGAS |
| UI | Gradio |
| Hosting | HuggingFace Spaces |

---

## Run locally

```bash
git clone https://github.com/athxrvxaa/reddit-rabbithole-rag
cd reddit-rabbithole-rag
pip install -r requirements.txt
```

Download the artifact files from HuggingFace Spaces and place them in the root directory:
- `ama_qa_enriched.jsonl`
- `embeddings.npy`
- `faiss_index.bin`
- `bm25_index.pkl`
- `doc_texts.json`

```bash
export GEMINI_API_KEY=your_key_here
python app.py
```

---
Large binary files (embeddings, FAISS index, BM25 index) are hosted on
HuggingFace Spaces and excluded from this repository via .gitignore.