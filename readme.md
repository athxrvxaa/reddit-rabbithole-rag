# Reddit AMA Rabbit Hole

A retrieval augmented generation demo over historical r/IAmA comments. It retrieves personal accounts and links back to the source thread. AMA posts and comments are user generated and are not independently verified. The active local corpus currently contains 3,639 pairs across 650 threads, collected from multiple segments of the Hugging Face dataset. It reflects Reddit's r/IAmA participants and has substantial historical, demographic, topic, and selection bias.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python build_index.py
$env:GEMINI_API_KEY = "your-key"
python app.py
```

The app creates cleaned records and stores normalized embeddings (`embeddings.npy`), a FAISS index (`faiss.index`), and a SQLite BM25 postings index (`bm25.sqlite3`) in `.cache/rag/`. Startup reuses the cache while the corpus fingerprint and embedding model name/revision match; it rebuilds when either changes or a cache artifact is missing/corrupt. Set `RAG_CACHE_DIR` to use a different cache location. No Python pickle is read or written. The first run downloads the embedding and reranker models and builds the indexes; later runs load them from disk. Put `GEMINI_API_KEY=your-key` in the local `.env` file; it is ignored by Git. The key is needed only for generated answers; retrieval can still run without it.

## Data pipeline

`ingest.py` streams the `HuggingFaceGECLM/REDDIT_comments` IAmA split, groups comments by thread, infers the OP from repeated top-level participation, and keeps OP replies to a different author's direct parent comment. It then fetches post titles/context from Arctic Shift and writes a cleaned JSONL corpus. Install the optional ingestion dependency with `pip install -r requirements-ingest.txt`. Use `--offset` to process a later segment and merge with the current corpus into a separate candidate file:

```bash
python ingest.py --offset 1000000 --max-comments 500000 --min-answer-score 0 \
  --base-corpus data/ama_qa_enriched.jsonl \
  --output data/ama_qa_enriched_next.jsonl
```

Streaming is not a complete or uniformly sampled Reddit archive. Metadata lookups can fail and omit some threads. Inspect the output and extraction assumptions before publishing a regenerated corpus. The previous local corpus is backed up at `data/ama_qa_enriched_before_hf.jsonl`.

`qa_pairs_combined.jsonl` is an intermediate output from the original Week 1 notebook. It contains un-enriched extracted pairs and is not read by the app; the enriched corpus is the runtime input.

## Retrieval and answer flow

The app cleans HTML entities and whitespace, filters very short/long questions and low-score answers, and deduplicates exact Q&A pairs. It searches cleaned question plus answer text with BM25 (normalized tokenization, stopword removal, stemming) and BGE small dense embeddings; Reciprocal Rank Fusion combines ranked lists. A cross-encoder reranks 40 candidates, then results are limited to one per AMA thread. A configurable abstention gate avoids generation when no reranked source clears its threshold. The default threshold is model specific; tune `ABSTAIN_THRESHOLD` against a labeled evaluation set.

Gemini 2.5 Flash receives full retrieved answers and is instructed to cite source numbers inline. The app removes citations to nonexistent source IDs. This basic validation does not prove each citation supports its claim, and the generated answer can still be wrong.

## Evaluation status

The original notebook reports faithfulness 0.81 from 20 questions judged by Gemini 2.5 Flash, which was also the generator family. The questions used the best retrieved answer as a reference, so this is not an independent benchmark. It is not evidence for a “no hallucination” claim. No reproducible labeled retrieval benchmark, ablation results, or latency measurements are currently checked in; those should be added before claiming comparative performance.

| Stage | Recall@k | MRR |
|---|---:|---:|
| BM25 | Not measured | Not measured |
| Dense | Not measured | Not measured |
| Hybrid RRF | Not measured | Not measured |
| Hybrid + reranker | Not measured | Not measured |

## Known limitations

- OP inference uses author frequency among top-level comments. It can misidentify the AMA subject, and the extractor cannot recover context missing from the streamed sample.
- Thread metadata enrichment relies on a third-party archive API.
- The corpus is old, uneven, user generated, and not representative of Reddit or the public generally. Votes are a noisy quality signal.
- The corpus contains sensitive topics and harmful language. Retrieval does not currently have a dedicated content moderation or self-harm escalation layer.
- Abstention is heuristic until calibrated on labeled data. Cross-encoder scores are not probabilities.
- This is a local demo pipeline, not a production service: there is no authentication, rate limiting, monitoring, or formal safety review.

## Source notebooks

The Week 1 and Week 2 notebooks in the original project folder document the exploratory collection and prototype evolution. The checked-in scripts are the maintained execution path. One notebook includes a hardcoded Gemini credential; revoke and rotate that key, and remove it from any shared notebook copies/history.
