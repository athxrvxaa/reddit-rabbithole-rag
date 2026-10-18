# Reddit RabbitHole RAG

Ask a question and get an answer cited from Reddit r/IAmA threads. Claims link to the original comment. AMA posts are personal anecdotes, not verified facts or professional advice.

## Pipeline

```text
Hugging Face IAmA comments --stream/parquet--> SQLite comments
Arctic Shift submission metadata ------------> SQLite threads
comments + threads --> clean Q&A extraction --> SQLite pairs
pairs --> dense embeddings + BM25 --> FAISS/BM25 index --> RRF --> cross-encoder
retrieved sources --> confidence gate --> grounded Gemini answer + citations
```

## Current corpus (Build 2)

Build 2 resumed the same database at row 6,220,000, where 915,905 comments were already stored. It used a thread sample rate of 0.19396 and was interrupted by the user around 26,200,000 scanned rows, with 4,020,606 comments kept. There is no completion marker. The scanned count exceeds the split metadata count of 25,778,822, so that metadata is stale or the stream is longer; stream completion and full timeline coverage are unconfirmed. The first 20,000 source rows span 2009-05-28 to 2009-07-11. True speed was about 2,600 rows/s (about 19.98M rows over 7,670 seconds); the old log incorrectly included the resume offset in the numerator.

The private checkpoint at `athxrvxaa/rabbithole-checkpoints` was 1.04 GB and contained comments only; pairs and thread metadata were not included. Metadata requested 11,003 threads (at least 10 comments): 10,952 found, 1 missing, and 50 failed in one batch. There were 20,607 threads with metadata in total.

Extraction saw 40,787 threads: 20,124 were too small, 7,239 had no pairs, and 13,424 had pairs. It produced 200,181 unique pairs; all had `op_source=submission`. Several threads reached the 200-pair cap. Fifty-six threads with at least 10 comments had no metadata. Answer length p10/p50/p90 was 106/259/780 characters (maximum 3,991).

Categories: other 23.2%, entertainment_media 15.2%, health_medicine 10.6%, tech 6.8%, education 6.3%, business_finance 5.7%, crime_law 4.8%, relationships_family 4.7%, identity_lifestyle 3.9%, jobs_trades 3.8%, military 3.7%, gaming_sports 3.6%, religion 2.1%, science_space 2.1%, travel_adventure 2.1%, politics_government 1.4%.

The index contains 200,181 documents with 384-dimensional embeddings. It took 277 seconds on a T4 (about 941 documents/s). The public artifact dataset `athxrvxaa/rabbithole-artifacts` contains 937 MB and replaced the prior Build 1 artifacts. A prison-life search returned four `crime_law` hits; the top reranker score was 2.05 for a hypothetical about painting prisons.

## Build history

| Build | Comments kept | Pairs | Threads with pairs | Index time | Artifact size | Interruption |
|---|---:|---:|---:|---:|---:|---|
| Build 1 | 915,905 | 64,600 | 5,912 | 76 s | 284 MB | Ingest stopped around 6.22M scanned rows |
| Build 2 | 4,020,606 | 200,181 | 13,424 | 277 s | 937 MB | Resumed at 6.22M and user-interrupted around 26.2M scanned rows; no completion marker |

## Preliminary retrieval results

**Measured on Build 1 (64,600-pair index), n=39 synthetic queries, not on the current index; regenerate with `eval-make` after a new Gemini key.** The table is preliminary, with approximately ±15 percentage points of uncertainty, and is not enough to rank modes. Latency is reported mean/p95 in milliseconds.

| Mode | R@1 | R@5 | R@10 | R@20 | MRR@20 | Thread R@5 | Latency ms (mean / p95) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | .308 | .615 | .641 | .667 | .436 | .667 | 4 / 3 |
| Dense | .462 | .641 | .744 | .821 | .555 | .769 | 28 / 22 |
| Hybrid | .513 | .667 | .692 | .744 | .573 | .718 | 23 / 26 |
| Hybrid + rerank | .615 | .692 | .769 | .769 | .653 | .795 | 100 / 119 |

The query set is synthetic, and only the source pair counts as pair-level relevant. Thread-level recall is also reported. New eval runs show 95% intervals; overlapping intervals mean the sample does not support naming a winner. An eval run against Build 2 used stale query ids and returned all zeros; those results were discarded.

## Confidence and calibration

Reranker scores depend on the reranker model. `RR_ABSTAIN_THRESHOLD` is the hard-abstain threshold (default `-5.0`): below it, skip Gemini. `RR_LOWCONF_THRESHOLD` (default `-1.0`) marks a result low confidence while still allowing an answer. Between the thresholds, retrieval does not hard-abstain and Gemini's grounded “not covered” instruction decides whether to answer. These two-tier defaults are unchanged.

Build 2 calibration (old rule, old 30/12 seed set): in-domain top rerank p10/median/p90 = 2.15/3.94/6.72; OOD = -6.97/-3.69/-1.07; suggested threshold 2.05; balanced accuracy 0.983. `TODO(recalibrate): rerun calibrate with the expanded seed set on the current artifacts`.

The ms-marco reranker once ranked a prison-painting hypothetical highly for a question about daily prison life, showing that lexical overlap can mislead. Retrieval evaluation and manual review remain necessary.

## Run locally

Python 3.10-3.13 is supported. On Windows, run `./run_local.ps1` from PowerShell. It creates `.venv`, installs app dependencies, downloads the public runtime artifacts on first start, and serves at <http://127.0.0.1:7860/>. Set `GRADIO_SERVER_PORT` to choose another port; if it is occupied, Gradio searches for an available port. Keep the window open while using the app. To skip dependency installation, run `./run_local.ps1 -SkipInstall`.

The public index is configured by default. Copy `.env.example` to `.env` and add a valid `GEMINI_API_KEY` for generated answers. Without an API key, retrieval can load but answer generation reports a missing Gemini configuration. Never commit `.env`.

For pipeline development, install dependencies and configure credentials in the environment:

```bash
python -m venv .venv
pip install -r requirements-pipeline.txt
python -m rabbithole inspect-source --target-rows 6500000
python -m rabbithole ingest --method parquet --thread-sample-rate 0.19396 --target-rows 6500000
python -m rabbithole meta
python -m rabbithole extract
python -m rabbithole stats
python -m rabbithole index
python -m rabbithole search "What is the hardest part of being a nurse?"
```

Parquet ingest resumes by completed shard; streaming is the fallback. Both methods use deterministic whole-thread sampling. Keep the same `--thread-sample-rate` when resuming. The source is time-ordered, so interrupting an early scan leaves only its oldest slice. See [docs/KAGGLE.md](docs/KAGGLE.md) for checkpointing and the Kaggle workflow.

`stats` reports years from the corpus DB. Artifacts built before `created_utc` was added have no year bounds; year coverage for those artifacts needs a rebuild. Build 2 scanned about 26.2M rows, more than the metadata count, and was interrupted before a completion marker, so full-timeline coverage is unconfirmed. **TODO(after stats): exact year range.**

## Evaluation and limitations

Generation quality has not been evaluated end to end because Gemini quota errors stopped query generation and generation evaluation. The current retrieval table is Build 1 only. Categories are keyword heuristics; many pairs can be classified as `other`. AMA posts are personal anecdotes, not verified claims or professional advice. The corpus contains sensitive topics, including abuse, addiction, and self-harm.

See [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for the architecture and interview guide, and [notebooks/kaggle_run.ipynb](notebooks/kaggle_run.ipynb) for the Kaggle notebook template.
