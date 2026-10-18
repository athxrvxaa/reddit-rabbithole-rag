# Kaggle runbook

The runnable cell-by-cell template is [notebooks/kaggle_run.ipynb](../notebooks/kaggle_run.ipynb). This guide mirrors that workflow.

Python 3.10-3.13 are supported.

## Before running

- Turn Internet on.
- Add `HF_TOKEN` and `GEMINI_API_KEY` under Kaggle Add-ons > Secrets.
- A GPU is needed for `index`; ingest, metadata, and extraction do not need one.
- Install `requirements-pipeline.txt`.

Set `RR_DATA_DIR=/kaggle/working/data`, `RR_ARTIFACTS_DIR=/kaggle/working/artifacts`, and `ARTIFACTS_REPO` to your public artifact dataset repo. Keep a separate private repo for DB checkpoints.

## Run and checkpoint

```bash
python -m rabbithole inspect-source --target-rows 6500000
python -m rabbithole ingest --method stream --thread-sample-rate 0.19396
python -m rabbithole meta
python -m rabbithole extract
python -m rabbithole stats
```

Ingest resumes the same DB. Its speed and ETA should use work scanned during the current session. Keep the sampling rate unchanged when resuming. Stream ingest must reach its real end and write the completion marker; split metadata count alone does not prove completion. Re-run `meta` once to retry failed batches.

After extraction and stats, checkpoint the full database privately so comments, metadata, and pairs survive session end:

```bash
python -m rabbithole upload --file /kaggle/working/data/rabbithole.db --in-repo rabbithole_extracted.db --repo YOUR_PRIVATE_CHECKPOINT_REPO --private
```

## Index and evaluate

```bash
python -m rabbithole index
python -m rabbithole upload --repo YOUR_PUBLIC_ARTIFACTS_REPO
python -m rabbithole search "What is the hardest part of being a nurse?"
```

After every index rebuild, delete `eval/synthetic_queries.jsonl` or regenerate it with `eval-make`; query identifiers from an older index can be stale. Generation and query creation require a working Gemini key.

## Build 2 run record

Build 2 resumed at row 6,220,000 with 915,905 comments already stored, sample rate 0.19396, and was interrupted around 26,200,000 scanned rows with 4,020,606 kept. The split metadata count is 25,778,822, so the greater scanned count means that metadata may be stale or the stream longer. There was no completion marker; timeline coverage is unconfirmed. True throughput was about 2,600 rows/s, not the cumulative-offset rate printed by that older run.

Metadata requested 11,003 threads (10,952 found, 1 missing, 50 failed in one batch); total metadata threads were 20,607. Extraction saw 40,787 threads: 20,124 too small, 7,239 with no pairs, and 13,424 with pairs. It produced 200,181 unique pairs, all from submission authors, with a cap of 200 per thread. The index has 200,181 documents, dimension 384, took 277 seconds on a T4 (about 941 docs/s), and the public artifacts were 937 MB. The private 1.04 GB checkpoint was uploaded before metadata and extraction, so it held comments only.

See [README.md](../README.md) for corpus, calibration, limitations, and the preliminary Build 1 retrieval table.

429 handling waits on the provided retry delay for per-minute quotas (capped at 60 seconds), then tries the model again. Per-day quota and exhausted retry attempts move to the next fallback model. If all models are exhausted, the command receives `GeminiQuotaError` and the eval command preserves completed rows.
