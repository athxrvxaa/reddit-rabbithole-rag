# Kaggle runbook

The runnable cell-by-cell version is [notebooks/kaggle_run.ipynb](../notebooks/kaggle_run.ipynb). This guide mirrors its current workflow.

Python 3.10–3.13 are supported.

## Before running

- Turn **Internet** on.
- Add `HF_TOKEN` and `GEMINI_API_KEY` under Kaggle Add-ons → Secrets.
- A GPU is needed only for `index`; ingest, metadata, and extraction do not need one.
- Install `requirements-pipeline.txt`.

Set `RR_DATA_DIR` to `/kaggle/working/data`, `RR_ARTIFACTS_DIR` to `/kaggle/working/artifacts`, and `ARTIFACTS_REPO` to your public artifact dataset repo. Keep a separate private repo for DB checkpoints.

## Smoke test

Use a separate directory for a small first-N smoke test. It checks the basic pipeline, but the selected rows are from the beginning of the source.

```bash
python -m rabbithole ingest --method stream --max-comments N
python -m rabbithole meta
python -m rabbithole extract
python -m rabbithole stats
```

Review pair quality, `op_source`, categories, answer lengths, and the year histogram before continuing.
Replace the placeholder `N` with a small smoke-test row cap.

## B0: inspect and choose a sample

```bash
python -m rabbithole inspect-source --target-rows 6500000
```

This reads the first 20,000 rows, measures their actual read speed, and estimates a full-scan duration from that measurement. The Hugging Face IAmA split is time-ordered. A thread-sampled full pass is needed for coverage across its timeline; interrupting early leaves only the oldest slice. Use the same `--thread-sample-rate` on resume.

## B1: ingest

```bash
python -m rabbithole ingest --method parquet --thread-sample-rate RATE --target-rows 6500000
```

Parquet is the default fast path. It discovers parquet shards whose filenames contain the configured split name, processes them with up to four downloads in flight, and records completed shard names. If no matching shards are available, it reports that and falls back to streaming. Choose `--method stream` to force streaming. Both methods filter deleted/removed comments and use the same deterministic thread hash sampling. Ingest reports measured rows per second and an ETA.

The ingest DB is resumable. To preserve a checkpoint across Kaggle sessions, upload it to a private dataset repo:

```bash
python -m rabbithole upload --file /kaggle/working/data/rabbithole.db --in-repo rabbithole.db --repo YOUR_PRIVATE_CHECKPOINT_REPO --private
```

Download it back into the same data directory before resuming in a new session. Keep the sample rate unchanged.

## B2: metadata, extraction, and stats

```bash
python -m rabbithole meta
python -m rabbithole extract
python -m rabbithole stats
```

Metadata retries failed batches up to three more times. The default minimum thread size comes from `ExtractConfig.min_thread_comments`; override `--min-comments` only for a deliberate different threshold. `stats` includes the year histogram and min/max years, and warns when the sample spans fewer than three years.

## B3: index and publish

Turn on the accelerator for the index build if available.

```bash
python -m rabbithole index
python -m rabbithole upload --repo YOUR_PUBLIC_ARTIFACTS_REPO
```

The manifest records the index's minimum and maximum years. The app shows that range in its header.

## C: retrieval and answer checks

```bash
python -m rabbithole search "What is the hardest part of being a nurse?"
python -m rabbithole ask "What is the hardest part of being a nurse?"
```

`search` prints the top reranker score and applied confidence tier. The hard-abstain threshold defaults to `RR_ABSTAIN_THRESHOLD=-5.0`; `RR_LOWCONF_THRESHOLD=-1.0` marks low-confidence answers while allowing Gemini to decide whether the sources cover the question. Threshold values depend on the selected reranker.

## D: evaluation

```bash
python -m rabbithole eval-make -n 200 --sleep 6.5 --model gemini-flash-lite-latest
python -m rabbithole eval-retrieval
python -m rabbithole calibrate
python -m rabbithole eval-generation --judge --judge-model gemini-flash-lite-latest --limit LIMIT
```

`eval-make` appends and resumes by source idx across notebook runs. If daily Gemini quota is reached, it saves progress and stops; continue after the quota resets or use configured fallback models. `eval-retrieval --reranker MODEL` compares a chosen reranker; `--sweep` checks dense RRF weights 1/1.5/2/3 with rerank candidate counts 30/50/100 without changing defaults. Retrieval output includes 95% intervals and per-query ranks. Calibration prints threshold pass/abstain rates and writes `eval/results/calibration.json`; calibrate separately for each reranker. `eval-generation` saves partial rows if Gemini quota is exhausted.

429 handling waits on the provided retry delay for per-minute quotas (capped at 60 seconds), then tries the model again. Per-day quota and exhausted retry attempts move to the next fallback model. If all models are exhausted, the command receives `GeminiQuotaError` and the eval command preserves completed rows.

See [README.md](../README.md) for the completed-run corpus numbers and preliminary retrieval table. The current Kaggle result covers only the oldest ~24% of the time-ordered split, and generation has not been evaluated end to end because of free-tier Gemini quota limits.
