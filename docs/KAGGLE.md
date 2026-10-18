> Easier: use `notebooks/kaggle_run.ipynb`, which has all of this as ready-to-run cells.

# Running the heavy steps on Kaggle (or Colab)

Settings → Internet **On**; Accelerator: **GPU T4** (only needed for `index`). Add secrets under Add-ons → Secrets: `GEMINI_API_KEY`, `HF_TOKEN`.

## Cell 1 — setup
```python
!git clone -b v2 https://github.com/athxrvxaa/reddit-rabbithole-rag.git
%cd reddit-rabbithole-rag
!pip install -q -r requirements-pipeline.txt

import os
from kaggle_secrets import UserSecretsClient
sec = UserSecretsClient()
os.environ["GEMINI_API_KEY"] = sec.get_secret("GEMINI_API_KEY")
os.environ["HF_TOKEN"] = sec.get_secret("HF_TOKEN")
os.environ["RR_DATA_DIR"] = "/kaggle/working/data"
os.environ["RR_ARTIFACTS_DIR"] = "/kaggle/working/artifacts"
os.environ["ARTIFACTS_REPO"] = "athxrvxaa/rabbithole-artifacts"   # your HF dataset repo
```

## Cell 2 — look before you leap (1–2 min)
```python
!python -m rabbithole ingest --max-comments 100000
!python -m rabbithole meta --min-comments 5
!python -m rabbithole extract
!python -m rabbithole stats
```
Check the stats: pairs per thread, `op_source` should be almost all `submission`, category spread, answer lengths. Fix filters in `rabbithole/extract.py` (`ExtractConfig`) if something looks off. Delete `/kaggle/working/data` before the full run.

## Sampling (you don't need the whole split)
`python -m rabbithole inspect-source --target-rows 5000000` shows the split size and layout and suggests a rate.
Then ingest with `--thread-sample-rate <rate>` (whole threads, spread over the full split; needs a full scan)
or `--max-comments 5000000` (first N rows; faster, but only the start of the split). The notebook does this for you.

## Cell 3 — ingest (resumable)
```python
!python -m rabbithole ingest
!python -m rabbithole upload --file /kaggle/working/data/rabbithole.db --in-repo rabbithole.db --repo athxrvxaa/rabbithole-checkpoints --private
```
The upload goes to a separate **private** repo (never the artifacts repo, which the Space downloads from). It is a checkpoint: Kaggle's disk disappears when the session ends. If the session dies mid-ingest, re-run the same cell; it resumes from the saved row count (if the working dir survived), or download the checkpoint first.

## Cell 4 — meta, extract, index
```python
!python -m rabbithole meta
!python -m rabbithole extract
!python -m rabbithole stats
!python -m rabbithole index
!python -m rabbithole upload --repo $ARTIFACTS_REPO
```

## Cell 5 — evaluate
```python
!python -m rabbithole eval-make -n 200
!python -m rabbithole eval-retrieval
!python -m rabbithole calibrate
```

## Troubleshooting
- `datasets` complains about scripts / trust_remote_code → `pip install "datasets<4"`.
- Arctic Shift returns many 429s → re-run `meta`; it only fetches threads it doesn't have yet.
- Embedding is slow → make sure the GPU is on; lower `RR_EMBED_MAX_LEN` (e.g. 256).
- Out of memory at index time → the build streams in batches; lower `RR_EMBED_BATCH`.
- Gemini 404 on the model → run `python -m rabbithole gemini-models` and set `GEMINI_MODEL`.
