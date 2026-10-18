"""Fetch / publish build artifacts via a Hugging Face *dataset* repo."""
from __future__ import annotations

from pathlib import Path

# embeddings.npy is only needed to re-index; the FAISS index already holds the vectors.
REQUIRED = ["manifest.json", "faiss.index", "corpus.db", "cat_ids.npy", "thread_codes.npy"]
APP_PATTERNS = REQUIRED + ["bm25/*"]


def artifacts_ready(d: Path) -> bool:
    return all((d / f).exists() for f in REQUIRED) and (d / "bm25").exists()


def ensure_artifacts(d: Path, repo_id: str = "") -> None:
    d = Path(d)
    if artifacts_ready(d):
        return
    if not repo_id:
        raise FileNotFoundError(
            f"Artifacts not found in {d}. Build them (python -m rabbithole index) or set ARTIFACTS_REPO to a HF dataset repo."
        )
    from huggingface_hub import snapshot_download

    print(f"Downloading artifacts from hf://datasets/{repo_id} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(d), allow_patterns=APP_PATTERNS)
    if not artifacts_ready(d):
        raise FileNotFoundError(f"Downloaded repo {repo_id} is missing required artifact files.")


def upload_folder(d: Path, repo_id: str, private: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(folder_path=str(d), repo_id=repo_id, repo_type="dataset", commit_message="Upload RabbitHole artifacts")
    print(f"Uploaded {d} -> https://huggingface.co/datasets/{repo_id}")


def upload_file(path: Path, repo_id: str, path_in_repo: str, private: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=repo_id, repo_type="dataset")
    print(f"Uploaded {path} -> hf://datasets/{repo_id}/{path_in_repo}")
