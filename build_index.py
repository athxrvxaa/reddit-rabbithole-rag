"""Validate/clean corpus and materialize optional embeddings for inspection."""
import argparse
import json
from pathlib import Path

from retrieval import load_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = Path("data/ama_qa_enriched.jsonl")
    if not default_input.exists():
        default_input = Path("ama_qa_enriched.jsonl")
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output", default="data/ama_qa_enriched.jsonl")
    args = parser.parse_args()
    rows = load_corpus(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} cleaned, unique Q&A rows to {output}")
    print("FAISS, embeddings, and BM25 are cached safely by app.py on first startup; no pickle is loaded.")


if __name__ == "__main__":
    main()
