#!/usr/bin/env bash
# Usage: deploy/push_space.sh <hf-username>/<space-name>
# Prereqs: create the Space on huggingface.co (SDK: Gradio), `huggingface-cli login`,
# and add Space secrets GEMINI_API_KEY and ARTIFACTS_REPO (Settings -> Variables and secrets).
set -euo pipefail
SPACE="${1:?usage: push_space.sh user/space-name}"
TMP="$(mktemp -d)"
git clone "https://huggingface.co/spaces/${SPACE}" "$TMP/space"
cp -r app.py rabbithole requirements.txt "$TMP/space/"
cp deploy/space_README.md "$TMP/space/README.md"
cd "$TMP/space"
git add -A
git commit -m "Deploy RabbitHole v2" || echo "nothing to commit"
git push
echo "Pushed. Build logs: https://huggingface.co/spaces/${SPACE}"
