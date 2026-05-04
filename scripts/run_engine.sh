#!/usr/bin/env bash
# Start FusionTrack FastAPI on a GPU instance (Vast.ai, RunPod, local).
# Usage: from repo root, after `source .venv/bin/activate` is optional — script activates .venv if present.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT="${FUSIONTRACK_PORT:-8000}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ROOT/.cache/ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
