#!/usr/bin/env bash
# prepare-backend.sh — Copy the torch-free backend from the main worktree into electron-wave2a.
# Run from the electron-wave2a worktree root: bash scripts/prepare-backend.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WORKTREE_ROOT/.." && pwd)"
BACKEND_SRC="$REPO_ROOT/backend"
BACKEND_DST="$WORKTREE_ROOT/backend"

if [ ! -d "$BACKEND_SRC" ]; then
  echo "ERROR: backend/ not found at $BACKEND_SRC (main worktree)" >&2
  echo "Ensure you are running from the electron-wave2a worktree." >&2
  exit 1
fi

echo "Copying backend from main worktree: $BACKEND_SRC"
echo "  → $BACKEND_DST"

mkdir -p "$BACKEND_DST"

rsync -av --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'tests/' \
  --exclude '.pytest_cache' \
  "$BACKEND_SRC/" "$BACKEND_DST/"

echo "Done. Backend copied to $BACKEND_DST"
echo "Torch-free check:"
grep -r "import torch\|from torch" "$BACKEND_DST/" || echo "  PASS — no torch imports"
grep -r "PyQt5\|import PyQt5" "$BACKEND_DST/" || echo "  PASS — no PyQt5 imports"
