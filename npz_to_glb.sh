#!/usr/bin/env bash
# Convert an existing lingbot-map NPZ directory to GLB (no re-inference).
# Usage:
#   bash npz_to_glb.sh /workspace/outputs/courthouse/courthouse /workspace/outputs/courthouse/courthouse.glb
set -euo pipefail

NPZ_DIR="${1:-/workspace/outputs/courthouse/courthouse}"
GLB_PATH="${2:-/workspace/outputs/courthouse/courthouse.glb}"
REPO_DIR="${REPO_DIR:-/workspace/lingbot-map}"
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"
ENV_NAME="${ENV_NAME:-lingbot-map}"

if [[ ! -d "$NPZ_DIR" ]]; then
  echo "NPZ directory not found: $NPZ_DIR" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$(dirname "$GLB_PATH")"
cd "$REPO_DIR/demo_render"

python -m interactive_viewer.npz_to_glb \
  --input_dir "$NPZ_DIR" \
  --output "$GLB_PATH" \
  --downsample 2 \
  --max_points 5000000

ls -lh "$GLB_PATH"
