#!/usr/bin/env bash
# =============================================================================
# LingBot-Map — RunPod initialization script
# =============================================================================
# Designed for a fresh GPU pod (RTX 4090 / similar). Idempotent: safe to re-run.
# Default target: offline batch pipeline (demo_render/batch_demo.py) → GLB / MP4.
#
# Recommended RunPod settings:
#   - GPU: RTX 4090 (24 GB) — good price/VRAM balance for this model
#   - Template: any recent PyTorch / CUDA 12.x image (driver ≥ 550)
#   - Disk: ≥ 40 GB Network Volume at /workspace
#   - No HTTP port required for offline/GLB (8080 only if using interactive viser)
#
# Usage on the pod:
#   bash setup_runpod.sh
#   # optional flags via env:
#   WITH_RENDER=0 bash setup_runpod.sh          # skip Kaolin / video renderer
#   MODEL=lingbot-map-long bash setup_runpod.sh # download the long-sequence ckpt
#   SKIP_MODEL_DOWNLOAD=1 bash setup_runpod.sh  # env + deps only
#   WORKSPACE=/workspace bash setup_runpod.sh
#
# After setup:
#   source "$WORKSPACE/activate_lingbot.sh"
#   bash "$WORKSPACE/run_example.sh"
# =============================================================================

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/lingbot-map}"
CONDA_ROOT="${CONDA_ROOT:-$WORKSPACE/miniforge3}"
ENV_NAME="${ENV_NAME:-lingbot-map}"
MODEL="${MODEL:-lingbot-map}"   # lingbot-map | lingbot-map-long | lingbot-map-stage1
CKPT_DIR="${CKPT_DIR:-$WORKSPACE/checkpoints}"
WITH_RENDER="${WITH_RENDER:-1}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# PyTorch pin from upstream README (Kaolin-compatible for optional render path)
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
log "Preflight checks"
mkdir -p "$WORKSPACE" "$CKPT_DIR"
need_cmd curl
need_cmd git

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  ok "NVIDIA GPU detected"
else
  warn "nvidia-smi not found — continuing, but inference will fail without a GPU"
fi

# Free disk (best-effort)
if df -h "$WORKSPACE" >/dev/null 2>&1; then
  df -h "$WORKSPACE" | tail -1
fi

# -----------------------------------------------------------------------------
# System packages (Debian/Ubuntu RunPod images)
# -----------------------------------------------------------------------------
install_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not available; skipping system package install"
    return
  fi
  log "Installing system packages (ffmpeg, build tools)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    >/dev/null
  ok "System packages ready"
}

# Only attempt apt if we appear to have root (RunPod usually does)
if [[ "$(id -u)" -eq 0 ]]; then
  install_apt_packages
else
  warn "Not root — skipping apt installs. Ensure ffmpeg + build-essential exist."
fi

# -----------------------------------------------------------------------------
# Miniforge (persists on /workspace if you use a Network Volume)
# -----------------------------------------------------------------------------
ensure_conda() {
  if [[ -x "$CONDA_ROOT/bin/conda" ]]; then
    ok "Conda already present at $CONDA_ROOT"
  else
    log "Installing Miniforge into $CONDA_ROOT"
    local installer="/tmp/miniforge.sh"
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$installer"
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
    ok "Miniforge installed"
  fi
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda config --set always_yes yes
  conda config --set channel_priority flexible
}

ensure_conda

# -----------------------------------------------------------------------------
# Conda env + PyTorch
# -----------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  ok "Conda env '$ENV_NAME' already exists"
else
  log "Creating conda env '$ENV_NAME' (Python ${PYTHON_VERSION})"
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"
ok "Activated $ENV_NAME ($(python -V))"

log "Installing PyTorch ${TORCH_VERSION} (CUDA 12.8 wheels)"
pip install --upgrade pip setuptools wheel
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "$TORCH_INDEX_URL"

python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: {torch.cuda.get_device_capability(0)}")
PY

# -----------------------------------------------------------------------------
# Clone / update lingbot-map
# -----------------------------------------------------------------------------
if [[ -d "$REPO_DIR/.git" ]]; then
  log "Updating existing clone at $REPO_DIR"
  git -C "$REPO_DIR" pull --ff-only || warn "git pull failed — using existing checkout"
else
  log "Cloning Robbyant/lingbot-map → $REPO_DIR"
  git clone --depth 1 https://github.com/Robbyant/lingbot-map.git "$REPO_DIR"
fi

# -----------------------------------------------------------------------------
# Install package + attention backend
# -----------------------------------------------------------------------------
log "Installing lingbot-map (editable) + visualization extras"
cd "$REPO_DIR"
pip install -e ".[vis]"

log "Installing FlashInfer (paged KV cache; falls back to SDPA if unavailable)"
# Prefer PyPI; --index-url avoids mirrors that omit the package
pip install --index-url https://pypi.org/simple flashinfer-python || {
  warn "FlashInfer install failed — demo can still run with --use_sdpa"
}

# Optional JIT cache for CUDA 12.8 (speeds first FlashInfer compile)
pip install flashinfer-jit-cache \
  -f https://flashinfer.ai/whl/cu128/flashinfer-jit-cache/ \
  || warn "flashinfer-jit-cache optional install skipped"

# Hugging Face CLI for checkpoint download
pip install -q "huggingface_hub[cli]"

# -----------------------------------------------------------------------------
# Optional offline render stack (Kaolin + CUDA extensions)
# -----------------------------------------------------------------------------
if [[ "$WITH_RENDER" == "1" ]]; then
  log "Installing offline render extras (WITH_RENDER=1)"
  # Upstream documents pip install -e ".[vis,render]" but pyproject has no
  # [render] extra yet — use demo_render/requirements.txt instead.
  if [[ -f demo_render/requirements.txt ]]; then
    pip install -r demo_render/requirements.txt
  else
    pip install "numpy==1.26.4" "open3d==0.19.0" pyyaml
    pip install onnxruntime-gpu || pip install onnxruntime
  fi
  pip install --index-url https://pypi.org/simple \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html \
    || warn "Kaolin wheel install failed — build from source if you need batch_demo.py"
  if [[ -d demo_render/render_cuda_ext ]]; then
    (cd demo_render/render_cuda_ext && python setup.py build_ext --inplace) \
      || warn "CUDA render extensions failed to build"
  fi
else
  warn "Skipping render stack (WITH_RENDER=0). Re-run with WITH_RENDER=1 for flythrough MP4 / Kaolin."
fi

# -----------------------------------------------------------------------------
# Model download
# -----------------------------------------------------------------------------
CKPT_PATH="$CKPT_DIR/${MODEL}.pt"
if [[ "$SKIP_MODEL_DOWNLOAD" == "1" ]]; then
  warn "Skipping model download (SKIP_MODEL_DOWNLOAD=1)"
elif [[ -f "$CKPT_PATH" ]]; then
  ok "Checkpoint already present: $CKPT_PATH"
else
  log "Downloading checkpoint '${MODEL}.pt' from Hugging Face → $CKPT_DIR"
  # Public repo; HF_TOKEN helps with rate limits if set
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  fi
  if command -v hf >/dev/null 2>&1; then
    hf download robbyant/lingbot-map "${MODEL}.pt" --local-dir "$CKPT_DIR"
  else
    huggingface-cli download robbyant/lingbot-map "${MODEL}.pt" --local-dir "$CKPT_DIR"
  fi
  # hf may nest under a subdir depending on version — normalize to CKPT_PATH
  if [[ ! -f "$CKPT_PATH" ]]; then
    found="$(find "$CKPT_DIR" -name "${MODEL}.pt" -type f | head -n 1 || true)"
    if [[ -n "$found" && "$found" != "$CKPT_PATH" ]]; then
      ln -sfn "$found" "$CKPT_PATH"
    fi
  fi
  [[ -f "$CKPT_PATH" ]] || die "Checkpoint download finished but $CKPT_PATH is missing"
  ok "Downloaded $CKPT_PATH"
fi

# -----------------------------------------------------------------------------
# Helper launcher
# -----------------------------------------------------------------------------
RUN_EXAMPLE="$WORKSPACE/run_example.sh"
OUTPUT_DIR="$WORKSPACE/outputs/courthouse"
cat > "$RUN_EXAMPLE" <<EOF
#!/usr/bin/env bash
# Offline courthouse example → GLB (skips flythrough MP4 by default).
set -euo pipefail
WORKSPACE="${WORKSPACE}"
REPO_DIR="${REPO_DIR}"
CONDA_ROOT="${CONDA_ROOT}"
ENV_NAME="${ENV_NAME}"
CKPT_PATH="${CKPT_PATH}"
OUTPUT_DIR="${OUTPUT_DIR}"

# shellcheck disable=SC1091
source "\$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "\$ENV_NAME"
cd "\$REPO_DIR"

if [[ ! -f "\$CKPT_PATH" ]]; then
  echo "Missing checkpoint: \$CKPT_PATH"
  echo "Re-run setup_runpod.sh or set MODEL=... and download manually."
  exit 1
fi

mkdir -p "\$OUTPUT_DIR"
echo "Running offline batch_demo on example/courthouse"
echo "  checkpoint: \$CKPT_PATH"
echo "  output:     \$OUTPUT_DIR/courthouse.glb  (via --save_glb)"
echo "  tip: drop --no_render from this script to also write an MP4 flythrough"
echo

exec python demo_render/batch_demo.py \\
  --input_folder example \\
  --scenes courthouse \\
  --output_folder "\$OUTPUT_DIR" \\
  --model_path "\$CKPT_PATH" \\
  --mask_sky \\
  --save_glb \\
  --no_render \\
  "\$@"
EOF
chmod +x "$RUN_EXAMPLE"

ACTIVATE_HINT="$WORKSPACE/activate_lingbot.sh"
cat > "$ACTIVATE_HINT" <<EOF
#!/usr/bin/env bash
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${REPO_DIR}"
echo "Activated ${ENV_NAME} — cwd=\$(pwd)"
echo "Checkpoint: ${CKPT_PATH}"
EOF
chmod +x "$ACTIVATE_HINT"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
log "Setup complete"
cat <<EOF

Workspace layout
  Repo:        $REPO_DIR
  Conda:       $CONDA_ROOT  (env: $ENV_NAME)
  Checkpoint:  $CKPT_PATH
  Launch:      $RUN_EXAMPLE
  Render:      WITH_RENDER=$WITH_RENDER

Quick start (offline → GLB)
  source $WORKSPACE/activate_lingbot.sh
  bash $RUN_EXAMPLE
  # → $OUTPUT_DIR/courthouse.glb

Your own data (GLB)
  python demo_render/batch_demo.py \\
    --input_folder /workspace/inputs --scenes myscene \\
    --output_folder /workspace/outputs/myscene \\
    --model_path $CKPT_PATH --mask_sky --save_glb --no_render

  # --model_path is always the .pt checkpoint, never a .glb path.
  # --save_glb writes <scene>.glb into --output_folder.

Long sequences: add windowed flags, e.g.
  --mode windowed --window_size 128 --overlap_keyframes 8 --keyframe_interval 10

If FlashInfer misbehaves: add --use_sdpa
If OOM: try --num_scale_frames 2

RunPod tip: Network Volume at /workspace so conda + checkpoints survive teardown.

EOF
ok "Ready."
