# LingBot-Map on RunPod

Bring up [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map) on a GPU pod for **offline** reconstruction (headless batch inference → point cloud / GLB / optional flythrough MP4).

## What this model actually outputs

LingBot-Map is a **feed-forward dense reconstructor** (poses + depth / world points). The offline entry point is `demo_render/batch_demo.py`. It is **not** a 3D Gaussian Splatting trainer.

| Output | How |
|--------|-----|
| **GLB** (point cloud mesh scene) | `--save_glb` → writes `{scene}.glb` under `--output_folder` |
| **NPZ** predictions | `--save_predictions` |
| **Flythrough MP4** | default (skip with `--no_render`) |

`--model_path` is always the neural network checkpoint (`.pt`). It is **not** an output path — do not put a `.glb` extension there.

## Pod settings (every time you create a pod)

| Setting | Suggestion |
|--------|------------|
| GPU | RTX 4090 (or 3090 / A6000 if cheaper) |
| Template | Recent PyTorch / CUDA 12.x |
| Volume | Network Volume → `/workspace` |
| Disk | ≥ 40 GB volume (container disk can stay small) |
| Ports | None required for offline/GLB. Expose **8080** only if you also use the interactive viser demo. |

What persists where:

- **On the volume (`/workspace`)** — Miniforge, conda env, cloned repo, checkpoints, your images/outputs. Survives pod teardown.
- **On the container (ephemeral)** — OS packages like `ffmpeg` / build tools from `apt`. Gone when the pod dies.

---

## Path A — New instance + **new** volume (first time)

Do this once per Network Volume. Expect a long run (downloads + Kaolin / CUDA extensions).

1. Create a Network Volume and attach it at `/workspace`.
2. Deploy a GPU pod that mounts that volume.
3. Copy `setup_runpod.sh` onto the pod, then:

```bash
# optional: export HF_TOKEN=hf_...   # helps with HF rate limits
bash setup_runpod.sh
```

This installs the **offline stack by default** (`WITH_RENDER=1`): Miniforge, PyTorch cu128, lingbot-map, FlashInfer, Kaolin, render CUDA extensions, and the default checkpoint.

Optional:

```bash
MODEL=lingbot-map-long bash setup_runpod.sh   # long-sequence checkpoint instead of default
WITH_RENDER=0 bash setup_runpod.sh            # skip Kaolin / video renderer (GLB-only still needs a careful path; prefer default)
```

4. Run the bundled courthouse example (offline → GLB, no MP4):

```bash
source /workspace/activate_lingbot.sh
bash /workspace/run_example.sh
```

Output lands under `/workspace/outputs/courthouse/` (including `courthouse.glb`).

When finished, **stop/terminate the pod**. Keep the Network Volume.

---

## Path B — New instance + **existing** volume (already set up)

Mount the **same** volume at `/workspace` on a new GPU pod.

**Usually just run:**

```bash
source /workspace/activate_lingbot.sh
bash /workspace/run_example.sh
```

**Only re-run setup if something is missing or you want an update:**

```bash
bash setup_runpod.sh   # safe / idempotent — skips work that's already done
```

Typical reasons:

- New pod template is missing `ffmpeg` or compilers
- You want `git pull` + dependency refresh
- You want another checkpoint (`MODEL=lingbot-map-long bash setup_runpod.sh`)

---

## Your own data → GLB

```bash
source /workspace/activate_lingbot.sh
cd /workspace/lingbot-map

# Images: put a scene folder under a parent, e.g. /workspace/inputs/myscene/*.jpg
python demo_render/batch_demo.py \
  --input_folder /workspace/inputs \
  --scenes myscene \
  --output_folder /workspace/outputs/myscene \
  --model_path /workspace/checkpoints/lingbot-map.pt \
  --mask_sky \
  --save_glb \
  --no_render

# Or from a video:
python demo_render/batch_demo.py \
  --video_path /workspace/inputs/clip.mp4 \
  --output_folder /workspace/outputs/clip \
  --model_path /workspace/checkpoints/lingbot-map.pt \
  --fps 10 \
  --mask_sky \
  --save_glb \
  --no_render
```

Result: `/workspace/outputs/.../<scene>.glb`

Also useful:

- Add `--save_predictions` to keep per-frame NPZs
- Drop `--no_render` (and keep the render stack installed) to also write a point-cloud flythrough MP4
- Long sequences: `--mode windowed --window_size 128 --overlap_keyframes 8 --keyframe_interval 10`

### Interactive viser (optional)

Not needed for GLB. If you want it anyway: expose port 8080 and run `python demo.py --model_path ... --image_folder ... --port 8080`.

## Cost tip

Terminate the pod when idle; keep the Network Volume. Path B is the cheap day-to-day loop: spin up → activate → batch_demo → shut down.
