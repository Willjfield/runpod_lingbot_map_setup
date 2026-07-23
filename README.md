# LingBot-Map on RunPod

Bring up [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map) on a GPU pod for **offline** reconstruction (headless batch inference → point cloud / GLB / optional flythrough MP4).

## What this model actually outputs

LingBot-Map is a **feed-forward dense reconstructor** (poses + depth / world points). The offline entry point is `demo_render/batch_demo.py`. It is **not** a 3D Gaussian Splatting trainer.

| Output | How |
|--------|-----|
| **NPZ** predictions | `--save_predictions` → `{output}/{scene}/frame_*.npz` |
| **GLB** (point cloud) | convert NPZs with `python -m interactive_viewer.npz_to_glb` (see below) |
| **Flythrough MP4** | omit `--no_render` (needs Kaolin / render stack) |

Do **not** rely on `batch_demo.py --save_glb` for now — it calls `predictions_to_glb`, which expects `world_points` keys that streaming inference does not produce, and fails with `'world_points_from_depth'`.

`--model_path` is always the neural network checkpoint (`.pt`). It is **not** an output path.

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

Upstream `--save_glb` is currently unreliable for streaming outputs (it expects
`world_points` / `world_points_from_depth`, which are not in the NPZs). Use
`--save_predictions` and convert with `npz_to_glb`:

```bash
source /workspace/activate_lingbot.sh
cd /workspace/lingbot-map

# 1) Inference → NPZ directory
# Long videos (>~300–1000 frames): use windowed mode + keyframes.
# Without --fps, every video frame is used (kitchen.mp4 ≈ 3600 frames).
python demo_render/batch_demo.py \
  --video_path /workspace/inputs/clip.mp4 \
  --output_folder /workspace/outputs/clip \
  --model_path /workspace/checkpoints/lingbot-map.pt \
  --config demo_render/config/indoor.yaml \
  --mode windowed \
  --window_size 128 \
  --keyframe_interval 10 \
  --overlap_keyframes 8 \
  --mask_sky \
  --save_predictions \
  --no_render

# Optional: subsample first with --fps 10 to cut frame count / runtime.

# 2) NPZ → GLB  (scene name = video basename without extension)
cd /workspace/lingbot-map/demo_render
python -m interactive_viewer.npz_to_glb \
  --input_dir /workspace/outputs/clip/clip \
  --output /workspace/outputs/clip/clip.glb \
  --downsample 2 \
  --max_points 5000000
```

`--model_path` is always the `.pt` checkpoint, never a `.glb` path.

Also useful:

- Drop `--no_render` (with the render stack installed) to also write a flythrough MP4
- Long sequences: `--mode windowed --window_size 128 --overlap_keyframes 8 --keyframe_interval 10`

## Optimizing capture → reconstruction

Rough length guide (after any `--fps` / stride sampling — i.e. frames the model actually sees):

| | Frames in | Typical clip |
|--|--|--|
| **Short** | up to ~300 | Single room, under ~30–60 s at 5–10 FPS |
| **Medium** | ~300–2000 | Apartment loop / multi-room, a few minutes |
| **Long** | ≳ ~2000–3000+ | Extended walkthroughs (your kitchen run was ~3600) |

Upstream trains with a ~320-keyframe RoPE / KV-cache comfort zone. Past that, use keyframes and/or windowed mode. Default streaming also hits issues around ~1024 frames without windowing.

1. **Capture** — Slow, steady motion with good overlap; avoid motion blur and big auto-exposure jumps. Prefer solid coverage over a very long shaky clip. Indoors: enough light, less blown-out windows.
2. **Fewer, better frames** — Use `--fps 5–10` (or a larger image/keyframe stride) instead of every frame of a long phone video. Less drift, less junk depth, faster runs.
3. **Windowed + keyframes** — For **medium/long** clips (≳ ~300–500 frames, and especially ≳ ~1000), use `--mode windowed` with `--keyframe_interval` (e.g. 8–10) so the KV cache stays near the ~320-slot training range. Short clips can stay in streaming mode.
4. **Checkpoint** — Default `lingbot-map` is fine for short/medium scenes. For **long** indoor walks or large spaces (multi-thousand frames / multi-minute coverage), try `lingbot-map-long` if the balanced checkpoint looks soft or unstable.
5. **TSDF** — Once NPZs look good: smaller `--voxel_length` for detail (more RAM), larger for smoother walls; raise `--conf_percentile` to drop noisy depth; don’t integrate every frame (`--stride`).

## NPZ → vertex-colored mesh (local, for Blender)

Open3D TSDF fusion turns depth + poses into a **triangle mesh with vertex colors** (real surfaces, not points). Run this on your Mac after downloading the NPZs.

Open3D needs **Python 3.11** (your system 3.13 won’t have wheels on Intel Mac):

```bash
cd ~/Documents/Projects/lingbot-map-runpod
python3.11 -m venv .venv-mesh
.venv-mesh/bin/pip install -r requirements-mesh.txt

.venv-mesh/bin/python npz_to_mesh.py \
  --input_dir kitchen \
  --output kitchen_mesh.ply \
  --stride 15 \
  --simplify_triangles 200000
```

Then in Blender: **File → Import → Stanford (.ply)**. For leftover coplanar wall verts, use **Mesh → Clean Up → Limited Dissolve**.

Useful knobs:

| Flag | Effect |
|------|--------|
| `--stride 15` | Use every 15th frame (lower = slower, denser) |
| `--voxel_length 0.015` | Smaller = more detail / more RAM |
| `--depth_trunc 3.5` | Drop far depths (model units, not meters) |
| `--conf_percentile 20` | Drop low-confidence depth pixels |
| `--simplify_triangles 200000` | Quadric-decimate toward this triangle count (`0` = off) |
| `--merge_vertices_dist 0.002` | Weld near-duplicate verts after cleanup |
| `--no_cleanup` | Skip duplicate/degenerate/non-manifold cleanup |

Cleanup runs by default (duplicates, degenerates, non-manifold edges). Simplify is opt-in via `--simplify_triangles`.

This is **not** a UV-textured mesh yet — colors live on vertices. Good enough for Blender viewing and cleanup.

## NPZ → Gaussian splat PLY (initialized, not trained)

The NPZs are **not** Gaussians. `npz_to_splat.py` unprojects depth into points and writes each as a small isotropic Gaussian in the standard 3DGS PLY layout (viewable in SuperSplat etc.). This is a quick preview — **not** the quality of training 3DGS on the images.

```bash
.venv-mesh/bin/python npz_to_splat.py \
  --input_dir kitchen \
  --output kitchen_splat.ply \
  --stride 15 \
  --max_points 2000000
```

Open `kitchen_splat.ply` in a **splat viewer** (SuperSplat, Spark, …). Blender will not treat it as a normal mesh.

## Cost tip

Terminate the pod when idle; keep the Network Volume. Path B is the cheap day-to-day loop: spin up → activate → batch_demo → shut down.
