#!/usr/bin/env python3
"""Convert lingbot-map NPZs into an *initialized* 3D Gaussian Splat PLY.

Important: the NPZs are depth + poses + RGB, not trained Gaussians. This script
unprojects confident depth pixels into world points and writes each as a small
isotropic Gaussian (INRIA / SuperSplat-compatible PLY). That is useful for
quick viewing — it is **not** the same quality as training 3DGS on the images.

Example:

  .venv-mesh/bin/python npz_to_splat.py \\
    --input_dir kitchen \\
    --output kitchen_splat.ply \\
    --stride 15 \\
    --max_points 2000000

View in SuperSplat, Spark, or other 3DGS viewers (not as a mesh in Blender).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Degree-0 SH basis constant used by the original 3DGS code.
_SH_C0 = 0.28209479177387814


def list_frame_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("frame_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.npz files in {input_dir}")
    return paths


def load_frame(path: Path) -> dict:
    with np.load(path) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]

        images = np.asarray(data["images"], dtype=np.float32)
        if images.ndim == 3 and images.shape[0] == 3:
            images = np.transpose(images, (1, 2, 0))
        images = np.clip(images, 0.0, 1.0)

        intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(data["extrinsic"], dtype=np.float64)

        conf = None
        for key in ("depth_conf", "confidence"):
            if key in data.files:
                conf = np.asarray(data[key], dtype=np.float32)
                break

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = extrinsic
    c2w = np.linalg.inv(w2c)

    return {
        "depth": depth,
        "image": images,
        "K": intrinsic,
        "c2w": c2w,
        "conf": conf,
    }


def unproject_frame(
    frame: dict,
    *,
    downsample: int,
    depth_trunc: float,
    conf_percentile: float,
    scale_pixels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world XYZ (N,3), RGB (N,3) in [0,1], and linear scales (N,)."""
    depth = frame["depth"]
    image = frame["image"]
    K = frame["K"]
    c2w = frame["c2w"]
    conf = frame["conf"]
    ds = max(int(downsample), 1)

    depth_ds = depth[::ds, ::ds]
    image_ds = image[::ds, ::ds]
    h, w = depth_ds.shape

    valid = (depth_ds > 1e-6) & (depth_ds < depth_trunc)
    if conf is not None and conf_percentile > 0:
        conf_ds = conf[::ds, ::ds]
        thr = float(np.percentile(conf, conf_percentile))
        valid &= conf_ds >= thr

    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    ys, xs = np.where(valid)
    z = depth_ds[ys, xs].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Pixel centers in the full-res intrinsic, accounting for downsample.
    u = (xs.astype(np.float64) * ds) + (ds - 1) * 0.5
    v = (ys.astype(np.float64) * ds) + (ds - 1) * 0.5
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    ones = np.ones_like(z)
    cam = np.stack([x_cam, y_cam, z, ones], axis=0)  # 4,N
    world = (c2w @ cam)[:3].T.astype(np.float32)

    rgb = image_ds[ys, xs].astype(np.float32)
    # Isotropic size ≈ scale_pixels of a pixel footprint at that depth.
    scale = (z / fx * scale_pixels * ds).astype(np.float32)
    scale = np.clip(scale, 1e-5, None)
    return world, rgb, scale


def collect_gaussians(
    frame_paths: list[Path],
    *,
    stride: int,
    max_frames: int | None,
    downsample: int,
    depth_trunc: float,
    conf_percentile: float,
    scale_pixels: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = frame_paths[:: max(stride, 1)]
    if max_frames is not None:
        selected = selected[:max_frames]

    xyzs: list[np.ndarray] = []
    rgbs: list[np.ndarray] = []
    scales: list[np.ndarray] = []

    print(f"Unprojecting {len(selected)} / {len(frame_paths)} frames "
          f"(stride={stride}, downsample={downsample})")

    for path in tqdm(selected, desc="Unproject", unit="frame"):
        frame = load_frame(path)
        xyz, rgb, scale = unproject_frame(
            frame,
            downsample=downsample,
            depth_trunc=depth_trunc,
            conf_percentile=conf_percentile,
            scale_pixels=scale_pixels,
        )
        if len(xyz) == 0:
            continue
        xyzs.append(xyz)
        rgbs.append(rgb)
        scales.append(scale)

    if not xyzs:
        raise SystemExit("No valid points after filtering. Relax --conf_percentile / --depth_trunc.")

    xyz = np.concatenate(xyzs, axis=0)
    rgb = np.concatenate(rgbs, axis=0)
    scale = np.concatenate(scales, axis=0)
    print(f"Total points before cap: {len(xyz):,}")

    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        idx.sort()
        xyz, rgb, scale = xyz[idx], rgb[idx], scale[idx]
        print(f"Downsampled to {len(xyz):,} points")

    return xyz, rgb, scale


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    return ((rgb - 0.5) / _SH_C0).astype(np.float32)


def inverse_sigmoid(x: float) -> float:
    x = min(max(x, 1e-4), 1.0 - 1e-4)
    return float(np.log(x / (1.0 - x)))


def write_gaussian_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    scale_lin: np.ndarray,
    *,
    opacity: float = 0.9,
) -> None:
    """Write a degree-0 SH 3DGS PLY (no f_rest_*). Compatible with most viewers."""
    n = len(xyz)
    f_dc = rgb_to_sh_dc(rgb)
    opac = np.full((n,), inverse_sigmoid(opacity), dtype=np.float32)
    log_scale = np.log(scale_lin.astype(np.float32))
    # Identity quaternion (w, x, y, z)
    rot = np.zeros((n, 4), dtype=np.float32)
    rot[:, 0] = 1.0
    normals = np.zeros((n, 3), dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ]
    )

    # Pack as structured array matching header order.
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    verts = np.empty(n, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["nx"], verts["ny"], verts["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    verts["opacity"] = opac
    verts["scale_0"] = verts["scale_1"] = verts["scale_2"] = log_scale
    verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"] = (
        rot[:, 0],
        rot[:, 1],
        rot[:, 2],
        rot[:, 3],
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Wrote {path} ({n:,} Gaussians, {size_mb:.1f} MB)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output .ply (3DGS attribute layout)")
    p.add_argument("--stride", type=int, default=15, help="Use every Nth frame")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--downsample", type=int, default=2, help="Spatial stride within each depth map")
    p.add_argument("--depth_trunc", type=float, default=3.5)
    p.add_argument("--conf_percentile", type=float, default=20.0)
    p.add_argument(
        "--scale_pixels",
        type=float,
        default=1.5,
        help="Gaussian size as multiples of a pixel footprint at that depth",
    )
    p.add_argument("--max_points", type=int, default=2_000_000)
    p.add_argument("--opacity", type=float, default=0.9, help="Linear opacity before logit encoding")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".ply":
        raise SystemExit("Output must be a .ply file (3DGS viewers expect this layout).")
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    frames = list_frame_paths(input_dir)
    xyz, rgb, scale = collect_gaussians(
        frames,
        stride=args.stride,
        max_frames=args.max_frames,
        downsample=args.downsample,
        depth_trunc=args.depth_trunc,
        conf_percentile=args.conf_percentile,
        scale_pixels=args.scale_pixels,
        max_points=args.max_points,
    )
    write_gaussian_ply(output, xyz, rgb, scale, opacity=args.opacity)
    print(
        "Note: this is an initialized splat from depth, not a trained 3DGS model. "
        "Open in a Gaussian splat viewer (e.g. SuperSplat), not as a Blender mesh."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
