#!/usr/bin/env python3
"""Fuse lingbot-map per-frame NPZs into a vertex-colored triangle mesh (Open3D TSDF).

The NPZs store relative-scale depth + W2C extrinsics + RGB (not metric meters).
Defaults are tuned for kitchen-scale scenes like the bundled example.

Example (from this repo, with the mesh venv):

  .venv-mesh/bin/python npz_to_mesh.py \\
    --input_dir kitchen \\
    --output kitchen_mesh.ply \\
    --stride 15 \\
    --simplify_triangles 200000

Open in Blender: File → Import → Stanford (.ply) or glTF (.glb).
For coplanar walls, Blender Limited Dissolve often finishes what quadric simplify starts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "open3d is not installed in this Python.\n"
            "Use the project mesh venv (Python 3.11):\n"
            "  python3.11 -m venv .venv-mesh\n"
            "  .venv-mesh/bin/pip install -r requirements-mesh.txt\n"
            "  .venv-mesh/bin/python npz_to_mesh.py ..."
        ) from exc
    return o3d


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
            images = np.transpose(images, (1, 2, 0))  # CHW → HWC
        images = np.clip(images, 0.0, 1.0)

        intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(data["extrinsic"], dtype=np.float64)  # W2C 3x4

        conf = None
        for key in ("depth_conf", "confidence"):
            if key in data.files:
                conf = np.asarray(data[key], dtype=np.float32)
                break

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = extrinsic

    return {
        "depth": depth,
        "image": images,
        "K": intrinsic,
        "w2c": w2c,
        "conf": conf,
    }


def make_intrinsic(o3d, K: np.ndarray, width: int, height: int):
    return o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )


def integrate_volume(
    o3d,
    frame_paths: list[Path],
    *,
    stride: int,
    max_frames: int | None,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
    conf_percentile: float,
) -> "o3d.geometry.TriangleMesh":
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    selected = frame_paths[:: max(stride, 1)]
    if max_frames is not None:
        selected = selected[:max_frames]

    print(f"Integrating {len(selected)} / {len(frame_paths)} frames "
          f"(stride={stride}, voxel_length={voxel_length}, sdf_trunc={sdf_trunc})")

    for path in tqdm(selected, desc="TSDF integrate", unit="frame"):
        frame = load_frame(path)
        depth = frame["depth"].copy()
        image = frame["image"]
        conf = frame["conf"]

        if conf is not None and conf_percentile > 0:
            # Keep higher-confidence depths; zero out the rest for Open3D.
            thr = float(np.percentile(conf, conf_percentile))
            depth[conf < thr] = 0.0

        depth[depth <= 0] = 0.0
        depth[depth > depth_trunc] = 0.0

        h, w = depth.shape
        color_u8 = (image * 255.0).astype(np.uint8)
        # Open3D Image expects contiguous uint8 RGB and float32 depth (meters-like units).
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_u8))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,  # already in model units
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = make_intrinsic(o3d, frame["K"], w, h)
        # Open3D integrate() expects world-to-camera extrinsic.
        volume.integrate(rgbd, intrinsic, frame["w2c"])

    print("Extracting triangle mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def export_mesh(o3d, mesh, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in {".ply", ".glb", ".gltf", ".obj"}:
        raise SystemExit(f"Unsupported output format '{suffix}'. Use .ply, .glb, .gltf, or .obj")

    ok = o3d.io.write_triangle_mesh(
        str(output),
        mesh,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    if not ok:
        raise SystemExit(f"Failed to write {output}")

    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    has_color = mesh.has_vertex_colors()
    print(f"Wrote {output} ({n_verts:,} verts, {n_tris:,} tris, vertex_colors={has_color})")


def _mesh_stats(mesh) -> tuple[int, int]:
    return len(mesh.vertices), len(mesh.triangles)


def cleanup_mesh(
    mesh,
    *,
    merge_vertices_dist: float,
) -> "o3d.geometry.TriangleMesh":
    """Remove junk geometry and optionally weld nearby vertices."""
    before_v, before_t = _mesh_stats(mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if merge_vertices_dist > 0:
        mesh.merge_close_vertices(merge_vertices_dist)
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()

    after_v, after_t = _mesh_stats(mesh)
    print(
        f"Cleanup: {before_v:,}→{after_v:,} verts, {before_t:,}→{after_t:,} tris"
        + (f" (merge_dist={merge_vertices_dist})" if merge_vertices_dist > 0 else "")
    )
    return mesh


def simplify_mesh(
    mesh,
    *,
    target_triangles: int,
) -> "o3d.geometry.TriangleMesh":
    """Quadric decimation toward a triangle budget (keeps vertex colors when possible)."""
    before_v, before_t = _mesh_stats(mesh)
    if target_triangles <= 0 or before_t <= target_triangles:
        print(
            f"Simplify skipped "
            f"(tris={before_t:,}, target={target_triangles if target_triangles > 0 else 'off'})"
        )
        return mesh

    # Open3D's quadric decimation preserves attributes better after normals exist.
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_triangles)
    )
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_unreferenced_vertices()
    simplified.compute_vertex_normals()

    after_v, after_t = _mesh_stats(simplified)
    print(f"Simplify: {before_v:,}→{after_v:,} verts, {before_t:,}→{after_t:,} tris "
          f"(target={target_triangles:,})")
    return simplified


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", type=Path, required=True, help="Directory of frame_*.npz files")
    p.add_argument("--output", type=Path, required=True, help="Output mesh path (.ply recommended for Blender)")
    p.add_argument("--stride", type=int, default=15, help="Use every Nth frame (default: 15)")
    p.add_argument("--max_frames", type=int, default=None, help="Optional cap after stride")
    p.add_argument(
        "--voxel_length",
        type=float,
        default=0.015,
        help="TSDF voxel size in model units (smaller = more detail, more RAM)",
    )
    p.add_argument(
        "--sdf_trunc",
        type=float,
        default=0.06,
        help="TSDF truncation distance (≈ 3–5 × voxel_length)",
    )
    p.add_argument(
        "--depth_trunc",
        type=float,
        default=3.5,
        help="Ignore depths farther than this (model units)",
    )
    p.add_argument(
        "--conf_percentile",
        type=float,
        default=20.0,
        help="Drop depths below this confidence percentile (0 disables)",
    )
    p.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Skip duplicate/degenerate/non-manifold cleanup",
    )
    p.add_argument(
        "--merge_vertices_dist",
        type=float,
        default=0.0,
        help="Weld vertices closer than this distance after cleanup (0 disables)",
    )
    p.add_argument(
        "--simplify_triangles",
        type=int,
        default=0,
        help="Quadric-decimate to this many triangles (0 disables). "
             "Example: 200000. For coplanar walls, also try Blender Limited Dissolve.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    o3d = _require_open3d()

    input_dir = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    frames = list_frame_paths(input_dir)
    mesh = integrate_volume(
        o3d,
        frames,
        stride=args.stride,
        max_frames=args.max_frames,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
        conf_percentile=args.conf_percentile,
    )

    if len(mesh.triangles) == 0:
        raise SystemExit(
            "Mesh has 0 triangles. Try a smaller --voxel_length, larger --depth_trunc, "
            "or lower --stride."
        )

    if not args.no_cleanup:
        mesh = cleanup_mesh(mesh, merge_vertices_dist=args.merge_vertices_dist)

    if args.simplify_triangles > 0:
        mesh = simplify_mesh(mesh, target_triangles=args.simplify_triangles)

    if len(mesh.triangles) == 0:
        raise SystemExit("Mesh became empty after cleanup/simplify; relax thresholds.")

    export_mesh(o3d, mesh, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
