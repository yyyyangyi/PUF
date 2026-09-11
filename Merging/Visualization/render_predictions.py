"""Render FROSS Gaussians from a fixed Open3D camera view to a PNG.

Usage:
    python render_predictions.py \\
        --dataset_path /path/to/3DSSG20 \\
        --scene <scan_id> \\
        --predictions output/scannet/predictions_gaussian_*.pkl \\
        --camera view.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from classes import obj_colors

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path",     type=str,   required=True)
parser.add_argument("--scene",            type=str,   required=True)
parser.add_argument("--predictions",      type=str,   required=True)
parser.add_argument("--camera",           type=str,   required=True,
                    help="Path to Open3D PinholeCameraParameters .json file.")
parser.add_argument("--output",           type=str,   default=None,
                    help="Output PNG path (default: <scene>_render.png).")
parser.add_argument("--label_categories", type=str,   default="scannet",
                    choices=["scannet", "replica"])
parser.add_argument("--sigma",            type=float, default=1.0,
                    help="Ellipsoid scale in standard deviations.")
parser.add_argument("--sphere_res",       type=int,   default=20)
parser.add_argument("--alpha",            type=float, default=0.6,
                    help="Ellipsoid opacity (0=transparent, 1=solid).")
args = parser.parse_args()

output_path = args.output or f"{args.scene}_render.png"

# ── Load camera parameters ────────────────────────────────────────────────────

cam_params = o3d.io.read_pinhole_camera_parameters(args.camera)
intrinsic  = cam_params.intrinsic   # PinholeCameraIntrinsic
extrinsic  = cam_params.extrinsic   # (4, 4) ndarray

width  = intrinsic.width
height = intrinsic.height
print(f"Camera: {width}x{height}")

# ── Load class names ──────────────────────────────────────────────────────────

dataset_path = Path(args.dataset_path)
if args.label_categories == "scannet":
    mapping_path   = dataset_path / "3DSSG_subset" / "3dssg_to_scannet.json"
    class_name_key = "ScanNet_list"
else:
    mapping_path   = dataset_path / "ReplicaSSG" / "replica_to_visual_genome.json"
    class_name_key = "VisualGenome_list"
with open(mapping_path) as f:
    class_names = json.load(f)[class_name_key]

# ── Load predictions ──────────────────────────────────────────────────────────

with open(args.predictions, "rb") as f:
    predictions = pickle.load(f)

if args.scene not in predictions:
    sys.exit(f"error: scan '{args.scene}' not found in {args.predictions}.")

pred          = predictions[args.scene]
cls_probs     = np.array(pred["cls"])    # (N, C)
means_arr     = np.array(pred["mean"])   # (N, 3)
covs_arr      = np.array(pred["cov"])    # (N, 3, 3)
class_indices = cls_probs.argmax(axis=1)

print(f"Scene '{args.scene}': {len(means_arr)} objects")
for i, ci in enumerate(class_indices):
    label = class_names[ci] if ci < len(class_names) else str(ci)
    print(f"  [{i}] {label}")

# ── Load scene mesh ───────────────────────────────────────────────────────────

if "Replica" in args.dataset_path:
    mesh_path = dataset_path / args.scene / "mesh.ply"
else:
    mesh_path = dataset_path / ".." / "data" / "3RScan" / args.scene / "mesh.refined.v2.obj"

scene_mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
scene_mesh.compute_vertex_normals()

# ── Build ellipsoid meshes ────────────────────────────────────────────────────

def _color_for_index(idx: int) -> np.ndarray:
    c = np.array(obj_colors[idx % len(obj_colors)], dtype=float)
    return c / 255.0 if c.max() > 1.0 else c

ellipsoids = []
for i, cls_idx in enumerate(class_indices):
    mean = means_arr[i].astype(float)
    cov  = covs_arr[i].astype(float)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    radii = args.sigma * np.sqrt(eigenvalues)

    ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=args.sphere_res)
    ellipsoid.paint_uniform_color(_color_for_index(int(cls_idx)))
    ellipsoid.compute_vertex_normals()

    transform          = np.eye(4)
    transform[:3, :3]  = eigenvectors @ np.diag(radii)
    transform[:3, 3]   = mean
    ellipsoid.transform(transform)
    ellipsoids.append((ellipsoid, _color_for_index(int(cls_idx))))

print(f"Built {len(ellipsoids)} ellipsoids")

# ── Off-screen render ─────────────────────────────────────────────────────────

renderer = rendering.OffscreenRenderer(width, height)
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background

# Scene mesh
mesh_mat        = rendering.MaterialRecord()
mesh_mat.shader = "defaultLit"
if scene_mesh.has_textures():
    mesh_mat.albedo_img = o3d.geometry.Image(np.asarray(scene_mesh.textures[0]))
renderer.scene.add_geometry("scene_mesh", scene_mesh, mesh_mat)

# Ellipsoid
ellipsoid_mat               = rendering.MaterialRecord()
ellipsoid_mat.shader        = "defaultLitTransparency"
ellipsoid_mat.base_color    = [1.0, 1.0, 1.0, args.alpha]  # overridden per object below

for i, (ellipsoid, color) in enumerate(ellipsoids):
    mat             = rendering.MaterialRecord()
    mat.shader      = "defaultLitTransparency"
    mat.base_color  = [float(color[0]), float(color[1]), float(color[2]), args.alpha]
    renderer.scene.add_geometry(f"ellipsoid_{i}", ellipsoid, mat)

renderer.setup_camera(intrinsic, extrinsic)

img = renderer.render_to_image()
o3d.io.write_image(output_path, img)
print(f"Saved render to {output_path}")
