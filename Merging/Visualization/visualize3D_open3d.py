"""Interactive 3D Gaussian visualizer using Open3D.

Usage:
    # vis_folder mode
    python visualize3D_open3d.py \\
        --dataset_path /path/to/3DSSG20 \\
        --scene <scan_id> \\
        --vis_folder output/vis/3rscan_10

    # predictions mode
    python visualize3D_open3d.py \\
        --dataset_path /path/to/3DSSG20 \\
        --scene <scan_id> \\
        --predictions output/scannet/predictions_gaussian_*.pkl
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from classes import obj_colors, valid_class

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path",      type=str, required=True)
parser.add_argument("--scene",             type=str, required=True)
# vis_folder mode
parser.add_argument("--vis_folder",        type=str, default=None,
                    help="Folder containing per-frame obj.pkl (vis_folder mode).")
parser.add_argument("--frame",             type=int, default=-1,
                    help="Frame index into obj.pkl (-1 = last). vis_folder mode only.")
# predictions mode
parser.add_argument("--predictions",       type=str, default=None,
                    help="Path to predictions_*.pkl (predictions mode).")
parser.add_argument("--label_categories",  type=str, choices=["scannet", "replica"],
                    default="scannet",
                    help="Label set used to name classes. predictions mode only.")
# shared
parser.add_argument("--sigma",      type=float, default=1.0,
                    help="Ellipsoid scale in standard deviations.")
parser.add_argument("--sphere_res", type=int,   default=20,
                    help="Sphere tessellation resolution.")
args = parser.parse_args()

if args.vis_folder is None and args.predictions is None:
    sys.exit("error: one of --vis_folder or --predictions is required.")
if args.vis_folder and args.predictions:
    sys.exit("error: --vis_folder and --predictions are mutually exclusive.")

# ── Load scene mesh ───────────────────────────────────────────────────────────

if "Replica" in args.dataset_path:
    mesh_path = f"{args.dataset_path}/{args.scene}/mesh.ply"
else:
    mesh_path = f"{args.dataset_path}/../data/3RScan/{args.scene}/mesh.refined.v2.obj"

scene_mesh = o3d.io.read_triangle_mesh(mesh_path)
scene_mesh.compute_vertex_normals()
if not scene_mesh.has_vertex_colors():
    scene_mesh.paint_uniform_color([0.7, 0.7, 0.7])

# ── Load object data ──────────────────────────────────────────────────────────

def _color_for_index(idx: int) -> np.ndarray:
    """Return an RGB [0,1] color cycling through obj_colors by index."""
    c = np.array(obj_colors[idx % len(obj_colors)], dtype=float)
    return c / 255.0 if c.max() > 1.0 else c

if args.vis_folder:
    # ── vis_folder mode ───────────────────────────────────────────────────────
    dataset = "ReplicaSSG" if "Replica" in args.dataset_path else "3RScan"
    pkl_dir = f"{args.vis_folder}/{dataset}/{args.scene}"
    with open(f"{pkl_dir}/obj.pkl", "rb") as f:
        obj = pickle.load(f)
    frame_data = obj[args.frame]
    raw_classes = frame_data["classes"]
    means = frame_data["means"]
    covs  = frame_data["covs"]

    # Map class index → color using valid_class/obj_colors
    objects = []
    for i, cls in enumerate(raw_classes):
        if cls not in valid_class:
            continue
        color = np.array(obj_colors[valid_class.index(cls)], dtype=float)
        if color.max() > 1.0:
            color /= 255.0
        objects.append((means[i], covs[i], color, str(cls)))

    title = f"FROSS 3D SG — {args.scene} — frame {args.frame}"
    print(f"vis_folder mode: frame {args.frame}, {len(objects)} objects")

else:
    # ── predictions mode ──────────────────────────────────────────────────────
    with open(args.predictions, "rb") as f:
        predictions = pickle.load(f)

    if args.scene not in predictions:
        sys.exit(f"error: scan '{args.scene}' not found in {args.predictions}.")

    pred = predictions[args.scene]
    cls_probs = np.array(pred["cls"])    # (N, C)
    means_arr = np.array(pred["mean"])   # (N, 3)
    covs_arr  = np.array(pred["cov"])    # (N, 3, 3)
    class_indices = cls_probs.argmax(axis=1)  # (N,)

    # Load class names for readable labels
    dataset_path = Path(args.dataset_path)
    if args.label_categories == "scannet":
        mapping_path = dataset_path / "3DSSG_subset" / "3dssg_to_scannet.json"
        class_name_key = "ScanNet_list"
    else:
        mapping_path = dataset_path / "ReplicaSSG" / "replica_to_visual_genome.json"
        class_name_key = "VisualGenome_list"
    with open(mapping_path) as f:
        class_names = json.load(f)[class_name_key]

    objects = []
    for i, cls_idx in enumerate(class_indices):
        label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        color = _color_for_index(int(cls_idx))
        objects.append((means_arr[i], covs_arr[i], color, label))

    title = f"FROSS 3D SG — {args.scene} — final ({len(objects)} objects)"
    print(f"predictions mode: {len(objects)} objects in '{args.scene}'")
    for i, (_, _, _, label) in enumerate(objects):
        print(f"  [{i}] {label}")

# ── Build ellipsoid meshes ────────────────────────────────────────────────────

geometries = [scene_mesh]

for mean, cov, color, label in objects:
    mean = np.asarray(mean, dtype=float)
    cov  = np.asarray(cov,  dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    radii = args.sigma * np.sqrt(eigenvalues)

    ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=args.sphere_res)
    ellipsoid.paint_uniform_color(color)
    ellipsoid.compute_vertex_normals()

    transform          = np.eye(4)
    transform[:3, :3]  = eigenvectors @ np.diag(radii)
    transform[:3, 3]   = mean
    ellipsoid.transform(transform)
    geometries.append(ellipsoid)

print(f"Rendering {len(geometries) - 1} ellipsoids + scene mesh")

# ── Draw ──────────────────────────────────────────────────────────────────────

o3d.visualization.draw_geometries(
    geometries,
    window_name=title,
    width=1280,
    height=720,
    mesh_show_back_face=True,
)
