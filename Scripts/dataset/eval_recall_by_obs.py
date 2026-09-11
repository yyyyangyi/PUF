"""Evaluate relationship recall broken down by per-triplet 2D observation count.
Produces our papers Fig. 3.

Usage:
    python eval_recall_by_obs.py \\
        --prediction_path output/scannet/predictions_gaussian_*.pkl \\
        --dataset_path /path/to/dataset \\
        --split test
"""
import json
import pickle
import argparse
import os
import pathlib
from collections import defaultdict

import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree
from tqdm import tqdm

# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Relationship Recall@1 broken down by 2D observation count bins.")
parser.add_argument("--prediction_path", type=pathlib.Path, required=True,
                    help="predictions_*.pkl produced by main.py")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="Dataset root (same as main.py --dataset_path)")
parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
parser.add_argument("--label_categories", type=str,
                    choices=["scannet", "replica"], default="scannet")
parser.add_argument("--method", type=str, choices=["bb", "rel_json"], default="bb",
                    help="Source for per-frame observed relations.")
parser.add_argument("--eval_overlap_threshold", type=float, default=0.1,
                    help="Point-cloud overlap threshold for GT/pred matching (default 0.1).")
parser.add_argument("--skip_no_rel_objects", action="store_true",
                    help="Exclude GT objects that appear in no relationship (mirrors evaluate.py).")
parser.add_argument("--use_aligned_ply", action="store_true",
                    help="Use labels.instances.align.annotated.v2.ply instead of the "
                         "unaligned variant (only valid for scannet).")
parser.add_argument("--subset", type=int, default=None,
                    help="Evaluate only the first N scans (for quick testing).")
parser.add_argument("--output", type=pathlib.Path, default=None,
                    help="Optional path to save results as JSON.")
args = parser.parse_args()

# ── Path configuration ─────────────────────────────────────────────────────────

dataset_path   = pathlib.Path(args.dataset_path)
scan_split_name = "validation" if args.split == "val" else args.split

if args.label_categories == "scannet":
    scan_split_path  = dataset_path / "3DSSG_subset" / f"{scan_split_name}_scans.txt"
    sg_ann_path      = dataset_path / "3DSSG_subset" / "relationships20.json"
    object_json_path = dataset_path / "objects.json"
    mapping_path     = dataset_path / "3DSSG_subset" / "3dssg_to_scannet.json"
    data_folder      = dataset_path / "../data/3RScan"
    sg2d_dir         = dataset_path / "2DSG20"
    OBJ_CLASS_NAME   = "ScanNet_list"
    REL_CLASS_NAME   = "ScanNet_rel"
    MAPPING_NAME     = "3DSSG2NYUv2"
    USE_VISIBILITY   = True
else:
    scan_split_path  = dataset_path / "ReplicaSSG" / f"{scan_split_name}_scans.txt"
    sg_ann_path      = dataset_path / "ReplicaSSG" / "relationships.json"
    object_json_path = dataset_path / "ReplicaSSG" / "objects.json"
    mapping_path     = dataset_path / "ReplicaSSG" / "replica_to_visual_genome.json"
    data_folder      = dataset_path / "data"
    sg2d_dir         = dataset_path / "2DSG"
    OBJ_CLASS_NAME   = "VisualGenome_list"
    REL_CLASS_NAME   = "VisualGenome_rel"
    MAPPING_NAME     = "Replica2VisualGenome"
    USE_VISIBILITY   = False

# ── Load metadata ──────────────────────────────────────────────────────────────

with open(mapping_path) as f:
    class_mapping = json.load(f)
with open(object_json_path) as f:
    object_data = json.load(f)
with open(sg_ann_path) as f:
    sg_ann = json.load(f)
with open(scan_split_path) as f:
    scan_ids = [line.strip() for line in f]

obj_classes     = class_mapping[OBJ_CLASS_NAME]
rel_class_names = class_mapping[REL_CLASS_NAME]
scan_id_set     = set(scan_ids)

with open(args.prediction_path, "rb") as f:
    predictions = pickle.load(f)

# ── Load 2D GT annotations (rel_json method only) ─────────────────────────────

if args.method == "rel_json":
    with open(sg2d_dir / "rel.json") as f:
        rel_by_imgid = json.load(f)[args.split]   # {str(img_id): [(ls, lo, r), ...]}
    with open(sg2d_dir / f"{args.split}.json") as f:
        coco_data = json.load(f)
    scan_frame_to_imgid = defaultdict(dict)
    for img in coco_data["images"]:
        fname_noext = img["file_name"][:-4]
        sep = fname_noext.rfind("-")
        scan_frame_to_imgid[fname_noext[:sep]][int(fname_noext[sep + 1:])] = img["id"]

obj_in_rel  = {}  # scan_id → set of obj_ids that appear in any relationship
scan2obj_rel = {}
OBJID2IDX    = {}

for scan in sg_ann["scans"]:
    if scan["scan"] not in scan_id_set:
        continue
    obj_in_rel[scan["scan"]] = set()
    for rel in scan["relationships"]:
        s, o = rel[0], rel[1]
        obj_in_rel[scan["scan"]].add(s)
        obj_in_rel[scan["scan"]].add(o)

for scan in object_data["scans"]:
    if scan["scan"] not in scan_id_set:
        continue
    sid = scan["scan"]
    scan2obj_rel[sid] = {"obj_id": [], "obj_cls": [], "rel_edge": [], "rel_cls": []}
    OBJID2IDX[sid] = {}
    for obj in scan["objects"]:
        oid = int(obj["id"])
        if args.skip_no_rel_objects and oid not in obj_in_rel.get(sid, set()):
            continue
        if args.label_categories == "replica" and obj["label"] not in class_mapping[MAPPING_NAME]:
            continue
        label = class_mapping[MAPPING_NAME].get(obj["label"])
        if label not in obj_classes:
            continue
        cls_idx = obj_classes.index(label)
        scan2obj_rel[sid]["obj_id"].append(oid)
        scan2obj_rel[sid]["obj_cls"].append(cls_idx)
        OBJID2IDX[sid][oid] = len(scan2obj_rel[sid]["obj_cls"]) - 1

for scan in sg_ann["scans"]:
    if scan["scan"] not in scan_id_set:
        continue
    sid = scan["scan"]
    for rel in scan["relationships"]:
        s, o = int(rel[0]), int(rel[1])
        if s not in OBJID2IDX[sid] or o not in OBJID2IDX[sid]:
            continue
        cls_idx = class_mapping[REL_CLASS_NAME].index(rel[3])
        scan2obj_rel[sid]["rel_edge"].append((OBJID2IDX[sid][s], OBJID2IDX[sid][o]))
        scan2obj_rel[sid]["rel_cls"].append([cls_idx])

# ── Phase 1: Compute per-triplet observation counts ───────────────────────────

print(f"Computing observation counts [{args.method} method] ...")

# Build obj_id → class index for every scan (needed for visibility filter).
OBJ2CLASS = {}
for scan in object_data["scans"]:
    sid = scan["scan"]
    OBJ2CLASS[sid] = {}
    for obj in scan["objects"]:
        oid    = int(obj["id"])
        mapped = class_mapping[MAPPING_NAME].get(obj["label"])
        OBJ2CLASS[sid][oid] = obj_classes.index(mapped) if (mapped and mapped in obj_classes) else -1

# Build 3D GT relation sets keyed by scan.
SCAN2REL3D = {}
for scan in sg_ann["scans"]:
    sid = scan["scan"]
    if sid not in scan_id_set:
        continue
    SCAN2REL3D[sid] = set((int(r[0]), int(r[1]), int(r[2])) for r in scan["relationships"])

eval_subset = args.subset if args.subset else len(scan_ids)
obs_counts  = {}  # scan_id → {(s_id, o_id, rel_idx): count}

for scan_id in tqdm(scan_ids[:eval_subset], desc="obs counts"):
    if scan_id not in SCAN2REL3D or scan_id not in OBJ2CLASS:
        continue
    gt_rel_set  = SCAN2REL3D[scan_id]
    obj_cls_map = OBJ2CLASS[scan_id]

    seq_dir  = data_folder / scan_id / "sequence"
    bb_files = sorted(f for f in os.listdir(seq_dir) if f.endswith(".bb.txt"))

    if USE_VISIBILITY:
        vis_files = sorted(f for f in os.listdir(seq_dir) if f.endswith(".visibility.txt"))

    local_counts = defaultdict(int)

    for fi, bb_file in enumerate(bb_files):
        with open(seq_dir / bb_file) as fh:
            objects = [line.split() for line in fh]

        if USE_VISIBILITY:
            with open(seq_dir / vis_files[fi]) as fh:
                vis_values = [list(map(float, line.split())) for line in fh]

        obj_id_in_img = []
        visible = set()
        for idx, obj in enumerate(objects):
            oid = int(obj[0])
            if USE_VISIBILITY:
                if idx >= len(vis_values) or vis_values[idx][3] * vis_values[idx][6] < 0.1:
                    continue
            if obj_cls_map.get(oid, -1) == -1:
                continue
            x, y = int(obj[1]), int(obj[2])
            w, h  = int(obj[3]) - x, int(obj[4]) - y
            if w * h < 1:
                continue
            visible.add(oid)
            obj_id_in_img.append(oid)

        if args.method == "bb":
            frame_rels = {(s, o, r) for s, o, r in gt_rel_set if s in visible and o in visible}
        else:  # rel_json
            frame_idx = int(bb_file[6:-7])
            img_id    = scan_frame_to_imgid.get(scan_id, {}).get(frame_idx)
            frame_rels = set()
            if img_id is not None:
                for ls, lo, rel_idx in rel_by_imgid.get(str(img_id), []):
                    if ls < len(obj_id_in_img) and lo < len(obj_id_in_img):
                        frame_rels.add((obj_id_in_img[ls], obj_id_in_img[lo], rel_idx))
                frame_rels &= gt_rel_set

        for rel in frame_rels:
            local_counts[rel] += 1

    obs_counts[scan_id] = local_counts

# ── Phase 2: Evaluate relationship recall ──────────────────────────────────────

print("Evaluating relationship recall ...")

# records: list of (obs_count, rank)   — one entry per GT relationship edge
records = []

for scan_id in tqdm(scan_ids[:eval_subset], desc="eval"):
    if scan_id not in scan2obj_rel or scan_id not in predictions:
        continue

    node_gt        = np.array(scan2obj_rel[scan_id]["obj_cls"])   # (N_gt,)
    edge_gt        = np.array(scan2obj_rel[scan_id]["rel_cls"])   # (E_gt, 1)
    edge_index_gt  = np.array(scan2obj_rel[scan_id]["rel_edge"])  # (E_gt, 2)
    obj_ids_gt     = scan2obj_rel[scan_id]["obj_id"]              # [N_gt] 3D obj ids

    if len(node_gt) == 0 or len(edge_gt) == 0:
        continue

    if args.label_categories == "scannet":
        ply_suffix = ".align" if args.use_aligned_ply else ""
        gt_mesh_path = (data_folder / scan_id /
                        f"labels.instances{ply_suffix}.annotated.v2.ply")
    else:
        gt_mesh_path = (data_folder / scan_id /
                        "labels.instances.annotated.v2.ply")

    gt_mesh      = PlyData.read(str(gt_mesh_path))
    gt_points    = np.stack([gt_mesh["vertex"]["x"],
                              gt_mesh["vertex"]["y"],
                              gt_mesh["vertex"]["z"]], axis=1)
    gt_point_ids = gt_mesh["vertex"]["objectId"]

    valid_ids = set(OBJID2IDX[scan_id].keys())
    mask      = np.isin(gt_point_ids, list(valid_ids))
    gt_points    = gt_points[mask]
    gt_point_ids = gt_point_ids[mask]

    pred           = predictions[scan_id]
    node_pred      = np.array(pred["cls"])        # (N_pred, C)
    edge_pred      = np.array(pred["edge_cls"])   # (E_pred, K)
    edge_index_pred = np.array(pred["edge_index"]) # (2, E_pred)

    # ── Match GT objects to predicted objects (same as evaluate.py) ───────────
    gt2pred = -np.ones((2, len(node_gt)), dtype=int)
    gt2pred[0] = np.arange(len(node_gt))

    if len(node_pred) > 0:
        overlap_count = np.zeros((len(node_gt), len(node_pred)))
        gt_kdtree = KDTree(gt_points)
        for pred_idx, seg in enumerate(pred["pcd"]):
            pred_num_points = len(seg)
            distances, indices = gt_kdtree.query(
                seg, distance_upper_bound=args.eval_overlap_threshold)
            matched_gt_idx = np.array([
                OBJID2IDX[scan_id][gt_point_ids[i]]
                for i in indices[indices != gt_kdtree.n]
            ])
            for gt_idx in range(len(node_gt)):
                overlap_count[gt_idx, pred_idx] = np.count_nonzero(matched_gt_idx == gt_idx)
            overlap_pct    = overlap_count[:, pred_idx] / pred_num_points
            sorted_gt_idx  = np.flip(np.argsort(overlap_count[:, pred_idx], kind="stable"))
            max_gt_idx     = sorted_gt_idx[0]
            second_gt_idx  = sorted_gt_idx[1]
            if (overlap_pct[max_gt_idx] < 0.5 or
                    overlap_pct[second_gt_idx] / overlap_pct[max_gt_idx] > 0.75):
                overlap_count[:, pred_idx] = 0
            else:
                overlap_count[np.arange(len(node_gt)) != max_gt_idx, pred_idx] = 0

        for gt_idx in range(len(node_gt)):
            max_pred_idx = np.argmax(overlap_count[gt_idx])
            if overlap_count[gt_idx, max_pred_idx] > 0:
                gt2pred[1, gt_idx] = max_pred_idx

    gt2pred_map = {gt2pred[0, i].item(): gt2pred[1, i].item()
                   for i in range(gt2pred.shape[1])}

    edge_index_pred_list = edge_index_pred.transpose().tolist()
    scan_obs = obs_counts.get(scan_id, {})

    # ── Per-edge relationship recall ──────────────────────────────────────────
    for i in range(len(edge_gt)):
        gt_rel     = edge_gt[i][0]
        sub_gt_idx = edge_index_gt[i, 0].item()
        obj_gt_idx = edge_index_gt[i, 1].item()

        # Look up observation count for this GT triplet.
        sub_obj_id = obj_ids_gt[sub_gt_idx]
        obj_obj_id = obj_ids_gt[obj_gt_idx]
        obs_count  = scan_obs.get((sub_obj_id, obj_obj_id, gt_rel), 0)

        sub_pred_idx = gt2pred_map.get(sub_gt_idx, -1)
        obj_pred_idx = gt2pred_map.get(obj_gt_idx, -1)

        if [sub_pred_idx, obj_pred_idx] not in edge_index_pred_list:
            records.append((obs_count, 99999))
            continue

        edge_pred_idx = edge_index_pred_list.index([sub_pred_idx, obj_pred_idx])
        pred_rel  = edge_pred[edge_pred_idx]
        pred_sub  = node_pred[sub_pred_idx]
        pred_obj  = node_pred[obj_pred_idx]

        gt_sub = node_gt[sub_gt_idx]
        gt_obj = node_gt[obj_gt_idx]

        so_preds    = np.einsum("n,m->nm", pred_sub, pred_obj)
        conf_matrix = np.einsum("nm,k->nmk", so_preds, pred_rel)
        _, cls_n, rel_k = conf_matrix.shape
        conf_flat   = conf_matrix.flatten()
        sorted_args = np.flip(np.argsort(conf_flat, kind="stable"))
        gt_index    = (gt_sub * cls_n + gt_obj) * rel_k + gt_rel
        rank        = int(np.nonzero(sorted_args == gt_index)[0].item())

        records.append((obs_count, rank))

# ── Phase 3: Bin by observation count and report ──────────────────────────────

BINS   = [0, 1, 2, 5, 10, 20, 50, np.inf]
LABELS = ["0", "1", "2-4", "5-9", "10-19", "20-49", "50+"]

SEP = "=" * 62
sep = "-" * 62

print(f"\n{SEP}")
print(f"RELATIONSHIP RECALL@1 BY OBSERVATION COUNT BIN")
print(f"  Prediction : {args.prediction_path.name}")
print(f"  Split      : {args.split}  |  Method: {args.method}  |  Scans: {eval_subset}")
print(f"  Total GT relationships evaluated: {len(records)}")
print(SEP)
print(f"  {'Bin':>8}  {'Count':>7}  {'Recall@1':>9}  {'Hist':}")
print(f"  {sep}")

bin_results = {}
for i, label in enumerate(LABELS):
    lo, hi = BINS[i], BINS[i + 1]
    bin_records = [(obs, rank) for obs, rank in records if lo <= obs < hi]
    n           = len(bin_records)
    recall1     = sum(1 for _, rank in bin_records if rank < 1) / n if n > 0 else float("nan")
    bar         = "#" * int(recall1 * 40) if n > 0 else ""
    print(f"  {label:>8}  {n:>7}  {recall1:>9.3f}  {bar}")
    bin_results[label] = {"count": n, "recall@1": recall1}

print(f"  {sep}")
overall_recall1 = sum(1 for _, rank in records if rank < 1) / len(records) if records else float("nan")
print(f"  {'Overall':>8}  {len(records):>7}  {overall_recall1:>9.3f}")
print(f"\n{SEP}\n")

# ── Optional JSON output ───────────────────────────────────────────────────────

if args.output:
    results = {
        "prediction_path": str(args.prediction_path),
        "split": args.split,
        "label_categories": args.label_categories,
        "method": args.method,
        "eval_overlap_threshold": args.eval_overlap_threshold,
        "total_gt_relationships": len(records),
        "overall_recall@1": overall_recall1,
        "bins": bin_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {args.output}")
