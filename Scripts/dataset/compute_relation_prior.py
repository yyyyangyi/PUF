"""Compute class-conditional relation prior P(r | c_i, c_j) and existence prior
P(exists | c_i, c_j) from training annotations.

Usage:
    python compute_relation_prior.py --path /path/to/3RScan --output relation_prior.npz

The output is a .npz file with two arrays:
    P_class : (num_obj_classes, num_obj_classes, num_rel_classes)
        P(relation r | subject class i, object class j, some relation exists)
    P_exist : (num_obj_classes, num_obj_classes)
        Empirical fraction of (c_i, c_j) directed pairs that have >= 1 annotation.
        Used to suppress over-completion of the scene graph.
"""
import json
import pathlib
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Compute class-conditional relation prior from training annotations.')
parser.add_argument('--path', type=pathlib.Path, required=True, help='3RScan directory')
parser.add_argument('--label_categories', type=str, choices=['scannet', 'replica'], default='scannet')
parser.add_argument('--output', type=pathlib.Path, default=None, help='Output .npz path (default: <path>/relation_prior.npz)')
parser.add_argument('--alpha_smooth', type=float, default=0.01, help='Laplace smoothing parameter')
args = parser.parse_args()

if args.label_categories == 'scannet':
    object_json_path = args.path / "objects.json"
    rel_json_path = args.path / "3DSSG_subset/relationships20.json"
    mapping_path = args.path / "3DSSG_subset/3dssg_to_scannet.json"
    train_scans_path = args.path / "3DSSG_subset/train_scans.txt"
    OBJ_CLASS_NAME = "ScanNet_list"
    REL_CLASS_NAME = "ScanNet_rel"
    MAPPING_NAME = "3DSSG2NYUv2"
elif args.label_categories == 'replica':
    object_json_path = args.path / "ReplicaSSG/objects.json"
    rel_json_path = args.path / "ReplicaSSG/relationships.json"
    mapping_path = args.path / "ReplicaSSG/replica_to_visual_genome.json"
    train_scans_path = args.path / "ReplicaSSG/test_scans.txt"
    OBJ_CLASS_NAME = "VisualGenome_list"
    REL_CLASS_NAME = "VisualGenome_rel"
    MAPPING_NAME = "Replica2VisualGenome"

# Load data
with open(mapping_path, 'r') as f:
    class_mapping = json.load(f)
with open(object_json_path, 'r') as f:
    object_data = json.load(f)
with open(rel_json_path, 'r') as f:
    rel_data = json.load(f)
with open(train_scans_path, 'r') as f:
    train_scans = set(line.strip() for line in f.readlines())

obj_classes = class_mapping[OBJ_CLASS_NAME]
rel_classes = class_mapping[REL_CLASS_NAME]
num_obj_classes = len(obj_classes)
num_rel_classes = len(rel_classes)

print(f"Object classes ({num_obj_classes}): {obj_classes}")
print(f"Relation classes ({num_rel_classes}): {rel_classes}")
print(f"Training scans: {len(train_scans)}")

# Build object ID to class mapping for training scans
OBJ2CLASS = {}
for scan in object_data["scans"]:
    scan_id = scan["scan"]
    if scan_id not in train_scans:
        continue
    id_to_label = {}
    for obj in scan["objects"]:
        mapped_label = class_mapping[MAPPING_NAME].get(obj["label"])
        if mapped_label not in obj_classes:
            id_to_label[int(obj["id"])] = -1
        else:
            id_to_label[int(obj["id"])] = obj_classes.index(mapped_label)
    OBJ2CLASS[scan_id] = id_to_label

# Count total directed object pairs per class combination
pair_counts = np.zeros((num_obj_classes, num_obj_classes), dtype=float)
for scan_id, id_to_label in OBJ2CLASS.items():
    obj_class_list = [cls for cls in id_to_label.values() if cls != -1]
    for ci in obj_class_list:
        for cj in obj_class_list:
            if ci != cj:
                pair_counts[ci, cj] += 1

# Count relation occurrences per class pair
counts = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=float)
annotated_pairs = np.zeros((num_obj_classes, num_obj_classes), dtype=float)
total_rels = 0
skipped_rels = 0

for scan in rel_data["scans"]:
    scan_id = scan["scan"]
    if scan_id not in train_scans:
        continue
    if scan_id not in OBJ2CLASS:
        continue
    for rel in scan["relationships"]:
        subj_id, obj_id, rel_idx = rel[0], rel[1], rel[2]
        if subj_id not in OBJ2CLASS[scan_id] or obj_id not in OBJ2CLASS[scan_id]:
            skipped_rels += 1
            continue
        subj_class = OBJ2CLASS[scan_id][subj_id]
        obj_class = OBJ2CLASS[scan_id][obj_id]
        if subj_class == -1 or obj_class == -1:
            skipped_rels += 1
            continue
        counts[subj_class, obj_class, rel_idx] += 1
        annotated_pairs[subj_class, obj_class] += 1
        total_rels += 1

print(f"Total training relations: {total_rels}, skipped: {skipped_rels}")

P_exist = np.where(pair_counts > 0, (annotated_pairs / pair_counts).clip(0.0, 1.0), 0.0)
totals = counts.sum(axis=-1, keepdims=True)
P_class = (counts + args.alpha_smooth) / (totals + num_rel_classes * args.alpha_smooth)

# Save both arrays in a single .npz file
output_path = args.output if args.output else args.path / "relation_prior.npz"
if str(output_path).endswith('.npy'):
    output_path = pathlib.Path(str(output_path)[:-4] + '.npz')
np.savez(output_path, P_class=P_class, P_exist=P_exist)
print(f"\nSaved prior to {output_path}")
print(f"P_class shape: {P_class.shape}, P_exist shape: {P_exist.shape}")
