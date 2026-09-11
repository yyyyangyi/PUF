"""Output node/edge labels alongside normalized entropy as predictive uncertainty.

Usage
-----
python infer_uncertainty.py \\
    --scan_id <ID> \\
    --prediction_path <path/to/predictions.pkl> \\
    --dataset_path <path/to/dataset_root>
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Uncertainty computation
# ---------------------------------------------------------------------------

def predictive_entropy(alpha: np.ndarray) -> float:
    """Categorical entropy of Dirichlet posterior mean, normalised to [0, 1].

    Args:
        alpha: Dirichlet concentration parameters, shape (K,). Must be > 0.
    """
    alpha = np.clip(alpha, 1e-300, None)   # guard against exact zeros
    K = len(alpha)
    if K <= 1:
        return 0.0
    p = alpha / alpha.sum()
    # Mask out zero entries to avoid 0·log(0) = nan  (limit is 0)
    mask = p > 0
    H = -np.sum(p[mask] * np.log(p[mask]))
    H_max = np.log(K)
    return float(np.clip(H / H_max, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Load class mapping
    if args.label_categories == "scannet":
        ssg_path = "3DSSG_subset"
        class_mapping_path = f"{ssg_path}/3dssg_to_scannet.json"
        obj_class_key = "ScanNet_list"
        rel_class_key = "ScanNet_rel"
    else:
        ssg_path = "ReplicaSSG"
        class_mapping_path = f"{ssg_path}/replica_to_visual_genome.json"
        obj_class_key = "VisualGenome_list"
        rel_class_key = "VisualGenome_rel"

    with open(Path(args.dataset_path) / class_mapping_path) as f:
        class_mapping = json.load(f)
    obj_classes = class_mapping[obj_class_key]
    rel_classes = class_mapping[rel_class_key]

    # Load predictions
    with open(args.prediction_path, "rb") as f:
        predictions = pickle.load(f)

    if args.scan_id not in predictions:
        available = list(predictions.keys())[:5]
        raise KeyError(
            f"scan_id '{args.scan_id}' not found in predictions.\n"
            f"Available (first 5): {available}"
        )

    pred = predictions[args.scan_id]

    if "node_alphas" not in pred:
        raise KeyError(
            "'node_alphas' not found in predictions. "
            "Re-run main.py with --use_puf to generate uncertainty-aware predictions."
        )

    node_alphas = pred["node_alphas"]   # (N, C)
    N = node_alphas.shape[0]

    # --- Nodes ---
    node_rows = []
    for i in range(N):
        alpha = node_alphas[i]
        label_idx = int(np.argmax(alpha))
        label = obj_classes[label_idx]
        unc = predictive_entropy(alpha)
        mean = pred["mean"][i].tolist() if "mean" in pred else [float("nan")] * 3
        node_rows.append({
            "node_idx": i,
            "label_idx": label_idx,
            "label": label,
            "uncertainty": round(unc, 4),
            "mean_pos": [round(float(v), 3) for v in mean],
        })

    # --- Edges ---
    edge_rows = []
    if "edge_alphas" not in pred:
        print(
            "[WARNING] 'edge_alphas' not in predictions — edge uncertainty unavailable.\n"
            "         Re-run main.py with --use_puf to capture edge alphas."
        )
    elif len(pred["edge_index"][0]) > 0:
        edge_alphas = pred["edge_alphas"]   # (E, K)
        edge_index = pred["edge_index"]     # (2, E)
        E = edge_alphas.shape[0]
        for e in range(E):
            s_idx = int(edge_index[0, e])
            o_idx = int(edge_index[1, e])
            alpha = edge_alphas[e]
            rel_idx = int(np.argmax(alpha))
            rel_label = rel_classes[rel_idx]
            unc = predictive_entropy(alpha)
            sub_label = obj_classes[int(np.argmax(node_alphas[s_idx]))]
            obj_label = obj_classes[int(np.argmax(node_alphas[o_idx]))]
            edge_rows.append({
                "sub_idx": s_idx,
                "obj_idx": o_idx,
                "sub_label": sub_label,
                "obj_label": obj_label,
                "rel_idx": rel_idx,
                "rel_label": rel_label,
                "uncertainty": round(unc, 4),
            })

    # --- Output ---
    if args.output_json:
        result = {
            "scan_id": args.scan_id,
            "uncertainty_metric": "H(alpha/alpha_0) / log(K)",
            "nodes": node_rows,
            "edges": edge_rows,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path}")
    else:
        _print_table(args.scan_id, node_rows, edge_rows)


def _print_table(scan_id, node_rows, edge_rows):
    print(f"\n=== Scan: {scan_id} ===\n")

    # Nodes
    print(f"Nodes ({len(node_rows)}):")
    NW = [5, 20, 12, 30]   # col widths: idx, label, uncertainty, mean_pos
    hdr = (f"  {'idx':>{NW[0]}}  {'label':<{NW[1]}}  {'uncertainty':<{NW[2]}}"
           f"  {'mean_pos (x, y, z)':<{NW[3]}}")
    print(hdr)
    print("  " + "-" * (sum(NW) + 3 * 2))
    for r in node_rows:
        pos = "({:.3f}, {:.3f}, {:.3f})".format(*r["mean_pos"])
        print(f"  {r['node_idx']:>{NW[0]}}  {r['label']:<{NW[1]}}"
              f"  {r['uncertainty']:<{NW[2]}.4f}  {pos:<{NW[3]}}")

    print()

    # Edges
    if not edge_rows:
        print("Edges: none.")
        return

    print(f"Edges ({len(edge_rows)}):")
    EW = [4, 4, 18, 18, 16, 12]   # s, o, sub_label, obj_label, predicate, uncertainty
    ehdr = (f"  {'s':>{EW[0]}}  {'o':>{EW[1]}}  {'subject':<{EW[2]}}  "
            f"{'object':<{EW[3]}}  {'predicate':<{EW[4]}}  {'uncertainty':<{EW[5]}}")
    print(ehdr)
    print("  " + "-" * (sum(EW) + 5 * 2))
    for r in edge_rows:
        print(f"  {r['sub_idx']:>{EW[0]}}  {r['obj_idx']:>{EW[1]}}  "
              f"{r['sub_label']:<{EW[2]}}  {r['obj_label']:<{EW[3]}}  "
              f"{r['rel_label']:<{EW[4]}}  {r['uncertainty']:<{EW[5]}.4f}")

    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Output scene graph labels with predictive uncertainty for a single scan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--scan_id", type=str, required=True,
                   help="Scan ID to inspect.")
    p.add_argument("--prediction_path", type=Path, required=True,
                   help="Path to predictions .pkl produced by main.py --use_puf.")
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to dataset root (used only to load class name lists).")
    p.add_argument("--label_categories", type=str, choices=["scannet", "replica"],
                   default="scannet")
    p.add_argument("--output_json", type=str, default=None,
                   help="If set, write output to this JSON file instead of printing.")
    args = p.parse_args()
    main(args)
