"""
inspect_tree.py
===============
Utility to inspect and validate tree.json.
Prints statistics about depth, branching factor, leaf distribution,
and semantic coherence scores.

Usage:
    python src/inspect_tree.py
    python src/inspect_tree.py tree.json
"""

import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


TREE_PATH = "tree.json"


def load_tree(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Tree statistics ──────────────────────────────────────────────

def tree_stats(node: dict, depth: int = 0, stats: dict = None) -> dict:
    if stats is None:
        stats = {"depth": 0, "internal_nodes": 0, "leaf_nodes": 0,
                 "branching": [], "sizes": []}

    stats["depth"] = max(stats["depth"], depth)
    children = node.get("children", {})

    if not children:
        return stats

    all_leaves = all("id" in c for c in children.values())
    if all_leaves:
        stats["leaf_nodes"] += len(children)
        stats["branching"].append(len(children))
        return stats

    stats["internal_nodes"] += 1
    stats["branching"].append(len(children))
    stats["sizes"].append(node.get("size", 0))

    for child in children.values():
        if "id" not in child:
            tree_stats(child, depth + 1, stats)

    return stats


def collect_all_leaves(node: dict) -> list[dict]:
    leaves = []
    for child in node.get("children", {}).values():
        if "id" in child:
            leaves.append(child)
        else:
            leaves.extend(collect_all_leaves(child))
    return leaves


def semantic_coherence(node: dict, sample_size: int = 20) -> float:
    """
    Intra-cluster coherence: mean cosine similarity of leaves to centroid.
    """
    leaves = collect_all_leaves(node)
    if not leaves or "centroid" not in node:
        return 0.0

    centroid = np.array(node["centroid"], dtype=np.float32).reshape(1, -1)
    leaf_embs = np.array([l["embedding"] for l in leaves[:sample_size]], dtype=np.float32)
    sims = cosine_similarity(leaf_embs, centroid).flatten()
    return float(sims.mean())


def validate_tree(tree: dict) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []

    def check_node(node, path):
        if "name" not in node:
            errors.append(f"{path}: missing 'name'")
        if "centroid" not in node:
            errors.append(f"{path}: missing 'centroid'")
        elif len(node["centroid"]) != 384:
            errors.append(f"{path}: centroid has {len(node['centroid'])} dims (expected 384)")
        if "size" not in node:
            errors.append(f"{path}: missing 'size'")
        if "children" not in node:
            errors.append(f"{path}: missing 'children'")
            return

        for cname, child in node["children"].items():
            cpath = f"{path}/{cname}"
            if "id" in child:
                # Leaf validation
                if "file_name" not in child:
                    errors.append(f"{cpath}: leaf missing 'file_name'")
                if "embedding" not in child:
                    errors.append(f"{cpath}: leaf missing 'embedding'")
                elif len(child["embedding"]) != 384:
                    errors.append(f"{cpath}: leaf embedding has {len(child['embedding'])} dims")
            else:
                check_node(child, cpath)

    check_node(tree, "root")
    return errors


# ── Main ─────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TREE_PATH
    print(f"Loading tree from {path} …")
    tree = load_tree(path)

    # Validate
    errors = validate_tree(tree)
    if errors:
        print(f"\n❌  {len(errors)} VALIDATION ERRORS:")
        for e in errors[:20]:
            print(f"   {e}")
    else:
        print("✅  Tree schema valid")

    # Statistics
    stats = tree_stats(tree)
    leaves = collect_all_leaves(tree)

    print("\n── Tree Statistics ──────────────────────────────────")
    print(f"  Total documents (leaves): {len(leaves)}")
    print(f"  Max depth:                {stats['depth']}")
    print(f"  Internal nodes:           {stats['internal_nodes']}")
    print(f"  Branching factor avg:     {np.mean(stats['branching']):.1f}")
    print(f"  Branching factor min/max: {min(stats['branching'])} / {max(stats['branching'])}")
    if stats["sizes"]:
        print(f"  Node size avg:            {np.mean(stats['sizes']):.1f}")
        print(f"  Node size std:            {np.std(stats['sizes']):.1f}")

    # Coherence of root-level children
    print("\n── Top-Level Cluster Coherence ─────────────────────")
    for cname, child in tree.get("children", {}).items():
        if "id" not in child:
            coh = semantic_coherence(child)
            size = child.get("size", "?")
            print(f"  {cname:30s}  size={size:4}  coherence={coh:.4f}")

    print()


if __name__ == "__main__":
    main()
