"""
build_tree.py
=============
Builds a hierarchical semantic tree from document embeddings using
recursive K-Means clustering and saves the result as tree.json.

Usage:
    python src/build_tree.py
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
METADATA_PATH   = "dataset/metadata.csv"
OUTPUT_TREE     = "tree.json"

# Tree topology
ROOT_CLUSTERS   = 6      # first-level clusters  (~domain level)
MID_CLUSTERS    = 5      # second-level clusters
LEAF_CLUSTERS   = 4      # third-level clusters  (leaves when size <= MIN_LEAF_SIZE)
MIN_LEAF_SIZE   = 6      # stop splitting if a cluster has <= this many docs
MAX_DEPTH       = 4      # hard cap on recursion depth
KMEANS_INIT     = "k-means++"
KMEANS_ITERS    = 300
KMEANS_RESTARTS = 10
RANDOM_STATE    = 42


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_embeddings(metadata_path: str):
    """Return DataFrame and (N, 384) numpy embedding matrix."""
    df = pd.read_csv(metadata_path)
    embeddings = np.array([json.loads(e) for e in df["embedding"]], dtype=np.float32)
    # L2-normalise so cosine similarity == dot product
    embeddings = normalize(embeddings, norm="l2")
    return df, embeddings


def compute_centroid(embeddings: np.ndarray) -> list:
    """Return L2-normalised centroid of a set of embeddings."""
    c = embeddings.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm > 0:
        c = c / norm
    return c.tolist()


def kmeans_cluster(embeddings: np.ndarray, n_clusters: int, random_state: int = RANDOM_STATE):
    """Run K-Means and return label array."""
    n_clusters = min(n_clusters, len(embeddings))
    if n_clusters <= 1:
        return np.zeros(len(embeddings), dtype=int)
    km = KMeans(
        n_clusters=n_clusters,
        init=KMEANS_INIT,
        max_iter=KMEANS_ITERS,
        n_init=KMEANS_RESTARTS,
        random_state=random_state,
    )
    return km.fit_predict(embeddings)


# ──────────────────────────────────────────────
# Recursive tree builder
# ──────────────────────────────────────────────

def build_subtree(
    df_subset: pd.DataFrame,
    emb_subset: np.ndarray,
    depth: int,
    n_clusters: int,
    node_prefix: str,
) -> dict:
    """
    Recursively partition documents into a subtree node.

    Returns either an internal node dict or a dict of leaf node dicts.
    """
    size = len(df_subset)

    # ── Base case: leaf ──
    if depth >= MAX_DEPTH or size <= MIN_LEAF_SIZE:
        children = {}
        for _, row in df_subset.iterrows():
            doc_id = str(row["id"])
            children[doc_id] = {
                "id": doc_id,
                "file_name": row["file_name"],
                "embedding": json.loads(row["embedding"]),
            }
        return {
            "name": node_prefix,
            "centroid": compute_centroid(emb_subset),
            "size": size,
            "children": children,
        }

    # ── Internal node: cluster and recurse ──
    labels = kmeans_cluster(emb_subset, n_clusters)
    unique_labels = np.unique(labels)

    children = {}
    for lbl in unique_labels:
        mask = labels == lbl
        child_df  = df_subset.iloc[mask]
        child_emb = emb_subset[mask]
        child_name = f"{node_prefix}_C{lbl}"

        # Choose branching factor for next level
        next_k = MID_CLUSTERS if depth == 0 else LEAF_CLUSTERS

        children[child_name] = build_subtree(
            df_subset=child_df.reset_index(drop=True),
            emb_subset=child_emb,
            depth=depth + 1,
            n_clusters=next_k,
            node_prefix=child_name,
        )

    return {
        "name": node_prefix,
        "centroid": compute_centroid(emb_subset),
        "size": size,
        "children": children,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading embeddings …")
    df, embeddings = load_embeddings(METADATA_PATH)
    print(f"  Loaded {len(df)} documents  (embedding dim={embeddings.shape[1]})")

    print("Building semantic tree …")
    tree = build_subtree(
        df_subset=df,
        emb_subset=embeddings,
        depth=0,
        n_clusters=ROOT_CLUSTERS,
        node_prefix="root",
    )
    tree["name"] = "root"

    print(f"Saving tree to {OUTPUT_TREE} …")
    with open(OUTPUT_TREE, "w") as f:
        json.dump(tree, f)

    size_mb = os.path.getsize(OUTPUT_TREE) / 1e6
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  |  tree.json size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
