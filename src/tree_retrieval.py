"""
tree_retrieval.py
=================
Loads tree.json, encodes queries with all-MiniLM-L6-v2, traverses the
tree with beam search, and writes JSONL predictions.

Usage:
    python src/tree_retrieval.py                          # defaults to val queries
    python src/tree_retrieval.py queries/queries_train.jsonl
    python src/tree_retrieval.py queries/queries_val.jsonl
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TREE_PATH       = "tree.json"
DEFAULT_QUERIES = "queries/queries_val.jsonl"
OUTPUT_PATH     = "query_results_tree.jsonl"
MODEL_NAME      = "all-MiniLM-L6-v2"
TOP_K_BRANCHES  = 4      # beam width at each internal node
TOP_K_RESULTS   = 5      # final results per query


# ──────────────────────────────────────────────
# Tree loading & indexing
# ──────────────────────────────────────────────

def load_tree(tree_path: str) -> dict:
    with open(tree_path, "r") as f:
        return json.load(f)


def collect_leaves(node: dict) -> list[dict]:
    """Recursively gather all leaf nodes from a subtree."""
    leaves = []
    children = node.get("children", {})
    if not children:
        return leaves

    for child in children.values():
        if "id" in child:
            # This IS a leaf
            leaves.append(child)
        else:
            leaves.extend(collect_leaves(child))
    return leaves


def is_leaf_container(node: dict) -> bool:
    """True if all children are document leaves (have 'id')."""
    children = node.get("children", {})
    if not children:
        return False
    return all("id" in v for v in children.values())


# ──────────────────────────────────────────────
# Cosine similarity helpers
# ──────────────────────────────────────────────

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (assumed L2-normalised)."""
    return float(np.dot(a, b))


def rank_children_by_similarity(
    query_emb: np.ndarray,
    node: dict,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Return top-k child names ranked by cosine similarity of their
    centroid (or embedding for leaves) to the query embedding.
    """
    scores = []
    for name, child in node["children"].items():
        vec_key = "centroid" if "centroid" in child else "embedding"
        child_vec = np.array(child[vec_key], dtype=np.float32)
        norm = np.linalg.norm(child_vec)
        if norm > 0:
            child_vec = child_vec / norm
        sim = cos_sim(query_emb, child_vec)
        scores.append((name, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ──────────────────────────────────────────────
# Beam search traversal
# ──────────────────────────────────────────────

def beam_search(
    query_emb: np.ndarray,
    tree: dict,
    top_k_branches: int = TOP_K_BRANCHES,
    top_k_results: int = TOP_K_RESULTS,
) -> list[str]:
    """
    Traverse the tree using beam search.

    At each level, keep the top-k_branches most similar internal nodes,
    collect all leaves in those branches, re-rank by direct similarity,
    and return the top-k_results document IDs.
    """
    # Beam = list of nodes to explore at the current level
    beam = [tree]
    candidate_leaves: list[dict] = []

    while beam:
        next_beam = []
        for node in beam:
            children = node.get("children", {})
            if not children:
                continue

            # Check if children are leaves (documents)
            if all("id" in c for c in children.values()):
                # Score leaves and accumulate
                for child in children.values():
                    vec = np.array(child["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    child["_score"] = cos_sim(query_emb, vec)
                    candidate_leaves.append(child)
            else:
                # Internal nodes: pick top-k branches
                top_children = rank_children_by_similarity(
                    query_emb, node, top_k_branches
                )
                for name, _ in top_children:
                    next_beam.append(node["children"][name])

        # Also grab leaves that were direct children of internal nodes
        beam = next_beam

    # If we didn't gather enough, fall back to all leaves
    if len(candidate_leaves) < top_k_results:
        candidate_leaves = collect_leaves(tree)
        for leaf in candidate_leaves:
            vec = np.array(leaf["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            leaf["_score"] = cos_sim(query_emb, vec)

    # Sort by score and return top-k IDs
    candidate_leaves.sort(key=lambda x: x.get("_score", 0), reverse=True)

    # De-duplicate by doc id
    seen = set()
    top_docs = []
    for leaf in candidate_leaves:
        doc_id = leaf["id"]
        if doc_id not in seen:
            seen.add(doc_id)
            top_docs.append(doc_id)
        if len(top_docs) == top_k_results:
            break

    # Pad if needed (shouldn't happen with 203 docs, but just in case)
    if len(top_docs) < top_k_results:
        all_leaves = collect_leaves(tree)
        for leaf in all_leaves:
            if leaf["id"] not in seen:
                top_docs.append(leaf["id"])
                seen.add(leaf["id"])
            if len(top_docs) == top_k_results:
                break

    return top_docs


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    query_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERIES

    # ── Load model ──
    print(f"Loading sentence-transformer model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    # ── Load tree ──
    print(f"Loading tree from {TREE_PATH} …")
    tree = load_tree(TREE_PATH)
    print(f"  Tree root size: {tree.get('size', '?')}")

    # ── Load queries ──
    print(f"Loading queries from {query_file} …")
    queries = []
    with open(query_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"  {len(queries)} queries loaded")

    # ── Encode queries ──
    print("Encoding queries …")
    texts = [q["query"] for q in queries]
    query_embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    query_embeddings = normalize(query_embeddings, norm="l2").astype(np.float32)

    # ── Retrieve ──
    print("Running tree-based retrieval …")
    results = []
    total_latency = 0.0
    for i, (q, q_emb) in enumerate(zip(queries, query_embeddings)):
        t0 = time.time()
        top_docs = beam_search(q_emb, tree, TOP_K_BRANCHES, TOP_K_RESULTS)
        latency = time.time() - t0
        total_latency += latency
        results.append({"query_id": q["query_id"], "results": top_docs})
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(queries)}  avg latency={total_latency/(i+1)*1000:.1f}ms")

    # ── Write output ──
    print(f"Writing results to {OUTPUT_PATH} …")
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    avg_latency = total_latency / len(queries) * 1000 if queries else 0
    print(f"Done!  {len(results)} results written  |  avg latency: {avg_latency:.1f} ms/query")


if __name__ == "__main__":
    main()
