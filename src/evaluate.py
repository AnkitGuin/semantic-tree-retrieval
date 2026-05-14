"""
evaluate.py
===========
Local evaluation tool for tree or flat retrieval results.
Computes Precision@3, Precision@5, MRR, and Routing Accuracy.

Usage:
    python src/evaluate.py query_results_tree.jsonl
    python src/evaluate.py query_results_flat.jsonl
    python src/evaluate.py query_results_tree.jsonl queries/queries_train.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


TRAIN_QUERIES = "queries/queries_train.jsonl"


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def reciprocal_rank(retrieved: list, relevant: set) -> float:
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def load_tree_leaves(tree_path: str = "tree.json") -> dict[str, set[str]]:
    """
    Build a mapping: subtree_name -> set of leaf doc IDs.
    Used for routing accuracy calculation.
    """
    try:
        with open(tree_path) as f:
            tree = json.load(f)
    except FileNotFoundError:
        return {}

    subtree_leaves: dict[str, set[str]] = {}

    def collect(node, path):
        node_id = path
        leaves = set()
        children = node.get("children", {})
        for name, child in children.items():
            if "id" in child:
                leaves.add(child["id"])
            else:
                child_leaves = collect(child, f"{path}/{name}")
                leaves.update(child_leaves)
        subtree_leaves[node_id] = leaves
        return leaves

    collect(tree, "root")
    return subtree_leaves


def routing_accuracy(
    retrieved: list,
    relevant: set,
    subtree_leaves: dict[str, set[str]],
) -> float:
    """
    1.0 if any visited subtree contains at least one ground-truth document.
    """
    if not relevant or not subtree_leaves:
        return 0.0
    retrieved_set = set(retrieved)
    for subtree_id, leaf_set in subtree_leaves.items():
        # If retrieved docs and relevant docs share a subtree
        if retrieved_set & leaf_set and relevant & leaf_set:
            return 1.0
    return 0.0


def evaluate(results_path: str, train_path: str = TRAIN_QUERIES) -> None:
    results = {r["query_id"]: r["results"] for r in load_jsonl(results_path)}
    train   = {q["query_id"]: set(q.get("relevant_doc_ids", [])) for q in load_jsonl(train_path)}
    subtree_leaves = load_tree_leaves()

    metrics = defaultdict(float)
    n = 0

    for qid, relevant in train.items():
        if not relevant:
            continue
        retrieved = results.get(qid, [])
        metrics["p3"]  += precision_at_k(retrieved, relevant, 3)
        metrics["p5"]  += precision_at_k(retrieved, relevant, 5)
        metrics["mrr"] += reciprocal_rank(retrieved, relevant)
        metrics["ra"]  += routing_accuracy(retrieved, relevant, subtree_leaves)
        n += 1

    if n == 0:
        print("No overlapping query IDs found. Make sure you're using train queries.")
        return

    p3  = metrics["p3"]  / n
    p5  = metrics["p5"]  / n
    mrr = metrics["mrr"] / n
    ra  = metrics["ra"]  / n

    # Weighted retrieval score (80% of total grade)
    weighted = 0.30 * p3 + 0.20 * p5 + 0.30 * mrr + 0.20 * ra

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Precision@3:       {p3:.4f}")
    print(f"Precision@5:       {p5:.4f}")
    print(f"MRR:               {mrr:.4f}")
    print(f"Routing Accuracy:  {ra:.4f}")
    print(f"Queries evaluated: {n}")
    print("=" * 60)
    print(f"Weighted Retrieval Score: {weighted:.4f}")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/evaluate.py <results.jsonl> [train_queries.jsonl]")
        sys.exit(1)

    results_path = sys.argv[1]
    train_path   = sys.argv[2] if len(sys.argv) > 2 else TRAIN_QUERIES
    evaluate(results_path, train_path)


if __name__ == "__main__":
    main()
