"""
flat_retrieval.py
=================
Baseline flat retrieval: cosine similarity over all 203 document embeddings.
Mirrors the provided helpers/flat_retrieval.py but works from src/.

Usage:
    python src/flat_retrieval.py                           # val queries
    python src/flat_retrieval.py queries/queries_train.jsonl
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

METADATA_PATH   = "dataset/metadata.csv"
DEFAULT_QUERIES = "queries/queries_val.jsonl"
OUTPUT_PATH     = "query_results_flat.jsonl"
MODEL_NAME      = "all-MiniLM-L6-v2"
TOP_K           = 5


def main():
    query_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERIES

    print(f"Loading model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading document embeddings …")
    df = pd.read_csv(METADATA_PATH)
    doc_embeddings = np.array(
        [json.loads(e) for e in df["embedding"]], dtype=np.float32
    )
    doc_embeddings = normalize(doc_embeddings, norm="l2")
    doc_ids = df["id"].astype(str).tolist()

    print(f"Loading queries from {query_file} …")
    queries = []
    with open(query_file) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    print("Encoding + retrieving …")
    texts = [q["query"] for q in queries]
    q_embs = model.encode(texts, batch_size=32, show_progress_bar=False)
    q_embs = normalize(q_embs, norm="l2").astype(np.float32)

    results = []
    for q, q_emb in zip(queries, q_embs):
        scores = doc_embeddings @ q_emb          # (N,)
        top_idx = np.argsort(scores)[::-1][:TOP_K]
        results.append({
            "query_id": q["query_id"],
            "results": [doc_ids[i] for i in top_idx],
        })

    print(f"Writing to {OUTPUT_PATH} …")
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done! {len(results)} predictions written.")


if __name__ == "__main__":
    main()
