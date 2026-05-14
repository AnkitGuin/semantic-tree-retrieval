# Semantic Tree-Based Document Retrieval
### Inter-IIT Tech Meet 14.0

A hierarchical semantic retrieval system that clusters 203 research-paper embeddings into a multi-level K-Means tree and answers natural language queries via beam-search traversal.

---

## Approach

### Tree Construction (`src/build_tree.py`)
- Loads 384-dim embeddings from `dataset/metadata.csv`
- L2-normalises all vectors so cosine similarity = dot product
- Recursively splits clusters with **K-Means** (k-means++ init, 10 restarts)
- Tree topology: **root → 6 → 5 → 4** sub-clusters, max depth 4
- Stops splitting a branch when it has ≤ 6 documents
- Each internal node stores: `name`, `centroid` (L2-normalised mean), `size`, `children`
- Each leaf stores: `id`, `file_name`, `embedding`
- Outputs `tree.json`

### Query Retrieval (`src/tree_retrieval.py`)
- Encodes query with `all-MiniLM-L6-v2`
- **Beam search** with beam width = 4 at every internal level
- At leaf containers, scores all documents by cosine similarity
- Returns the top-5 unique document IDs
- Outputs `query_results_tree.jsonl`

### Flat Baseline (`src/flat_retrieval.py`)
- Brute-force cosine similarity over all 203 documents
- Outputs `query_results_flat.jsonl`

---

## Project Structure

```
semantic_tree_retrieval/         ← repo root
│
├── src/
│   ├── build_tree.py            # Step 1 – build tree.json
│   ├── tree_retrieval.py        # Step 2 – tree-based retrieval
│   ├── flat_retrieval.py        # Step 3 – flat baseline
│   ├── evaluate.py              # Local evaluation (train queries)
│   └── inspect_tree.py          # Tree statistics & validation
│
├── dataset/                     # PROVIDED – do not modify
│   ├── metadata.csv
│   └── documents/
│
├── queries/                     # PROVIDED – do not modify
│   ├── queries_train.jsonl
│   └── queries_val.jsonl
│
├── helpers/                     # PROVIDED baseline helpers
├── schemas/                     # PROVIDED JSON schemas
│
├── run.sh                       # Reproduces full pipeline
├── requirements.txt
└── README.md
```

> **Note:** `dataset/`, `queries/`, `helpers/`, and `schemas/` are provided by the competition package. Do **not** commit `metadata.csv` or PDF files to the repository.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Full pipeline (builds tree + runs both retrievals)
bash run.sh

# 3. Evaluate on training queries
bash run.sh --eval
```

### Run scripts individually

```bash
# Build tree
python src/build_tree.py

# Tree retrieval (val queries)
python src/tree_retrieval.py

# Flat baseline (val queries)
python src/flat_retrieval.py

# Evaluate against train ground truth
python src/evaluate.py query_results_tree.jsonl

# Inspect tree stats
python src/inspect_tree.py
```

---

## Outputs

| File | Description |
|---|---|
| `tree.json` | Hierarchical semantic tree |
| `query_results_tree.jsonl` | Tree retrieval predictions (val) |
| `query_results_flat.jsonl` | Flat baseline predictions (val) |

---

## Constraints

- CPU only, ≤ 16 GB RAM, ≤ 30 min pipeline
- Must use `all-MiniLM-L6-v2` for query embeddings
- No FAISS / Pinecone / LangChain / LLM APIs
- `dataset/metadata.csv` must remain unmodified

---

## Evaluation Metrics

| Metric | Weight |
|---|---|
| Precision@3 | 30% |
| Precision@5 | 20% |
| MRR | 30% |
| Routing Accuracy | 20% |

Weighted Retrieval Score = `0.30×P@3 + 0.20×P@5 + 0.30×MRR + 0.20×RA`
