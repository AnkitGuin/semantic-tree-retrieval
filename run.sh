#!/usr/bin/env bash
# ============================================================
# run.sh  –  Full pipeline for Semantic Tree Retrieval
# Inter-IIT Tech Meet 14.0
# ============================================================
# Usage:
#   bash run.sh             (full pipeline)
#   bash run.sh --eval      (full pipeline + evaluate on train set)
# ============================================================

set -euo pipefail

EVAL=false
for arg in "$@"; do
  [[ "$arg" == "--eval" ]] && EVAL=true
done

echo "============================================================"
echo "  Semantic Tree-Based Document Retrieval Pipeline"
echo "============================================================"

# ── 0. Dependencies ──────────────────────────────────────────
echo ""
echo ">>> [0/4] Installing dependencies …"
pip install -r requirements.txt --quiet

# ── 1. Build semantic tree ───────────────────────────────────
echo ""
echo ">>> [1/4] Building semantic tree …"
python src/build_tree.py

# ── 2. Tree-based retrieval on validation queries ────────────
echo ""
echo ">>> [2/4] Running tree-based retrieval (val queries) …"
python src/tree_retrieval.py queries/queries_val.jsonl

# ── 3. Flat baseline on validation queries ───────────────────
echo ""
echo ">>> [3/4] Running flat baseline (val queries) …"
python src/flat_retrieval.py queries/queries_val.jsonl

# ── 4. Optional: evaluate on training set ───────────────────
if [ "$EVAL" = true ]; then
  echo ""
  echo ">>> [4/4] Evaluating tree retrieval on train queries …"
  python src/tree_retrieval.py queries/queries_train.jsonl
  mv query_results_tree.jsonl query_results_tree_train.jsonl
  python src/evaluate.py query_results_tree_train.jsonl

  echo ""
  echo ">>> Evaluating flat baseline on train queries …"
  python src/flat_retrieval.py queries/queries_train.jsonl
  mv query_results_flat.jsonl query_results_flat_train.jsonl
  python src/evaluate.py query_results_flat_train.jsonl

  # Restore val outputs
  python src/tree_retrieval.py queries/queries_val.jsonl
  python src/flat_retrieval.py queries/queries_val.jsonl
fi

echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "  Outputs:"
echo "    tree.json                  (semantic tree)"
echo "    query_results_tree.jsonl   (tree retrieval – val)"
echo "    query_results_flat.jsonl   (flat baseline  – val)"
echo "============================================================"
