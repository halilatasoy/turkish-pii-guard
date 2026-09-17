#!/usr/bin/env bash
# End-to-end pipeline on a CUDA machine. Usage: bash run_gpu.sh [RUN_NAME]
set -euo pipefail
RUN=${1:-v1}
cd "$(dirname "$0")"
python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
mkdir -p models data results outputs
[ -d models/pii-guard-turkish-270m ] || python -c "from huggingface_hub import snapshot_download as s; s('cagrigungor/pii-guard-turkish-270m', local_dir='models/pii-guard-turkish-270m')"
[ -f data/benchmark_1000.csv ] || python -c "from huggingface_hub import hf_hub_download as h; h('cagrigungor/turkish-pii-masking-benchmark','benchmark_1000.csv',repo_type='dataset',local_dir='data')"
[ -f data/train.jsonl ] || python scripts/gen_data.py --n 300000 --val 4000 --out data/train.jsonl --val_out data/val.jsonl
[ -f results/baseline_270m.json ] || python scripts/eval.py --model models/pii-guard-turkish-270m --out results/baseline_270m.csv --batch 64
python scripts/train.py --model models/pii-guard-turkish-270m --out outputs/$RUN --epochs 2 --lr 5e-5 --bsz 32 --eval_steps 1000 --save_steps 2000
python scripts/eval.py --model outputs/$RUN/final --out results/$RUN.csv --batch 64
python scripts/error_report.py results/baseline_270m.csv results/$RUN.csv
