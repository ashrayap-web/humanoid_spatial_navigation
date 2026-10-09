#!/usr/bin/env bash
# Evaluate every pair in data/ground_truth (whose run exists) plus the synthetic C5 suite.
# Writes runs/eval_summary.md. Extra arguments are passed on (e.g. --scenes 100).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m changedet.cli eval-all "$@"
