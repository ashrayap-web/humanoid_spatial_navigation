#!/usr/bin/env bash
# Download every model checkpoint named in configs/default.yaml into checkpoints/hf.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m changedet.models
