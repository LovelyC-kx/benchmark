#!/usr/bin/env bash
# Runs with standard-library Python. Does not install a GPU framework.
set -euo pipefail
SNN_DOWNLOAD_ROOT="${1:-downloads}"
python -m snnbench.download --out "$SNN_DOWNLOAD_ROOT" --ids D31 D32 D33 D35 D46 --extract
