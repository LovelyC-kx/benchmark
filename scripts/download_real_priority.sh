#!/usr/bin/env bash
# Real sources and MNIST-derived sequence controls. No author split substitution.
set -euo pipefail
SNN_REAL_DOWNLOAD_ROOT="${1:-downloads}"
python -m snnbench.download --out "$SNN_REAL_DOWNLOAD_ROOT" --ids D31 D32 D33 D35 D41 D46 D47 D89 D96 D97 D98 D99 --extract
# UCI WISDM outer archive can contain wisdm-dataset.zip: extract that inner archive
# as well. OpenBMI per-subject MAT and EarthNet HDF5 follow their manual sources.
