#!/usr/bin/env bash
# Install three small, verified files; no environment, data or training-source changes.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Usage: bash install_stage1_compiler.sh <fixed 40-character commit>" >&2
  exit 2
fi
: "${SNN_PROJECT_ROOT:?source configs/server_paths.sh first}"
: "${SNN_CODE:?source configs/server_paths.sh first}"
[[ -d "$SNN_PROJECT_ROOT/scripts" && -d "$SNN_CODE/snnbench" ]] || {
  echo "Configured project directories do not exist" >&2; exit 2;
}
stage1_revision="$1"
stage1_python="${SNN_PYTHON:-python}"
stage1_tmp="$(mktemp -d "$SNN_PROJECT_ROOT/scripts/.stage1_install.XXXXXX")"
trap 'rm -rf -- "$stage1_tmp"' EXIT

stage1_sources=(scripts/compile_stage1.py scripts/efficient_campaign.py snnbench/coverage.py)
stage1_hashes=(
  b53a96d54acb313078b9b3b5bc29e09d28365dcb0a33d723a6631ec38badf0f6
  3063ec23f0243c2718576db3792a8ebbe9610c5f93bc0fe05b5fe72b99282a4a
  98a530e905b12f3ccb070374296305adf3b9c8fdc12bb905c1635f3bea7d7de8
)
stage1_targets=(
  "$SNN_PROJECT_ROOT/scripts/compile_stage1.py"
  "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py"
  "$SNN_CODE/snnbench/coverage.py"
)

# Verify every download before replacing any installed file.
for stage1_i in 0 1 2; do
  curl -fL --retry 3 --connect-timeout 20 --max-time 180 \
    "https://raw.githubusercontent.com/LovelyC-kx/benchmark/$stage1_revision/${stage1_sources[$stage1_i]}" \
    -o "$stage1_tmp/$stage1_i.py"
  printf '%s  %s\n' "${stage1_hashes[$stage1_i]}" "$stage1_tmp/$stage1_i.py" | sha256sum -c -
  "$stage1_python" -m py_compile "$stage1_tmp/$stage1_i.py"
done

stage1_suffix="before_stage1_$(date -u +%Y%m%dT%H%M%SZ)_$$"
for stage1_i in 0 1 2; do
  stage1_target="${stage1_targets[$stage1_i]}"
  if [[ -f "$stage1_target" ]]; then
    cp -p -- "$stage1_target" "$stage1_target.$stage1_suffix"
  fi
done
for stage1_i in 0 1 2; do
  mv -- "$stage1_tmp/$stage1_i.py" "${stage1_targets[$stage1_i]}"
  printf 'Installed: %s\n' "${stage1_targets[$stage1_i]}"
done
echo "INSTALL_OK: compiler installed; existing training, data and weights retained."
