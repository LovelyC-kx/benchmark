#!/usr/bin/env bash
#SBATCH --job-name=snnbench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Submit with: sbatch --array=0-63 scripts/slurm_array.sbatch manifest.jsonl 64
set -euo pipefail
SNN_MANIFEST="${1:?manifest path required}"
SNN_SHARDS="${2:?shard count required}"
SNN_QUEUE="${SLURM_TMPDIR:-/tmp}/snnbench-${SLURM_JOB_ID}.sqlite"
python -m snnbench queue-init --db "$SNN_QUEUE" --manifest "$SNN_MANIFEST" --shard "$SLURM_ARRAY_TASK_ID" --shards "$SNN_SHARDS"
python -m snnbench worker --db "$SNN_QUEUE" --out runs --data prepared --device cuda:0 --analyze
