"""Prepare the user's outer project layout; usable before installing PyTorch.

Keep the complete checkout/archive under src/benchmark. Generated outer scripts
source one path contract and use the currently activated Python interpreter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time


DEFAULT_ROOT = '/home/bohan_zhang/research/benchmark_temporal'

COMMON = '''#!/usr/bin/env bash
set -euo pipefail
SNN_WRAPPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SNN_WRAPPER_ROOT/configs/server_paths.sh"
cd "$SNN_CODE"
'''

GPU_ARGS = '''if (( $# )); then
  SNN_GPU_IDS=("$@")
else
  SNN_GPU_LIST="$("$SNN_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch required"; print(*range(torch.cuda.device_count()), sep="\\n")')"
  mapfile -t SNN_GPU_IDS <<< "$SNN_GPU_LIST"
fi
'''


def _revision(code, project):
    if (code / '.git').exists():
        return subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    marker = project / 'src' / 'benchmark_source.commit'
    if marker.is_file():
        value = marker.read_text(encoding='utf-8-sig').strip()
        if not re.fullmatch(r'[0-9a-fA-F]{40}', value):
            raise ValueError('benchmark_source.commit must contain the full Git commit SHA')
        return value.lower()
    return None


def _atomic_text(path, content, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def setup(project_root, code_root=None, campaign_name='campaign_v04', replace_config=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', campaign_name):
        raise ValueError('A campaign name must be one plain directory name')
    project = Path(project_root).expanduser().resolve()
    code = Path(code_root).expanduser().resolve() if code_root else Path(__file__).resolve().parents[1]
    required = ('pyproject.toml', 'snnbench/cli.py', 'SNN_Research_Master_Index_2026-10-24.md')
    for relative in required:
        if not (code / relative).is_file():
            raise FileNotFoundError('Upload the complete repository; missing ' + str(code / relative))
    paths = {
        'SNN_PROJECT_ROOT': project,
        'SNN_ROOT': project,
        'SNN_CODE': code,
        'SNN_CONFIGS': project / 'configs',
        'SNN_LOGS': project / 'logs',
        'SNN_RAW': project / 'data' / 'raw',
        'SNN_CANONICAL': project / 'data' / 'canonical',
        'SNN_DATA': project / 'data' / 'cache' / campaign_name,
        'SNN_UPSTREAM': project / 'src' / 'upstream',
        'SNN_CAMPAIGN': project / 'results' / campaign_name,
        'SNN_SMOKE': project / 'results' / 'smoke',
        'SNN_SMOKE_DATA': project / 'data' / 'cache' / 'smoke',
    }
    receipt = {'schema': 'benchmark_temporal_server_layout_v1', 'campaign_name': campaign_name,
               'paths': {key: str(value) for key, value in paths.items()},
               'source_revision': _revision(code, project),
               'python_policy': 'Use the activated environment; SNN_PYTHON can select an explicit interpreter',
               'campaign_data_argument': '--data-root "$SNN_DATA"',
               'queue_location': str(paths['SNN_CAMPAIGN'] / 'train.sqlite')}
    archive = project / 'src' / 'benchmark_source.tar.gz'
    if archive.is_file():
        h = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(2**20), b''):
                h.update(chunk)
        receipt['uploaded_source_archive_sha256'] = h.hexdigest()
    env = '# Generated server path contract; source after activating the Python environment.\n'
    env += ''.join('export ' + key + '=' + shlex.quote(str(value)) + '\n' for key, value in paths.items())
    env += 'export SNN_PYTHON="${SNN_PYTHON:-python}"\n'

    scripts = {
        'benchmark.sh': COMMON + 'exec "$SNN_PYTHON" -m snnbench "$@"\n',
        'download_real.sh': COMMON + '''exec "$SNN_PYTHON" -m snnbench.download --out "$SNN_RAW" \\
  --ids D31 D32 D33 D35 D41 D46 D47 D89 D96 D97 D98 D99 --extract "$@"
''',
        'train.sh': COMMON + '''if [[ ! -f "$SNN_CAMPAIGN/campaign.json" || ! -f "$SNN_CAMPAIGN/train.sqlite" ]]; then
  printf '%s\\n' 'Materialize the campaign with --data-root "$SNN_DATA" before training.' >&2
  exit 2
fi
"$SNN_PYTHON" -c 'import json,sys; from pathlib import Path; p=json.load(open(sys.argv[1])); expected={"data_output":sys.argv[2],"training_output":sys.argv[3]}; mismatches=[k for k,v in expected.items() if Path(p[k]).resolve()!=Path(v).resolve()]; sys.exit("Campaign paths differ from server_paths.sh: "+str(mismatches)) if mismatches else None' \\
  "$SNN_CAMPAIGN/campaign.json" "$SNN_DATA" "$SNN_CAMPAIGN/runs"
''' + GPU_ARGS + '''exec "$SNN_PYTHON" scripts/launch_local_gpus.py --gpus "${SNN_GPU_IDS[@]}" \\
  --db "$SNN_CAMPAIGN/train.sqlite" --runs "$SNN_CAMPAIGN/runs" --data "$SNN_DATA"
''',
        'smoke.sh': COMMON + GPU_ARGS + '''"$SNN_PYTHON" -m snnbench campaign-plan --root "$SNN_SMOKE" \\
  --data-root "$SNN_SMOKE_DATA" --profiles smoke --seeds 11 --materialize
"$SNN_PYTHON" -m snnbench preflight --manifest "$SNN_SMOKE/manifests/smoke.jsonl" --data "$SNN_SMOKE_DATA"
"$SNN_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(1 if r["blocked"] else 0)' "$SNN_SMOKE/manifests/smoke.preflight.json"
"$SNN_PYTHON" scripts/launch_local_gpus.py --gpus "${SNN_GPU_IDS[@]}" \\
  --db "$SNN_SMOKE/train.sqlite" --runs "$SNN_SMOKE/runs" --data "$SNN_SMOKE_DATA" --analyze
"$SNN_PYTHON" -m snnbench summarize --runs "$SNN_SMOKE/runs" --out "$SNN_SMOKE/summary"
''',
        'postprocess.sh': COMMON + '''if (( $# == 0 )); then
  printf '%s\\n' 'Usage: bash postprocess.sh preliminary|all_shared [GPU_ID ...]' >&2
  exit 2
fi
SNN_POST_PHASE="$1"
shift
case "$SNN_POST_PHASE" in preliminary|all_shared) ;; *) exit 2 ;; esac
if [[ ! -f "$SNN_CAMPAIGN/campaign.json" ]]; then
  printf '%s\\n' 'Campaign has not been created.' >&2
  exit 2
fi
''' + GPU_ARGS + '''"$SNN_PYTHON" -m snnbench campaign-stage --root "$SNN_CAMPAIGN" \\
  --phase "$SNN_POST_PHASE" --checkpoints all --archive-batch "${SNN_ARCHIVE_BATCH:-8}"
SNN_POST_PIDS=()
stop_post() {
  trap - INT TERM
  if (( ${#SNN_POST_PIDS[@]} )); then
    kill -TERM "${SNN_POST_PIDS[@]}" 2>/dev/null || true
    for SNN_POST_PID in "${SNN_POST_PIDS[@]}"; do wait "$SNN_POST_PID" || true; done
  fi
  exit 130
}
trap stop_post INT TERM
for SNN_GPU in "${SNN_GPU_IDS[@]}"; do
  "$SNN_PYTHON" -m snnbench post-worker --db "$SNN_CAMPAIGN/post_${SNN_POST_PHASE}.sqlite" \\
    --device "cuda:$SNN_GPU" >> "$SNN_LOGS/post_${SNN_POST_PHASE}_gpu${SNN_GPU}.log" 2>&1 &
  SNN_POST_PIDS+=("$!")
done
SNN_POST_RC=0
for SNN_POST_PID in "${SNN_POST_PIDS[@]}"; do wait "$SNN_POST_PID" || SNN_POST_RC=1; done
trap - INT TERM
"$SNN_PYTHON" -m snnbench queue-status --db "$SNN_CAMPAIGN/post_${SNN_POST_PHASE}.sqlite"
"$SNN_PYTHON" -m snnbench summarize --runs "$SNN_CAMPAIGN/runs" \\
  --out "$SNN_CAMPAIGN/results_${SNN_POST_PHASE}"
exit "$SNN_POST_RC"
''',
        'status.sh': COMMON + '''for SNN_DB in train post_preliminary post_all_shared; do
  if [[ -f "$SNN_CAMPAIGN/$SNN_DB.sqlite" ]]; then
    printf '%s\\n' "$SNN_DB"
    "$SNN_PYTHON" -m snnbench queue-status --db "$SNN_CAMPAIGN/$SNN_DB.sqlite"
  fi
done
''',
    }
    files = {paths['SNN_CONFIGS'] / 'server_paths.sh': env,
             paths['SNN_CONFIGS'] / 'server_layout.json': json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'}
    files.update({project / 'scripts' / name: value for name, value in scripts.items()})
    changed = [p for p, value in files.items() if p.exists() and p.read_text() != value]
    if changed and not replace_config:
        raise FileExistsError('Existing server configuration differs. --replace-config keeps a backup then updates: ' + ', '.join(str(p) for p in changed))
    for directory in set(paths.values()) | {project / 'scripts', project / 'src'}:
        directory.mkdir(parents=True, exist_ok=True)
    for path, value in files.items():
        if path in changed:
            backup = path.with_name(path.name + '.previous-' + str(time.time_ns()))
            _atomic_text(backup, path.read_text())
        if not path.exists() or path.read_text() != value:
            _atomic_text(path, value, 0o755 if path.parent == project / 'scripts' else 0o644)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=DEFAULT_ROOT)
    parser.add_argument('--code-root')
    parser.add_argument('--campaign-name', default='campaign_v04')
    parser.add_argument('--replace-config', action='store_true')
    args = parser.parse_args()
    print(json.dumps(setup(args.project_root, args.code_root, args.campaign_name, args.replace_config), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
