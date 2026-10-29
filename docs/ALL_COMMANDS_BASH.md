# 全部执行命令 / Complete Bash command reference

更新：2026-10-30。以下适用于Linux/Bash，在已经可用的CUDA版PyTorch环境执行。代码来自`codex/l1-l4-unified-pipeline-20261029`分支；运行期间固定代码版本。SQLite队列所在的`SNN_ROOT`使用本机磁盘。

主流程依次为1—9；第10部分是专项和运维命令。`all_shared`执行当前14项公共分析。原计划其余专用子节的适配、第三步深入研究和第四步论文构思沿覆盖表继续进行。

## 1. 获取代码、安装和检查

先按实际服务器修改`SNN_ROOT`。下面克隆到新目录，已有实验使用原来的目录与版本。

```bash
export SNN_ROOT=/data/snn
export SNN_CODE="$SNN_ROOT/code/benchmark_v04"
export SNN_CAMPAIGN="$SNN_ROOT/campaign_v04"

mkdir -p "$SNN_ROOT/code" "$SNN_ROOT/raw" "$SNN_ROOT/canonical" "$SNN_ROOT/logs"

git clone --branch codex/l1-l4-unified-pipeline-20261029 --single-branch \
  https://github.com/LovelyC-kx/benchmark.git "$SNN_CODE"
cd "$SNN_CODE"
git rev-parse HEAD > "$SNN_ROOT/source_commit.txt"

python -c 'import sys,torch; print(sys.version); print(torch.__version__); assert torch.cuda.is_available(), "CUDA PyTorch required"; print(torch.cuda.device_count())'
python -m pip install -e '.[data]'
python -m snnbench doctor > "$SNN_ROOT/logs/doctor.json"
cat "$SNN_ROOT/logs/doctor.json"
nvidia-smi
df -h "$SNN_ROOT"

python -m unittest discover -s tests -v > "$SNN_ROOT/logs/unit_tests.log" 2>&1
tail -n 8 "$SNN_ROOT/logs/unit_tests.log"

mapfile -t SNN_GPU_IDS < <(python -c 'import torch; print(*range(torch.cuda.device_count()), sep="\n")')
declare -p SNN_ROOT SNN_CODE SNN_CAMPAIGN SNN_GPU_IDS > "$SNN_ROOT/session_env.sh"

python -m snnbench datasets --tier all > "$SNN_ROOT/logs/datasets.jsonl"
python -m snnbench models > "$SNN_ROOT/logs/models.json"
python -m snnbench models --sources > "$SNN_ROOT/logs/model_sources.json"
```

新终端在激活同一个Python环境后，先恢复路径与设备列表：

```bash
source /data/snn/session_env.sh
cd "$SNN_CODE"
```

## 2. 下载当前注册的直接文件

第一条会下载普通直接文件并解压压缩包；CSV保留在各ID的原目录。原16任务和G01—G09由生成器产生。

```bash
python -m snnbench.download --list --tier all
bash scripts/download_real_priority.sh "$SNN_ROOT/raw"
python -m snnbench.download --ids D31 D32 D33 D35 D41 D46 D47 D89 D96 D97 D98 D99 \
  --out "$SNN_ROOT/raw" --verify-only
```

作者原生轨道另外可下载5个可选包。它们与保留subject的原始数据适配分开使用：

```bash
python -m snnbench.download --ids D26 D47 D64 D66 D67 \
  --include-native --out "$SNN_ROOT/raw" --extract
```

HAR/WISDM外层包内可能另有压缩包。以下使用项目的解压器处理这两个目录内的嵌套容器：

```bash
python - <<'PY'
import os
from pathlib import Path
from snnbench.download import extract_archive
raw = Path(os.environ['SNN_ROOT']) / 'raw'
for dataset in ('D46', 'D47'):
    root = raw / dataset / 'extracted'
    seen = set()
    for _ in range(4):
        todo = [p for p in root.rglob('*') if p.is_file() and
                str(p) not in seen and p.name.lower().endswith(('.zip', '.tar.gz', '.tgz', '.tar'))]
        if not todo:
            break
        for path in todo:
            extract_archive(path, path.parent)
            seen.add(str(path))
            print('extracted:', path)
PY
```

## 3. 转换全部已接通的直接来源

```bash
python -m snnbench prepare har --root "$SNN_ROOT/raw/D46/extracted" \
  --out "$SNN_ROOT/canonical/har"
python -m snnbench prepare spikes --dataset SHD \
  --train "$SNN_ROOT/raw/D31/extracted/shd_train.h5" \
  --test "$SNN_ROOT/raw/D31/extracted/shd_test.h5" \
  --steps 100 --window 1.4 --out "$SNN_ROOT/canonical/shd"
python -m snnbench prepare spikes --dataset SSC \
  --train "$SNN_ROOT/raw/D32/extracted/ssc_train.h5" \
  --val "$SNN_ROOT/raw/D32/extracted/ssc_valid.h5" \
  --test "$SNN_ROOT/raw/D32/extracted/ssc_test.h5" \
  --steps 100 --window 1.4 --out "$SNN_ROOT/canonical/ssc"
python -m snnbench prepare gsc --root "$SNN_ROOT/raw/D33/extracted" \
  --out "$SNN_ROOT/canonical/gsc35"
python -m snnbench prepare gesture --root "$SNN_ROOT/raw/D35/extracted" \
  --steps 100 --window 3 --size 32 --out "$SNN_ROOT/canonical/gesture"

for SNN_MODE in sequential permuted repeat; do
  python -m snnbench prepare mnist --root "$SNN_ROOT/raw/D89/extracted" \
    --mode "$SNN_MODE" --permutation-seed 20261029 --steps 16 \
    --out "$SNN_ROOT/canonical/mnist_$SNN_MODE"
done

for SNN_FOLD in 1 2 3 4 5; do
  python -m snnbench prepare esc50 --root "$SNN_ROOT/raw/D41/extracted" \
    --test-fold "$SNN_FOLD" --out "$SNN_ROOT/canonical/esc50_f$SNN_FOLD"
done

for SNN_DEVICE in watch phone; do
  for SNN_SENSOR in gyro accel; do
    python -m snnbench prepare wisdm --root "$SNN_ROOT/raw/D47/extracted" \
      --device "$SNN_DEVICE" --sensor "$SNN_SENSOR" --window 200 --stride 100 \
      --out "$SNN_ROOT/canonical/wisdm_${SNN_DEVICE}_${SNN_SENSOR}"
  done
done

for SNN_PAIR in D96:ETTh1 D97:ETTh2 D98:ETTm1 D99:ETTm2; do
  SNN_DATA_ID="${SNN_PAIR%%:*}"
  SNN_DATA_NAME="${SNN_PAIR##*:}"
  for SNN_HORIZON in 24 96 192 336 720; do
    python -m snnbench prepare forecast-csv \
      --csv "$SNN_ROOT/raw/$SNN_DATA_ID/$SNN_DATA_NAME.csv" \
      --dataset-id "${SNN_DATA_ID}_L96_H${SNN_HORIZON}" \
      --context 96 --horizon "$SNN_HORIZON" --stride 24 --embargo 96 \
      --role-allocation minimum_first \
      --out "$SNN_ROOT/canonical/${SNN_DATA_NAME}_L96_H${SNN_HORIZON}"
  done
done
```

预测统一显式使用minimum_first十角色派生协议，支持长horizon；其分数与原生LTSF协议分别报告。五折ESC-50及四设备/传感器组合分别形成不同数据合同。

## 4. 作者文件到位后运行的转换

以下路径需要替换为服务器上的实际文件，再运行对应段落。OpenBMI要求保留subject的`data_N.mat/label_N.mat`；采样率填写这些输入文件的实际采样率。EarthNet要求作者转换的训练/测试HDF5。

```bash
python -m snnbench prepare openbmi --root /data/openbmi/eeg_data \
  --sampling-hz 1000 --downsample 10 --out "$SNN_ROOT/canonical/openbmi_subject"

python -m snnbench prepare earthnet --train /data/earthnet/train.hdf5 \
  --test /data/earthnet/ood_test_split.hdf5 --steps 6 --size 32 \
  --group-unit tile --out "$SNN_ROOT/canonical/earthnet_causal"

python -m snnbench prepare manifest --manifest /data/exported/reconstruction_manifest.json \
  --objective reconstruction --out "$SNN_ROOT/canonical/real_reconstruction"
python -m snnbench prepare manifest --manifest /data/exported/classification_manifest.json \
  --objective classification --classes 10 --out "$SNN_ROOT/canonical/real_classification"
python -m snnbench prepare manifest --manifest /data/exported/regression_manifest.json \
  --objective regression --out "$SNN_ROOT/canonical/real_regression"
```

manifest格式见`examples/tensor_manifest.schema-example.json`；视频、无线、GeoTIFF、原生event格式先由对应加载器导出。仅有数据目录或示例schema不能代替实际样本清单。

## 5. 验证并冻结本轮真实数据名单

完成本轮全部转换后执行。每份数据完整验证通过才写入名单。创建campaign后沿用同一名单；后续数据批次另设campaign根目录。

```bash
python - <<'PY'
import os
from pathlib import Path
from snnbench.prepare import verify_dataset
root = Path(os.environ['SNN_ROOT'])
archives = sorted(p.parent.resolve() for p in (root / 'canonical').glob('*/dataset.json'))
if not archives:
    raise SystemExit('No canonical datasets prepared')
for path in archives:
    verify_dataset(path)
    print('verified:', path)
(root / 'canonical_roots.txt').write_text(''.join(str(p) + '\n' for p in archives))
PY
mapfile -t SNN_ARCHIVES < "$SNN_ROOT/canonical_roots.txt"
cat "$SNN_ROOT/canonical_roots.txt"
```

## 6. GPU基础验收和完整归档测量

8项短训练用于工程验收，其独立路径不进入正式campaign。完整测试中的模型矩阵检查在CPU执行；后续按实际长序列/空间/真实形状继续实测GPU资源。

```bash
mkdir -p "$SNN_ROOT/smoke"
python -m snnbench campaign-plan --root "$SNN_ROOT/smoke" \
  --profiles smoke --seeds 11 --materialize
python -m snnbench preflight --manifest "$SNN_ROOT/smoke/manifests/smoke.jsonl" \
  --data "$SNN_ROOT/smoke/data"
python scripts/launch_local_gpus.py --gpus "${SNN_GPU_IDS[@]}" \
  --db "$SNN_ROOT/smoke/train.sqlite" \
  --runs "$SNN_ROOT/smoke/runs" --data "$SNN_ROOT/smoke/data" --analyze
python -m snnbench queue-status --db "$SNN_ROOT/smoke/train.sqlite"
python -m snnbench summarize --runs "$SNN_ROOT/smoke/runs" --out "$SNN_ROOT/smoke/summary"

SNN_SMOKE_RUN=$(python -c 'import os,pathlib; print(next((pathlib.Path(os.environ["SNN_ROOT"])/"smoke/runs").glob("*/spec.json")).parent)')
python -m snnbench archive --run "$SNN_SMOKE_RUN" --checkpoint best.pt \
  --data "$SNN_ROOT/smoke/data" --device cuda:0 --scenes 8 --batch-size 4
```

最后一条显式采8场景，只用于归档测量。正式队列采用合同中的完整场景库。

## 7. 全量设计、预检和训练

```bash
mapfile -t SNN_ARCHIVES < "$SNN_ROOT/canonical_roots.txt"
python -m snnbench campaign-plan --root "$SNN_CAMPAIGN" \
  --profiles full paired learning --seeds 11 22 33 44 55 \
  --archives "${SNN_ARCHIVES[@]}"
cat "$SNN_CAMPAIGN/campaign.json"
cat "$SNN_CAMPAIGN/hardware.json"

python -m snnbench campaign-plan --root "$SNN_CAMPAIGN" \
  --profiles full paired learning --seeds 11 22 33 44 55 \
  --archives "${SNN_ARCHIVES[@]}" --materialize

for SNN_PROFILE in full paired learning; do
  python -m snnbench preflight --manifest "$SNN_CAMPAIGN/manifests/$SNN_PROFILE.jsonl" \
    --data "$SNN_CAMPAIGN/data"
done

python - <<'PY'
import json,os
from pathlib import Path
root = Path(os.environ['SNN_CAMPAIGN'])
for name in ('full','paired','learning'):
    result = json.loads((root/'manifests'/f'{name}.preflight.json').read_text())
    print(name, result['checked_rows'], 'blocked:', result['blocked'])
    if result['blocked']:
        raise SystemExit('Resolve preflight blocked rows before launching')
PY

mkdir -p "$SNN_CAMPAIGN/logs"
nohup python scripts/launch_local_gpus.py --gpus "${SNN_GPU_IDS[@]}" \
  --db "$SNN_CAMPAIGN/train.sqlite" --runs "$SNN_CAMPAIGN/runs" \
  --data "$SNN_CAMPAIGN/data" > "$SNN_CAMPAIGN/logs/train_launcher.log" 2>&1 &
echo "$!" > "$SNN_CAMPAIGN/logs/train_launcher.pid"
```

查看进度；训练队列pending/running归零后执行下一节。也可以给归档分配空闲卡，在训练期间多次stage已完成模型。

```bash
python -m snnbench queue-status --db "$SNN_CAMPAIGN/train.sqlite"
tail -n 60 "$SNN_CAMPAIGN/logs/train_launcher.log"
nvidia-smi
du -sh "$SNN_CAMPAIGN"
```

## 8. 全保存进度归档与初步分析

该函数在当前可见的每张卡上启动一个后处理worker，并等待全部worker退出。训练已结束或已为后处理留出设备时使用。

```bash
snn_post_all_gpus() {
  local snn_post_phase="$1"
  local snn_post_db="$SNN_CAMPAIGN/post_${snn_post_phase}.sqlite"
  local snn_post_gpu snn_post_pid snn_post_rc=0
  local -a snn_post_pids=()
  for snn_post_gpu in "${SNN_GPU_IDS[@]}"; do
    python -m snnbench post-worker --db "$snn_post_db" \
      --device "cuda:$snn_post_gpu" \
      > "$SNN_CAMPAIGN/logs/post_${snn_post_phase}_gpu${snn_post_gpu}.log" 2>&1 &
    snn_post_pids+=("$!")
  done
  for snn_post_pid in "${snn_post_pids[@]}"; do
    wait "$snn_post_pid" || snn_post_rc=1
  done
  return "$snn_post_rc"
}

python -m snnbench campaign-stage --root "$SNN_CAMPAIGN" \
  --phase preliminary --checkpoints all --archive-batch 8
snn_post_all_gpus preliminary
python -m snnbench queue-status --db "$SNN_CAMPAIGN/post_preliminary.sqlite"
python -m snnbench summarize --runs "$SNN_CAMPAIGN/runs" \
  --out "$SNN_CAMPAIGN/results_stage1"
```

stage命令重复执行只追加新作业。默认归档所有声明场景和全部坐标；初析执行task_controls/working_point并保存原生预测、逐场景风险及其统计。

## 9. 全部14项公共分析和覆盖总账

```bash
python -m snnbench campaign-stage --root "$SNN_CAMPAIGN" \
  --phase all_shared --checkpoints all
snn_post_all_gpus all_shared
python -m snnbench queue-status --db "$SNN_CAMPAIGN/post_all_shared.sqlite"
python -m snnbench summarize --runs "$SNN_CAMPAIGN/runs" \
  --out "$SNN_CAMPAIGN/results_shared_analysis"
python -m json.tool "$SNN_CAMPAIGN/all_layer_coverage.json" \
  > "$SNN_CAMPAIGN/results_shared_analysis/original_plan_index.json"
```

公共模块为task_controls、working_point、mechanism、probes、geometry、temporal、communication、access、selective、interventions、recovery、reconstruction、anova、theory。其余专用分析逐项接入，实际输出回接总账。

## 10. 专项与运维命令

### 10.1 状态、失败重试和恢复

先处理日志中的具体原因，再重置失败作业；训练仍用同一个配置、版本和输出根恢复。

```bash
python -m snnbench queue-retry --db "$SNN_CAMPAIGN/train.sqlite"
# 随后重新执行第7节的同一训练启动命令。

python -m snnbench queue-retry --db "$SNN_CAMPAIGN/post_preliminary.sqlite"
snn_post_all_gpus preliminary
python -m snnbench queue-retry --db "$SNN_CAMPAIGN/post_all_shared.sqlite"
snn_post_all_gpus all_shared
```

单worker前台执行或调试：

```bash
python -m snnbench worker --db "$SNN_CAMPAIGN/train.sqlite" \
  --out "$SNN_CAMPAIGN/runs" --data "$SNN_CAMPAIGN/data" --device cuda:0 --one
python -m snnbench post-worker --db "$SNN_CAMPAIGN/post_preliminary.sqlite" \
  --device cuda:0 --one
```

### 10.2 单模型、完整归档及确认/审计

把RUN_ID替换为真实目录名。确认/审计在相应研究阶段运行。

```bash
export SNN_RUN="$SNN_CAMPAIGN/runs/RUN_ID"
python -m snnbench train --spec "$SNN_RUN/spec.json" \
  --out "$SNN_CAMPAIGN/runs" --data "$SNN_CAMPAIGN/data" --device cuda:0
python -m snnbench archive --run "$SNN_RUN" --checkpoint last.pt \
  --data "$SNN_CAMPAIGN/data" --device cuda:0 --batch-size 8
python -m snnbench analyze --run "$SNN_RUN" --checkpoint best.pt \
  --data "$SNN_CAMPAIGN/data" --device cuda:0 \
  --modules task_controls working_point mechanism probes geometry temporal \
    communication access selective interventions recovery reconstruction anova theory
python -m snnbench analyze --run "$SNN_RUN" --checkpoint best.pt \
  --data "$SNN_CAMPAIGN/data" --device cuda:0 \
  --fit-scenes 1024 --select-scenes 256 --eval-scenes 1024 \
  --batch-size 8 --max-features 512 --max-targets 512
python -m snnbench analyze --run "$SNN_RUN" --checkpoint best.pt \
  --data "$SNN_CAMPAIGN/data" --device cuda:0 --split confirmation
python -m snnbench analyze --run "$SNN_RUN" --checkpoint best.pt \
  --data "$SNN_CAMPAIGN/data" --device cuda:0 --split audit
python -m snnbench archive --run "$SNN_RUN" --checkpoint best.pt \
  --data "$SNN_CAMPAIGN/data" --device cuda:0 --batch-size 8 \
  --roles confirmation audit
```

分析样本数不得超过对应真实角色的实际样本数；变更预算会生成新分析身份。完整档案与坐标预算缓存独立。

### 10.3 自定义配置、扫描与冻结编码器

```bash
mkdir -p "$SNN_ROOT/custom"
python -m snnbench init-spec --out "$SNN_ROOT/custom/vector_regression.json"
python -m snnbench init-spec --classification --out "$SNN_ROOT/custom/vector_classification.json"
python -m snnbench init-spec --image --out "$SNN_ROOT/custom/image_reconstruction.json"
python -m snnbench init-spec --image --classification --out "$SNN_ROOT/custom/image_classification.json"
python -m snnbench sweep --base "$SNN_ROOT/custom/vector_regression.json" \
  --grid examples/hpo_grid.json --out "$SNN_ROOT/custom/hpo.jsonl"
python -m snnbench sweep --base "$SNN_ROOT/custom/vector_regression.json" \
  --grid examples/phase_neuron_grid.json --out "$SNN_ROOT/custom/phase_neuron.jsonl"
python -m snnbench queue-init --db "$SNN_ROOT/custom/hpo.sqlite" \
  --manifest "$SNN_ROOT/custom/hpo.jsonl"
python -m snnbench worker --db "$SNN_ROOT/custom/hpo.sqlite" \
  --out "$SNN_ROOT/custom/runs" --data "$SNN_ROOT/custom/data" --device cuda:0 --analyze
python -m snnbench export-encoder --run "$SNN_RUN" --checkpoint best.pt \
  --out "$SNN_ROOT/custom/frozen_encoder.pt"
```

冻结编码器的实际路径与返回SHA填写到新配置的train.encoder_checkpoint和encoder_checkpoint_sha256。只有该模型实际存在已训练编码器时可导出。

### 10.4 独立编译与多节点分片

campaign已包含full/paired/learning；下面用于独立设计或集群分片。

```bash
python -m snnbench compile --profile expanded --model-roster expanded \
  --seeds 11 22 33 44 55 --archives "${SNN_ARCHIVES[@]}" \
  --out "$SNN_ROOT/custom/expanded.jsonl"
python -m snnbench queue-init --db /local/snn/node_0.sqlite \
  --manifest "$SNN_CAMPAIGN/manifests/full.jsonl" --shard 0 --shards 64
python -m snnbench worker --db /local/snn/node_0.sqlite \
  --out "$SNN_CAMPAIGN/runs" --data "$SNN_CAMPAIGN/data" --device cuda:0
```

每节点使用自己的本地SQLite，shard取0—63中对应编号。跨节点读取同一manifest；数据/运行目录按集群共享或汇集安排。

### 10.5 作者原生模型入口

```bash
for SNN_SUITE in soul spikingtsf spikf spikemamba neuroseq seqsnn tslif spikingssm tsl; do
  python -m snnbench upstream-checkout --suite "$SNN_SUITE" \
    --out "$SNN_ROOT/upstream/$SNN_SUITE"
done
```

各套代码使用独立作者依赖环境、原生数据及实际运行JSON。如下为通用入口示例，JSON需按相应作者参数填写，模板格式见examples/tsl_native_plan.json：

```bash
python -m snnbench upstream-run --suite spikingtsf \
  --source "$SNN_ROOT/upstream/spikingtsf" \
  --config /data/snn/native_plans/spikingtsf.json \
  --out "$SNN_ROOT/native_receipts/spikingtsf_001"
```

9个suite的checkout/receipt入口已经实现；各作者完整实验配置及公共状态适配继续按来源清单补齐。

### 10.6 查询任一入口的完整参数

```bash
python -m snnbench --help
python -m snnbench.download --help
for SNN_COMMAND in doctor datasets models campaign-plan campaign-stage post-worker post-job \
  archive prepare verify-data compile preflight train analyze summarize queue-init worker \
  queue-status queue-retry init-spec sweep export-encoder upstream-checkout upstream-run; do
  python -m snnbench "$SNN_COMMAND" --help
done
for SNN_ADAPTER in har spikes gsc gesture mnist esc50 wisdm openbmi forecast-csv earthnet manifest; do
  python -m snnbench prepare "$SNN_ADAPTER" --help
done
```
