# benchmark_temporal服务器目录与上传 / Server layout and upload

日期：2026-10-30。服务器项目根目录固定为：

```text
/home/bohan_zhang/research/benchmark_temporal
```

保留现有六个目录，新增`configs/`存路径合同和实验配置。完整仓库放在`src/benchmark/`，包括根目录研究文档、历史L3/L4目录、统一包和脚本；原计划索引和部分来源解析需要这些文件。

## 1. 目录对应

| 路径，相对项目根目录 | 用途 |
|---|---|
| src/benchmark/ | 本GitHub仓库的完整版本 |
| src/upstream/ | 已固定版本的作者模型/benchmark源码 |
| scripts/ | 自动生成的服务器启动、下载、查询和后处理入口 |
| configs/server_paths.sh | 所有服务器绝对路径，激活环境后source |
| configs/server_layout.json | 路径、源码revision和上传包SHA的记录 |
| configs/canonical_roots.txt | 本轮验证通过的真实数据根目录名单 |
| configs/native_plans/ | 作者原生运行JSON；按实际任务填写 |
| data/raw/ | 下载原包与解压文件 |
| data/canonical/ | 转换、校验后的十角色真实数据 |
| data/cache/campaign_v04/ | 正式实验的数据生成/采样缓存 |
| data/cache/smoke/ | 独立工程短验收缓存 |
| logs/ | 环境、下载、启动器和后处理主日志 |
| results/smoke/ | 短验收的队列、检查点和分析 |
| results/campaign_v04/ | 正式队列、manifest、来源总账和汇总 |
| results/campaign_v04/runs/RUN_ID/ | 权重、优化器/RNG、逐更新日志、archive/和analysis/ |

逐run/逐job的日志随对应结果保留；主启动和后处理日志写到项目`logs/`。SQLite队列所在results应位于单主机本地文件系统；若home实际挂载NFS，先将队列/结果目录映射到本地SSD再启动worker。

## 2. 本地Windows：拉取最终分支并打包

在PowerShell执行，修改本地目录与服务器地址。使用`git archive`生成Linux源码包，保留Git记录的文件内容、换行和权限，不包含本机环境与未跟踪文件。

```powershell
$SnnLocalRoot = "D:\research\benchmark_temporal"
$SnnRepo = Join-Path $SnnLocalRoot "src\benchmark"
$SnnArchive = Join-Path $SnnLocalRoot "benchmark_source.tar.gz"
$SnnRevision = Join-Path $SnnLocalRoot "benchmark_source.commit"
$SnnServer = "bohan_zhang@YOUR_SERVER_IP"

New-Item -ItemType Directory -Force -Path (Join-Path $SnnLocalRoot "src") | Out-Null

git -c core.longpaths=true clone --single-branch `
  --branch codex/l1-l4-unified-pipeline-20261029 `
  https://github.com/LovelyC-kx/benchmark.git $SnnRepo
if ($LASTEXITCODE -ne 0) { throw "Git clone failed" }

git -C $SnnRepo archive --format=tar.gz "--output=$SnnArchive" HEAD
if ($LASTEXITCODE -ne 0) { throw "Git archive failed" }
git -C $SnnRepo rev-parse HEAD | Set-Content -Encoding ascii $SnnRevision
Get-FileHash -Algorithm SHA256 $SnnArchive

ssh $SnnServer "mkdir -p /home/bohan_zhang/research/benchmark_temporal/src/benchmark"
if ($LASTEXITCODE -ne 0) { throw "Remote mkdir failed" }
scp $SnnArchive $SnnRevision "${SnnServer}:/home/bohan_zhang/research/benchmark_temporal/src/"
if ($LASTEXITCODE -ne 0) { throw "SCP failed" }
```

已有本地clone时使用其实际路径作为`$SnnRepo`，核对目标分支后获取最新提交，再执行archive和scp。SSH端口非22时，对ssh使用`-p PORT`，对scp使用`-P PORT`。

本地Linux/macOS也可以直接从Git生成同样的包：

```bash
git clone --single-branch --branch codex/l1-l4-unified-pipeline-20261029 \
  https://github.com/LovelyC-kx/benchmark.git benchmark_source
git -C benchmark_source archive --format=tar.gz --output="$PWD/benchmark_source.tar.gz" HEAD
git -C benchmark_source rev-parse HEAD > benchmark_source.commit
ssh bohan_zhang@YOUR_SERVER_IP 'mkdir -p /home/bohan_zhang/research/benchmark_temporal/src/benchmark'
scp benchmark_source.tar.gz benchmark_source.commit \
  bohan_zhang@YOUR_SERVER_IP:/home/bohan_zhang/research/benchmark_temporal/src/
```

## 3. 服务器：解压与初始化目录

这是首次部署的目录。运行中的实验保留其原源码版本；后续升级使用版本目录和对应路径合同。

```bash
export SNN_PROJECT_ROOT=/home/bohan_zhang/research/benchmark_temporal
mkdir -p "$SNN_PROJECT_ROOT/src/benchmark"

tar -xzf "$SNN_PROJECT_ROOT/src/benchmark_source.tar.gz" \
  -C "$SNN_PROJECT_ROOT/src/benchmark"

python3 "$SNN_PROJECT_ROOT/src/benchmark/scripts/setup_server_layout.py" \
  --project-root "$SNN_PROJECT_ROOT"

source "$SNN_PROJECT_ROOT/configs/server_paths.sh"
bash "$SNN_PROJECT_ROOT/scripts/benchmark.sh" --help
cat "$SNN_PROJECT_ROOT/configs/server_layout.json"
```

初始化脚本仅需Python标准库，可在GPU框架安装前执行。它创建子目录、路径合同和六个外层入口，按上传包附带的commit记录源码版本。重复相同配置可直接执行；修改已有配置时，`--replace-config`会先保留previous备份再更新。

```bash
python3 "$SNN_CODE/scripts/setup_server_layout.py" \
  --project-root "$SNN_PROJECT_ROOT" --replace-config
```

## 4. 环境准备与数据下载

使用独立主环境；PyTorch GPU构建按服务器驱动选择。作者模型的特殊依赖放各自环境。以下收集实际环境信息：

```bash
nvidia-smi
conda --version
python -c 'import sys,platform; print(platform.platform()); print(sys.version); print(sys.executable)'
```

创建并激活主环境、安装可用GPU版PyTorch后，再安装项目依赖：

```bash
source /home/bohan_zhang/research/benchmark_temporal/configs/server_paths.sh
python -m pip install -e "${SNN_CODE}[data]"
bash "$SNN_PROJECT_ROOT/scripts/benchmark.sh" doctor > "$SNN_LOGS/doctor.json"
cat "$SNN_LOGS/doctor.json"
```

下载当前全部普通直接文件，含CSV解压修复：

```bash
nohup bash "$SNN_PROJECT_ROOT/scripts/download_real.sh" \
  > "$SNN_LOGS/download_real.log" 2>&1 &
tail -n 40 "$SNN_LOGS/download_real.log"
```

格式转换沿[全部命令行](ALL_COMMANDS_BASH.md)的数据部分执行，路径使用当前合同：旧`$SNN_ROOT/raw`改用`$SNN_RAW`，旧`$SNN_ROOT/canonical`改用`$SNN_CANONICAL`。例如：

```bash
bash "$SNN_PROJECT_ROOT/scripts/benchmark.sh" prepare har \
  --root "$SNN_RAW/D46/extracted" --out "$SNN_CANONICAL/har"
bash "$SNN_PROJECT_ROOT/scripts/benchmark.sh" verify-data --root "$SNN_CANONICAL/har"
```

## 5. 作者源码统一落到src/upstream

如下读取注册表、按仓库与commit去重，下载当前21个固定仓库，并保存回执。已有目录会核对commit与tracked修改；其他候选继续保留在来源表中。

```bash
python - <<'PY'
import json,os,subprocess
from pathlib import Path
code=Path(os.environ['SNN_CODE'])
out=Path(os.environ['SNN_UPSTREAM'])
registry=json.loads((code/'snnbench/resources/model_sources.json').read_text())
seen=set(); receipts=[]
for item in registry['external']:
    repository=item.get('repository'); commit=item.get('commit')
    if not repository or not commit:
        continue
    key=(repository.removesuffix('.git').rstrip('/').lower(),commit)
    if key in seen:
        continue
    seen.add(key); target=out/item['source']
    if not target.exists():
        target.mkdir(parents=True)
        for args in (['init'],['remote','add','origin',repository],
                     ['fetch','--depth','1','origin',commit],['checkout','--detach','FETCH_HEAD']):
            subprocess.run(['git','-C',str(target),*args],check=True)
    actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(target),'status','--porcelain','--untracked-files=no'],text=True)
    if actual!=commit or dirty:
        raise RuntimeError('Existing source differs from the pinned revision: '+str(target))
    receipts.append({'source':item['source'],'repository':repository,'commit':commit,'path':str(target)})
    print('ready:',item['source'],commit,flush=True)
(Path(os.environ['SNN_CONFIGS'])/'upstream_sources.json').write_text(json.dumps(receipts,indent=2)+'\n')
print('Pinned repositories:',len(receipts))
PY
```

这里获取作者源码，随后按各自依赖/数据协议运行；其中9套已有upstream-run回执入口。

## 6. 工程短验收

主环境可用后执行；无GPU编号参数时使用全部当前可见卡，也可明确写`0`或`0 1`。

```bash
bash "$SNN_PROJECT_ROOT/scripts/smoke.sh" 0 > "$SNN_LOGS/smoke.log" 2>&1
tail -n 30 "$SNN_LOGS/smoke.log"
```

短验收输出仅在`results/smoke`和`data/cache/smoke`。主训练与后处理都使用正式路径。

## 7. 正式计划、训练和后处理

本轮全部canonical数据完成校验后，冻结名单：

```bash
python - <<'PY'
import os
from pathlib import Path
from snnbench.prepare import verify_dataset
archives=sorted(p.parent.resolve() for p in Path(os.environ['SNN_CANONICAL']).glob('*/dataset.json'))
if not archives:
    raise SystemExit('Prepare real canonical data before materializing the campaign')
for path in archives:
    verify_dataset(path)
(Path(os.environ['SNN_CONFIGS'])/'canonical_roots.txt').write_text(''.join(str(p)+'\n' for p in archives))
PY
mapfile -t SNN_ARCHIVES < "$SNN_CONFIGS/canonical_roots.txt"

bash "$SNN_PROJECT_ROOT/scripts/benchmark.sh" campaign-plan \
  --root "$SNN_CAMPAIGN" --data-root "$SNN_DATA" \
  --profiles full paired learning --seeds 11 22 33 44 55 \
  --archives "${SNN_ARCHIVES[@]}" --materialize
```

`--data-root`已接通，campaign保存该位置，后处理读取同一个字段。train.sh在启动前校验路径合同，避免训练与归档读到不同缓存位置。

```bash
for SNN_PROFILE in full paired learning; do
  bash "$SNN_PROJECT_ROOT/scripts/benchmark.sh" preflight \
    --manifest "$SNN_CAMPAIGN/manifests/$SNN_PROFILE.jsonl" --data "$SNN_DATA"
done
python - <<'PY'
import json,os
from pathlib import Path
for name in ('full','paired','learning'):
    p=Path(os.environ['SNN_CAMPAIGN'])/'manifests'/f'{name}.preflight.json'
    r=json.loads(p.read_text()); print(name,r['checked_rows'],r['blocked'])
    if r['blocked']:
        raise SystemExit('Resolve blocked preflight rows before training')
PY

nohup bash "$SNN_PROJECT_ROOT/scripts/train.sh" \
  > "$SNN_LOGS/train_launcher.log" 2>&1 &
bash "$SNN_PROJECT_ROOT/scripts/status.sh"
```

训练结束或已给后处理分配空闲卡后执行：

```bash
bash "$SNN_PROJECT_ROOT/scripts/postprocess.sh" preliminary
bash "$SNN_PROJECT_ROOT/scripts/postprocess.sh" all_shared
bash "$SNN_PROJECT_ROOT/scripts/status.sh"
```

postprocess.sh会stage全部保存进度、为所选可见卡启动worker、汇总实际结果。`preliminary`包含完整归档与初析；`all_shared`执行14项共享分析，原计划专用小节继续沿总账逐项接入。

本次64项CPU测试通过，六个生成脚本和本页Bash段落通过语法检查，包含外部缓存的实际训练/后处理；[工程回执](verification/2026-10-30/bohan_server_layout/summary.json)记录范围。用户服务器尚未由这里连接或启动训练。

English summary: Keep the complete repository in src/benchmark; keep pinned author repositories in src/upstream. Add configs for path contracts, use data for raw/canonical/cache, and results for checkpoints, queues and analysis artifacts. Generate the outer launch scripts after uploading the Git archive.
