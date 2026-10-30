# SNN时序能力判断与归因基准

当前执行入口是 `snnbench/`，版本0.4.0，更新于2026-10-30。研究按四步推进：**广泛训练与完整留存 → 原计划全部层级分析 → 深入发现、机制和理论 → 据实际发现组织论文。** 旧L3/L4目录保留为研究档案，连同原Master一起进入逐项覆盖总账。

研究对象是：**任务需要哪种历史 → 系统实际保留什么 → 内容如何组织和传递 → 原下游实际使用什么，以及怎样改变它。** 各层复用同一数据世界、训练检查点和来源合同。

| 先读哪份文件 | 内容 |
|---|---|
| [当前服务器目录与SCP上传](docs/BOHAN_SERVER_LAYOUT.md) | 匹配`/home/bohan_zhang/research/benchmark_temporal`，初始化路径与外层启动脚本 |
| [四阶段全量执行方案](docs/FOUR_PHASE_EXECUTION_PLAN.md) | 实验空间、84GB资源、完整归档、L1—L6覆盖、发现链与论文形成；当前首要执行方案 |
| [全部命令行](docs/ALL_COMMANDS_BASH.md) | 从获取代码、真实数据转换、GPU验收到全量训练、归档分析、重试与作者模型入口的Bash命令 |
| [当前进展与边界](docs/CURRENT_STATUS.md) | 已实现、已验收、待服务器数据/GPU验证分别列出 |
| [实验与分析设计](docs/EXPERIMENT_DESIGN.md) | L1—L4逐模块、变量、对照、读出和理论范围 |
| [服务器执行](docs/SERVER_RUNBOOK.md) | 环境、下载、转换、多GPU/Slurm、恢复及结果读取 |
| [近期基准与作者代码复核](docs/BENCHMARK_REVIEW_2025_2026.md) | STP/NSA/NeuroBench/Soul/SpikingTSF与2025—2026重要模型；相近之处、差异和复用 |
| [真实数据与模型扩展](docs/REAL_DATA_AND_MODEL_EXPANSION.md) | 真实数据适配、14种注入骨架、18种连续参照、作者原生轨道与缺口 |
| [数据下载说明](docs/DATA_DOWNLOAD.md) | 24个注册文件（5个可选作者包）、校验、转换命令 |
| [全部121项来源/协议](docs/DATASETS_ALL.md) | 112项来源/任务/素材＋G01—G09；不能计作121个真实数据集 |
| [原研究计划到代码的对应](docs/COVERAGE_AND_PROVENANCE.md) | 文献/旧实现复用、改动、本轮新增及未移植边界 |

## 安装与第一条运行

推荐Python3.12。先在服务器安装与驱动匹配的PyTorch GPU版本，再安装本项目；不要把本地CPU验收环境复制为GPU环境。

```bash
python -m pip install -e '.[data]'
python -m snnbench doctor
python -m unittest discover -s tests -v
python -m snnbench compile --profile smoke --seeds 11 --out manifests/smoke.jsonl
python -m snnbench preflight --manifest manifests/smoke.jsonl
python -m snnbench queue-init --db smoke.sqlite --manifest manifests/smoke.jsonl
python -m snnbench worker --db smoke.sqlite --device cuda:0 --analyze
python -m snnbench summarize --runs runs --out results_summary
```

没有GPU时明确使用 `--device cpu`。请求CUDA而环境不可用会报错。`smoke`只有8个短训练验收，不代表论文性能或收敛。

## 下载数据

下载器只用Python标准库，也可脱离训练环境单独运行。

```bash
python -m snnbench.download --list --tier all
python -m snnbench.download --ids D31 D32 D33 D35 D46 --out downloads --extract
bash scripts/download_real_priority.sh downloads
```

默认不下载任何数据。EarthNet/U-TILISE、视频、无线等沿作者入口获取后使用对应转换接口；候选条目不会被编译器悄悄替换成合成数据。

## 正式设计入口

广泛全量运行优先使用v0.4的campaign入口。下面会生成完整设计、硬件/原计划索引，并物化扩展模型名单的full、paired、learning配置到同一个去重队列；去掉`--materialize`可先检查计划。

```bash
python -m snnbench campaign-plan --root /data/snn/campaign_v04 \
  --profiles full paired learning --seeds 11 22 33 44 55 --materialize
```

真实数据通过`--archives`显式加入。三族合成设计共524,770配置/五seed，尚未跨族去重；该数量不代表已训练结果。全进度归档、初析和14项公共分析使用独立后处理队列，详见[四阶段方案](docs/FOUR_PHASE_EXECUTION_PLAN.md)。原L1—L6专用小节沿覆盖表继续接通和执行。

单独编译入口继续保留，可用`--model-roster expanded`切换扩展名单：

```bash
python -m snnbench compile --profile screen --seeds 11 22 33 44 55 --out manifests/screen.jsonl
python -m snnbench compile --profile full --seeds 11 22 33 44 55 --out manifests/full.jsonl
python -m snnbench compile --profile expanded --seeds 11 22 33 44 55 --out manifests/expanded.jsonl
```

`screen`覆盖原16任务、分类/连续目标、12单元、8参考骨架、连续基线、tagged、符号任务和G01—G09。`full`增加内容/需求、负荷、年龄、写入、能量、版本、频率、采样等受控轴。`paired`增加机制/观察/读出对照；`learning`增加梯度、SG、reset梯度与编码对照。具体数量由伴随receipt计算，配置数不等于完成模型数。

所有参考骨架和单元是独立实现；官方模型复现需要另外固定原作者代码、数据与训练协议。本文不把简单对角SSM称作S4/Mamba，也不把逐帧spike attention称作完整时间Transformer。

`expanded`使用新的14种可注入骨架和18种连续参照、output访问、gain=0.5；62任务设置每seed 4954配置。真实数据先转换/verify，再通过`--archives`加入；不会生成替身。重要官方模型通过`models --sources`和`upstream-checkout`/`upstream-run`独立执行，不能把参考实现当作官方复现。

v0.4新增完整状态档案、逐更新训练证据、关键拟合器保存、四阶段执行队列及原六层覆盖索引。严格source hash按版本校验；旧结果用原源码分析，新训练另用目录（如`campaign_v04`）。[工程验收记录](docs/verification/2026-10-29/campaign_v04/README.md)与历史v0.3记录分别保存。
