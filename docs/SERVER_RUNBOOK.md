# 服务器执行手册

所有命令在仓库根目录运行。先查看当前分支/PR交付的版本，再安装项目；原main旧档案没有这些新入口。

v0.3新增真实预测/惯性/EEG/声音等适配与32种公共参考骨架。先读[真实数据及模型扩展](REAL_DATA_AND_MODEL_EXPANSION.md)和[近期基准复核](BENCHMARK_REVIEW_2025_2026.md)，使用`expanded`及显式`--archives`加入真实数据。旧checkpoint的严格代码hash不能跨版本强行续跑；新版本建议另用`--out runs_v03`。

## 1. 环境与CPU/GPU验收

推荐单独Python3.12环境。保留服务器与驱动匹配的PyTorch GPU版本。安装依赖会遵循当前pip环境；如已配好GPU框架，先确认 `python -c "import torch; print(torch.__version__,torch.cuda.is_available())"`。

```bash
python -m pip install -e '.[data]'
python -m snnbench doctor
python -m unittest discover -s tests -v
python -m snnbench compile --profile smoke --seeds 11 --out manifests/smoke.jsonl
python -m snnbench preflight --manifest manifests/smoke.jsonl
python -m snnbench queue-init --db smoke.sqlite --manifest manifests/smoke.jsonl
python -m snnbench worker --db smoke.sqlite --device cuda:0 --analyze
python -m snnbench queue-status --db smoke.sqlite
python -m snnbench summarize --runs runs --out results_summary
```

GPU smoke需要8项全部完成、参数确实更新、分析receipt生成且无非有限值。CPU通过后仍不能省掉这一轮。当前没有自动AMP、torch.compile、DDP或混合精度；优先固定可复现基线，再单独比较加速配置。

## 2. 数据准备

按[数据交接](DATA_DOWNLOAD.md)下载并转换，逐项执行 `verify-data`。输出中的manifest SHA与各角色实际样本数量应存档。十个数据角色必须按独立父来源划分，数量不足会报错。

```bash
python -m snnbench compile --profile screen --seeds 11 22 33 44 55 \
  --archives /data/snn/canonical/har /data/snn/canonical/shd \
  --out manifests/screen_with_real.jsonl
```

外部样本数按实际角色上限缩小，不循环重复。真实重建还需加入自己的canonical根目录。路径/数据版本不同必须有新的manifest，不能在同一目录替换文件后继续沿用旧ID。

## 3. 分阶段编译

```bash
python -m snnbench compile --profile screen --seeds 11 22 33 44 55 --out manifests/screen.jsonl
python -m snnbench compile --profile full --seeds 11 22 33 44 55 --out manifests/full.jsonl
python -m snnbench compile --profile paired --seeds 11 22 33 44 55 --out manifests/paired.jsonl
python -m snnbench compile --profile learning --seeds 11 22 33 44 55 --out manifests/learning.jsonl
```

各manifest是可执行配置行，不是任务说明占位。伴随receipt给实际去重数量与SHA。同一run出现在多个manifest时，在同一队列/输出根下按ID去重，避免重复训练。

`screen`先建立所有任务/骨架/单元锚点；`full`展开数据能力轴；`paired`展开机制、观察和读出；`learning`展开信用与编码。完整profile可能产生大量配置，编译本身不会启动训练。调参前先核对数量和存储估算。

## 4. 单机多GPU

SQLite队列必须放在该主机本地盘，多个GPU worker共用。已有CUDA_VISIBLE_DEVICES会被尊重；`--gpus`是该可见列表里的逻辑编号。

```bash
python -m snnbench queue-init --db /local/snn/full.sqlite --manifest manifests/full.jsonl
python scripts/launch_local_gpus.py --gpus 0 1 2 3 \
  --db /local/snn/full.sqlite --runs runs --data prepared --analyze
python -m snnbench queue-status --db /local/snn/full.sqlite
```

每GPU一个训练子进程，日志在runs/worker_gpu*.log与runs/_queue_specs/。data按任务身份共享，生成时持有目录租约，其余进程等它完成。进程异常退出后，本机已死亡PID的租约可自动恢复；跨主机不擅自删仍可能有效的锁。

首次发现OOM应记为失败并调整明确的batch配置/ID，不能在同一个run内静默降batch以改掉优化过程。采集batch可单独在analysis配置中缩小；拟合样本数和观测维数要保留原合同。

## 5. 多节点/Slurm

使用稳定run ID取模分片；每个节点/array job使用本地SQLite，所有节点读同一个只读manifest，结果写可共享存储。

```bash
sbatch --array=0-63 scripts/slurm_array.sbatch manifests/full.jsonl 64
```

脚本的GPU/CPU/内存/时限按实际集群修改。不要把SQLite放共享NFS来代替分布式调度器。重投同一分片时，模型检查点仍按ID续跑；最终台账按实际receipt去重。

## 6. 单次调试、恢复与冻结编码器

```bash
python -m snnbench init-spec --out configs/my_run.json
python -m snnbench train --spec configs/my_run.json --out runs --data prepared --device cuda:0
# 相同命令即续跑。max-updates只用于故意暂停工程验收。
python -m snnbench export-encoder --run runs/RUN_ID --out frozen_encoder.pt
```

将导出路径和返回SHA写入TrainSpec的encoder_checkpoint/encoder_checkpoint_sha256，其他模型使用同一输入形状/encoder结构。冻结encoder的来源训练预算和标签权限必须登记，不能当免费外部数据。

续跑会校验运行身份、代码SHA和运行环境，恢复优化器、随机状态、轮次顺序与游标。跨GPU型号、不同库版本或修改训练代码不声称逐比特精确续跑。需要迁移时建立明确的新实验，不能绕开校验冒充原运行。

失败任务保留日志：修复原因后再显式运行 `queue-retry --db ...`；该命令只重置failed，不改done。不要反复重试数据权限/合同错误来凑完成数量。

## 7. 自定义实验轴与公平调参

```bash
python -m snnbench sweep --base configs/my_run.json --grid examples/hpo_grid.json --out manifests/hpo.jsonl
```

grid为实际字段列表，例如model.width、train.lr、model.sg_scale。非法配置在receipt的blocked中列原因。只有支持可变深度的骨架才能改变depth；IF不允许有限DC增益归一。纯分析设置用analyze的独立参数，不通过增加训练数量制造更多模型。

相同宽度、参数量、状态字节、更新数与训练计算并不等价。先做固定宽度/深度的控制表，同时用实际参数和完整状态计数建立预算匹配表；为各族设相同的验证调参预算，再冻结选择，不用测试挑配置。

## 8. 分析与结果读取

```bash
python -m snnbench analyze --run runs/RUN_ID --checkpoint best.pt --data prepared --device cuda:0
python -m snnbench analyze --run runs/RUN_ID --data prepared --device cuda:0 \
  --modules working_point probes geometry temporal --fit-scenes 1024 --select-scenes 256 --eval-scenes 1024
python -m snnbench summarize --runs runs --out results_summary
```

每份analysis目录有：report.json、receipt.json、scene_statistics.npz、评价集完整native_outputs.npz，以及observations/{split}/的分批NPY、坐标index与labels_and_ids。可以重读中间表征、完整预测和真值，避免只留平均分。若只取了128个状态/诊断目标坐标，报告分别给出原维数与具体坐标；完整原生图像损失始终使用所有合法像素。需要完整诊断时增大max_features/max_targets并重新分析；不能将像素坐标自动视为独立潜在因素。

理论/探针与原生风险分开读取。优先看来源恢复随时间、请求负荷与不相关写入的变化，然后看编码/几何、时间迁移、局部干预和四组恢复是否形成一致解释。失败、不适用、负R²、有限交叉项和未定义恢复比都应保留。

## 9. 确认与最终审计

先把候选规律、作用方向、对照、预计有效/无效模型改动、主要指标与比较族写成版本化文档并提交，再显式使用 `--split confirmation`。确定主结论后才用 `--split audit`；默认训练和分析不会读取audit。

跨模型用相同任务世界与seed成对比较；跨seed以seed为重复单位。真实数据按父来源组考虑不确定性，不能把数百万重叠窗口当作独立样本。当前自动bootstrap是探索性边际区间，尚未代替确认阶段预先规定的多重比较控制与停止规则。

## 10. 存储与吞吐

先用每种真实形状的一次GPU smoke实测单步耗时、峰值显存、检查点体积和表征文件体积，再乘具体manifest数量。表征约随N×T×站点数×被采样维数增长；不要拿小向量结果估整幅图的存储。

训练中已保存初始、指定进度、best和last；若要对所有进度做分析，逐个 `--checkpoint step_*.pt`调用并使用相同场景/读出预算。第一轮可对全部终点做轻量分析，对机制深挖与轨迹保留明确的固定抽样和验证集，完整配置不因结果失败而丢弃。

调整max_features/max_targets等尚未作为CLI简写暴露的分析字段时，编辑该run目录spec.json中的analysis部分，再调用analyze；训练身份不变，原始训练配置仍保存在checkpoint。不要修改task/model/train部分来冒充同一个实验。
