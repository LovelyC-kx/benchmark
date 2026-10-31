# 服务器执行手册

所有命令在仓库根目录运行。先查看当前分支/PR交付的版本，再安装项目；原main旧档案没有这些新入口。

当前用户目录使用[benchmark_temporal专用部署](BOHAN_SERVER_LAYOUT.md)，完整代码置于`src/benchmark`，外层脚本会自动切换工作目录；`campaign-plan --data-root`将缓存定向到项目`data/cache`。

按顺序复制执行的完整版本见[全部命令行](ALL_COMMANDS_BASH.md)，其中包含全部已接通真实来源的转换循环、多GPU后处理和专项/恢复命令。

v0.4的首要入口见[四阶段执行方案](FOUR_PHASE_EXECUTION_PLAN.md)。84GB资源按实际设备字节检测；先广泛跑并完整保存，随后执行原L1—L6全部分析。真实数据及32种公共参考骨架见[扩展说明](REAL_DATA_AND_MODEL_EXPANSION.md)，作者轨道见[近期基准复核](BENCHMARK_REVIEW_2025_2026.md)。新实验另用`campaign_v04`，旧checkpoint在原代码版本读取。

## 0. 全量campaign入口

以下命令生成完整配置、来源/硬件/原计划索引，并在同一个队列中按run ID去重。真实数据用`--archives`加入，去掉`--materialize`可先查看设计而不编译大清单。

```bash
python -m snnbench campaign-plan --root /local/snn/campaign_v04 \
  --profiles full paired learning --seeds 11 22 33 44 55 \
  --archives /data/snn/canonical/har /data/snn/canonical/shd --materialize
python scripts/launch_local_gpus.py --gpus 0 1 2 3 \
  --db /local/snn/campaign_v04/train.sqlite \
  --runs /local/snn/campaign_v04/runs --data /local/snn/campaign_v04/data
python -m snnbench campaign-stage --root /local/snn/campaign_v04 \
  --phase preliminary --checkpoints all --archive-batch 8
python -m snnbench post-worker \
  --db /local/snn/campaign_v04/post_preliminary.sqlite --device cuda:0
python -m snnbench campaign-stage --root /local/snn/campaign_v04 \
  --phase all_shared --checkpoints all
python -m snnbench post-worker \
  --db /local/snn/campaign_v04/post_all_shared.sqlite --device cuda:0
```

`campaign-stage`重复执行会加入新完成模型；post-worker处理当前队列后退出。训练、完整归档和后处理按卡分配；不要直接让示例中的进程争用同一张卡。`preliminary`默认保存所有声明场景和所有保存进度；`all_shared`执行14项公共分析，原专用小节另沿`all_layer_coverage.json`逐项完成。

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

每份analysis目录有：report.json、receipt.json、scene_statistics.npz、评价集完整native_outputs.npz、fitted_readers/，以及observations/{split}/的分批NPY、坐标index与labels_and_ids。已接通的线性/非线性读出、标准化/实际投影、几何和transport拟合参数列出文件SHA。可以重读中间表征、完整预测、真值与拟合器。若只取了128个状态/诊断目标坐标，报告分别给出原维数与具体坐标；原生图像损失使用所有合法像素。

完整档案独立于快速分析缓存：

```bash
python -m snnbench archive --run runs/RUN_ID --checkpoint best.pt \
  --data prepared --device cuda:0 --batch-size 8
```

archive保存全部坐标和原精度，每时步状态、cache、读出及全部叶子模块输入输出；固定场景库数量写入合同。工程测量可显式`--scenes 8`；正式全量不加该覆盖参数。追加诊断预算可用analyze的`--max-features`、`--max-targets`、`--batch-size`，实际设置进入独立分析ID。

理论/探针与原生风险分开读取。优先看来源恢复随时间、请求负荷与不相关写入的变化，然后看编码/几何、时间迁移、局部干预和四组恢复是否形成一致解释。失败、不适用、负R²、有限交叉项和未定义恢复比都应保留。

## 9. 确认与最终审计

先把候选规律、作用方向、对照、预计有效/无效模型改动、主要指标与比较族写成版本化文档并提交，再显式使用 `--split confirmation`。确定主结论后才用 `--split audit`；默认训练和分析不会读取audit。

跨模型用相同任务世界与seed成对比较；跨seed以seed为重复单位。真实数据按父来源组考虑不确定性，不能把数百万重叠窗口当作独立样本。当前自动bootstrap是探索性边际区间，尚未代替确认阶段预先规定的多重比较控制与停止规则。

## 10. 存储与吞吐

先用每种实际形状实测单步耗时、峰值显存、检查点、逐更新日志、完整轨迹和拟合器体积，再按实际唯一run/进度/场景数汇总。完整轨迹按全部坐标估算，同时记录逻辑字节与压缩落盘字节。二值位压缩、内容去重与精确KV前缀复用已经接通；空间叶子输出按实际形状测量。

训练中保存初始、指定进度、best和last；campaign-stage默认将所有这些进度加入归档与初析队列。每次更新的样本/损失、梯度统计和资源记录在updates.jsonl，失败现场单独保存。完整轨迹采集固定场景库的全部时步，所有训练batch的激活按检查点、RNG和确定性数据重放追加。

改变分析配置会生成新的analysis身份；训练身份和checkpoint保留。旧原计划的专用分析逐项适配已训练检查点，并把实际输出回写覆盖总账。
