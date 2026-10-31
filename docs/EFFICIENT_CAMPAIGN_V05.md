# 覆盖优先的 v05 训练计划与四卡调度

本次只新增外部规划/调度工具 `scripts/efficient_campaign.py`，不改动
`snnbench/` 的训练公式、初始化、训练预算、身份哈希、随机种子和检查点策略。
继续使用服务器现有 `bench` 环境及 v04 源码。无需重新上传仓库或数据。

## 1. 精确缩减的是什么

v04 的 555,405 项是独立训练配置，不是科学问题数量。v05 保留原任务条件集合、
基础模型库及局部机制交叉，把无差别的全局交叉改为预先声明的诊断锚点。
未执行的全局模型×条件交互仍保留在原数据库中，不能把它们称作已经验证。

使用当前 62 个基础合成合同、762 个 full 合成合同，以及 27 个真实合同形状的
本地夹具，实际生成 **43,885 个唯一训练身份**。服务器规划器必须再次读取真实
canonical 合同，并确认每个身份存在于原 v04 队列；任一缺失都会报错，不会近似匹配。

| 包 | 加入的配置数（包间有交集） | 范围 |
|---|---:|---|
| core | 21,705 | 62 基础合成＋27 真实合同，所有兼容参考模型，11/22/33 三种子 |
| full_condition_axes | 6,300 | 其余 700 个合成条件，SNN/连续单元/门控连续模型三锚点，三种子 |
| paired_local_factorial | 7,776 | 八个指定诊断任务，每个完整 324 项动力学/训练读出设置，三种子 |
| learning_local_factorial | 2,304 | 四个指定任务，每个完整 192 项学习设置，三种子 |
| cross_backbone_boundary | 1,992 | 八个预设边界条件，所有兼容基础模型，三种子 |
| five_seed_sentinels | 3,886 | 每个基础/真实合同的单骨架十二单元＋所有连续基线，额外 44/55 种子 |

合计 43,963 次加入，78 项交集按原始 `run_id` 去重，得到 43,885 项。
相对 555,405 项是 **12.66 倍训练数量缩减**，不是实测时间加速比。
精确更新步数预算由服务器计划报告；真实数据的样本数按原编译器规则封顶。

不缩短默认 60 epoch、不更改 batch=128、场景数、学习率、种子、模型宽度，
不删中间检查点、不缩减声明的完整坐标轨迹。已有有限数值风险仍是实际训练的结果，
不以 NaN 替换、截断或改成“通过”处理。

主结果使用共同的 11/22/33 三种子；44/55 是预先指定的确认对照，不做 best-of-five
筛选，也不把它们与三种子主比较混算。五个训练种子不是五套独立的数据生成实现。

### 诊断锚点

- paired：tagged_history 回归；G01/G02/G05/G06/G09 回归；G07/G08 重建。
- learning：Copy Memory 序列分类；G02 分类；G07 重建；真实 SSC 分类。
- 边界桥接：G01、G02 的两个预设极端；G05 高测量数下独立/重复采集；G06 两个频率距离极端。

这些选择不使用模型成绩，因此没有胜者筛选。所有任务上的训练读出/梯度交互
不是都被验证；要求更广泛结论时，从保留的旧配置中增加对应确认实验。
训练时变更 observation/readout/gradient 的配置保留独立训练，绝不由事后探针替代。
core 的 expanded 单元通常为 gain=0.5，原 paired/learning 额外配置通常为 gain=2。
本工具保留其原始设置，不能把不同包的结果不加匹配就混算为某个单因素效应。

## 2. 原始需求的可见状态

工具逐字节复制原 `all_layer_coverage.json`，保留原六层、40 张主卡、29 个专项小节及
其来源索引（以服务器实际原账本为准），并输出 `requirement_routes.json`。
这些条目保持 pending，不因训练已排队或 14 个共同模块存在就标为完成。

`training_coverage.json` 是已生成训练配置覆盖，不是科学结论或全局交互覆盖。
`analysis_plan.json` 保留全部共同分析模块、所有保存检查点及原完整场景库。
专项检查点适配器、作者原生模型实验、尚未准备的 OpenBMI/EarthNet 等来源仍需
各自实现/准备与执行；本工具不能把这些未完成项变成已完成。

## 3. 安装：只更新一个外部工具

在服务器上先 source 项目路径。将本次固定 commit 的该脚本下载到
`$SNN_PROJECT_ROOT/scripts/efficient_campaign.py`；应先核对发布时给出的 SHA256。
无需覆盖 `src/benchmark/snnbench/`，也无需重新 SCP 压缩包。

```bash
conda activate bench
cd /home/bohan_zhang/research/benchmark_temporal
source configs/server_paths.sh
python "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py" --help
```

不要因为计划目录改变就改写 `SNN_CAMPAIGN`：它仍指向旧 v04，数据和权重会复用。
以下显式使用新目录 `results/campaign_v05`。

## 4. 只生成新计划，旧训练继续

```bash
nohup python -u "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py" plan \
  --source-root "$SNN_CAMPAIGN" \
  --out "$SNN_PROJECT_ROOT/results/campaign_v05" \
  > "$SNN_LOGS/efficient_plan.log" 2>&1 < /dev/null &
tail -n 30 "$SNN_LOGS/efficient_plan.log"
```

计划目录必须是新的。45,000 是编译保护上限，超过就报错，不会抽掉配置来凑数。
本步骤只读旧数据库，输出新 manifest、队列、覆盖账本和预算。
检查 `selected_runs`、`package_memberships`、`training_coverage.json`。

## 5. 平滑切换，先让正在跑的任务结束

```bash
nohup python -u "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py" activate \
  --root "$SNN_PROJECT_ROOT/results/campaign_v05" --wait \
  > "$SNN_LOGS/efficient_activate.log" 2>&1 < /dev/null &
tail -n 30 "$SNN_LOGS/efficient_activate.log"
```

此步会：

1. 核对训练源码、清单和新数据库的科学身份一致。
2. 对旧 SQLite 做在线备份并检查备份完整性。
3. 把旧 pending 改为 `deferred_v05`，避免继续领取巨大旧清单；不删除旧任务。
4. 等原 worker/train 进程自然退出，不发终止信号。
5. 复用选中配置的已训练结果，检查 spec、status、last/best 检查点，保存校验记录。
6. 保留已有失败；所有不在新训练清单中的旧完成/失败结果也写入 `legacy_evidence.jsonl`，继续进入发现集。

已有权重和缓存留在原位置，避免复制几万份文件。
原进程正在跑一个很长任务时，本步骤会等待；日志列出在跑数量和 PID。
部分检查点恢复要求回原 GPU UUID；不会换卡后悄悄重新训练。

激活成功后旧 `status.sh` 会显示大量 `deferred_v05`，这是旧队列保留状态。
新进度应使用下面的 v05 status 命令。

## 6. 四张 84GB 卡：必须明确物理编号或 UUID

先检查：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

当前会话已确认本项目使用过 1/3/4，GPU2 曾运行另一个项目。第四张不能靠猜。
如果确认第四张是 GPU0，下面使用 `0 1 3 4`；否则替换成实际授权空闲卡编号。
工具只使用显式列表，选中卡上有其他计算进程时拒绝启动，不终止别人或其他项目。

先运行至多 16 个新的完整任务作为部署检查：

```bash
nohup python -u "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py" run \
  --root "$SNN_PROJECT_ROOT/results/campaign_v05" \
  --gpus 0 1 3 4 --max-slots 4 --reserve-gib 12 \
  --max-new-jobs 16 \
  > "$SNN_LOGS/efficient_pilot.log" 2>&1 < /dev/null &
```

这 16 项仍使用原训练预算、输出正式可用结果，不是把 8 步冒充完成训练。
它们只是调度部署检查，不代表所有形状和骨架已经过 GPU 资源验证。
它们结束后，继续整个队列：

```bash
nohup python -u "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py" run \
  --root "$SNN_PROJECT_ROOT/results/campaign_v05" \
  --gpus 0 1 3 4 --max-slots 4 --reserve-gib 12 \
  > "$SNN_LOGS/efficient_train.log" 2>&1 < /dev/null &
```

### 调度策略

- 四卡分别管理，不把 4×84GB 当一个共享显存池。
- 上限每卡四个小任务；中等任务消耗两个槽位，大任务独占一张卡。
- 分别检查实际空闲显存、80%卡内预算、12GiB预留及已经启动但尚未显现的显存预约。
- 使用成功完成任务的峰值 reserved memory 更新同资源签名的估计，并添加余量。
- 初始 2/8/24GiB 是保守入场估计，不是 GPU 实测承诺；OOM 是需诊断的失败，不自动改变 batch。
- CPU线程预算默认使用可见 CPU 的80%；遵守原每任务 TrainSpec.threads，不更改训练配置。
- 部分检查点绑定原UUID；缺少对应卡会输出 `admission_wait.json`，不跨卡伪装精确续跑。
- 同一调度器互斥。协调器意外退出时，已启动的训练子进程保留，下次启动可接管。
- 顺序优先覆盖不同任务和模型，先首种子，再其余主体种子及专项确认，不再按随机 run_id 逐个扫。
- 数值失败单独保留，不被自动重试到成功后覆盖。

本版仍是每个独立 run 新建训练进程。常驻训练进程、融合时间循环、遥测批量化
属于后续性能版本，未验证前不加入当前训练身份，也不承诺未经实测的额外倍数。

## 7. 状态、暂停、恢复

```bash
python "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py" status \
  --root "$SNN_PROJECT_ROOT/results/campaign_v05"

python "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py" pause \
  --root "$SNN_PROJECT_ROOT/results/campaign_v05"
```

pause 只停止领取新任务，已经在跑的任务自然结束。暂停后恢复时使用同样 GPU 列表，增加 `--resume`：

```bash
nohup python -u "$SNN_PROJECT_ROOT/scripts/efficient_campaign.py" run \
  --root "$SNN_PROJECT_ROOT/results/campaign_v05" \
  --gpus 0 1 3 4 --max-slots 4 --reserve-gib 12 --resume \
  >> "$SNN_LOGS/efficient_train.log" 2>&1 < /dev/null &
```

不要继续运行旧 add_gpu_workers.py 或旧 start/resume_full_campaign.sh。
原失败仍需区分数值机制、OOM和工程错误，再创建有明确原因的重试/补充实验。
SQLite 队列应放本机 SSD/NVMe；先用 `df -Th` 核对文件系统。如果 home 是网络盘，
应把新计划目录放本机存储，继续引用原数据/结果路径，而不是在网络盘上增加更多 worker。

## 8. 训练之后仍要做的事

现有 postprocessing 入口可以读取新 campaign.json 的原始权重、缓存路径。
按检查点实际就绪情况，使用原命令创建 preliminary / all_shared 后处理队列。
共同模块和原专项小节仍分别结账；这些不计入“43,885 次独立训练”。
作者原生仓库和未准备数据继续保留在原来源账本，不以独立参考模型冒充作者复现。

## 9. 验证边界

本地 unittest 覆盖实际任务生成器的 43,885 项设计、全部789个任务条件身份、
完整局部因素水平、旧数据库身份匹配、只读规划、预算拒绝而不抽样、迁移备份/自然排空、
结果与失败复用、四卡资源准入的模拟、其他进程保护、崩溃窗口进程接管。
GPU执行使用模拟器；本地没有验证用户的GPU吞吐、模型收敛或全部专项分析。
真实服务器需先完成计划核验及16项部署检查，再报告实测资源和时间。

English: v05 retains every existing task condition and the original scientific
training contracts, using explicit anchor factorials instead of a global Cartesian
product. It preserves the original six-layer ledger as pending work, reuses legacy
evidence without overwriting training source, and schedules explicitly selected
GPUs with independent memory/CPU budgets and exact-resume GPU affinity. The local
tests are CPU design/control tests with mocked GPU execution, not a GPU benchmark.
