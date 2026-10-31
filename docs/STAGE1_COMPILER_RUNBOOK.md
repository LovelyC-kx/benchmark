# 第一阶段编译器：全部层级、训练依赖与专项缺口

日期：2026-10-31。先编译，不修改旧训练队列，不启动GPU。

本工具落实TWO_STAGE_ALL_LAYER_ANALYSIS.md：在较少的匹配实验中保留全部层级责任；重要发现决定深挖顺序。编译器不把未适配的小节、未执行的分析或未准备的真实数据标为完成。

## 编译结果与验证范围

当前实际任务/模型工厂生成492个唯一训练身份，三个训练种子11/22/33，原60 epoch、batch128及检查点策略保持。默认样本上限不降低，真实合同按原有规则封顶。

在使用原目录模型规则、27个真实合同形状夹具构造的旧队列测试中，429项属于原有配置，63项是明确新增的科学配置。服务器必须用真实合同和旧数据库重新核验；429不是已经完成429项。已完成的匹配结果须经检查点、源代码及预算核验才可复用。

新增配置包括物理宽度16的容量对照，以及在相同gain/observation设置下补齐的梯度、SG/reset梯度、编码、归一化、学习率、读出观察和DC归一化对照。它们不假装成旧配置，也不复用不匹配的权重。

| 作业/账本 | 当前数量 | 说明 |
|---|---:|---|
| 独立训练身份 | 492 | 组间按原run_id去重，已完成项可经核验复用 |
| 共同分析配方 | 732 | 492个best检查点主分析＋240个同模型形成过程分析；尚未执行 |
| 完整轨迹归档请求 | 492 | 每个请求需在训练后展开实际保存的全部检查点；不是492个已归档文件 |
| 原主卡 | 40 | 36个六层分析主卡＋4个治理主卡 |
| 根协议及JSON专项编号 | 92 | 原JSON清单29项不穷尽协议；当前明确标题合并后所得 |

上述训练/共同分析数量在本地使用合同夹具验证，不是服务器真实语料、GPU速度或收敛验证。归档保持全部坐标、原始张量精度和原场景银行；文件数、存储量和时间需实际展开与测量。

## 第一阶段的科学分组

核心机制48项优先完成内容×载体×发放/重置对照，包含四种二维交叉单元、soft reset、ALIF、双状态及GRU；当前目标/历史目标两任务均有三个种子。随后补空白年龄×额外写入、学习规则、空间重建、物理状态量、真实SHD/HAR、关联查询、次序/间隔、更新保护、新增观测、物理时间、频率和骨架对照。

所有实验组的具体科学问题、模型与逐节依赖分别写入scientific_groups.json和requirement_dependencies.json。L3七节、L4五节，以及L5与L6全部进入第一阶段责任表，不能以共同模块完成代替每个原专项闭环。

容量组显式改变物理宽度，未将探针坐标数充当N；当前原生损失会对请求坐标平均，固定anchor训练权重控制仍需验证。年龄组的“零额外写入”仍保留当前源C的写入，不称完全无干扰。

## 服务器安装与只读编译

使用发布消息中的固定commit和install_stage1_compiler.sh。该安装脚本只更新三个小文件，先核对SHA256并做语法检查，再保留旧文件备份并替换；不覆盖训练公式、权重或数据。

```bash
conda activate bench
cd /home/bohan_zhang/research/benchmark_temporal
source configs/server_paths.sh
```

安装成功后：

```bash
python -u "$SNN_PROJECT_ROOT/scripts/compile_stage1.py" \
  --source-root "$SNN_CAMPAIGN" \
  --source-code "$SNN_CODE" \
  --out "$SNN_PROJECT_ROOT/results/stage1_all_layers"
```

SNN_CAMPAIGN仍指原materialized的campaign_v04。输出目录必须是新的；重跑应保留原目录并选择另一个新目录，不能删除已产生证据来消除报错。编译器核对原主卡集合变化、合同及源队列身份；超出预算保护上限会报错，不自动抽掉需求。

旧训练此时继续运行。查看：

```bash
cat "$SNN_PROJECT_ROOT/results/stage1_all_layers/runtime_budget.json"
cat "$SNN_PROJECT_ROOT/results/stage1_all_layers/implementation_audit.json"
```

## 产物含义

- campaign.json / manifest.jsonl / train.sqlite：可核验的训练计划；初始状态planned_not_activated。
- training_provenance.jsonl：旧配置/新增配置、已有完成候选/失败/运行中/待训练。
- analysis_jobs.jsonl：共同分析调用参数与检查点依赖，所有自动分析请求只使用development。
- capture_requests.jsonl：全部已保存检查点的完整归档请求；实际文件在后续扩展。
- original_all_layer_coverage.json：逐字保留旧账本。
- all_layer_coverage.json：合并显式协议标题后的当前账本。
- requirement_dependencies.json：每张主卡/每个专项与训练组、部分共同模块、原代码候选及未完成状态的对应。
- implementation_audit.json：仍需适配或验收的依赖。
- runtime_budget.json / STAGE1_REPORT.md：计数换算、复用候选及全层审阅入口。

analysis_jobs.jsonl是新的配方合同，不能直接当作旧post-worker的数据库或旧post-job输入；旧入口不会自动解释这里的模块覆盖和冻结规则门槛。当前编译器没有自动执行这些配方，也没有替专项协议关闭缺口。

## 训练切换支持与边界

更新后的efficient_campaign.py支持本计划的显式新增配置。activate仍先核对训练源码/manifest、备份旧数据库、暂停旧pending领取并自然排空，不杀进程。若另一个优化队列的监督器共享同一训练输出，会请求该队列停止领取新任务并等其结束；其他项目不受影响。新增配置没有旧行时保留pending，已有失败继续保留，不自动重试到成功。

训练和分析调度仍需分别安排。此手册只部署并执行只读编译；不在编译命令内自动激活队列、占用GPU或打开留出结果。

## 仍需闭环的实际问题

当前共同selective模块主要处理focal源，非目标历史操作和独立命中/误伤验证需要补充。原专项的已知答案校准、RRR/通信、实际二值通路、恢复再封堵、空间区域/频率等要在当前训练模型上适配并验收。

L6的same-weight前向等价检查独立于重训；L5.4需冻结诊断到修改的规则和成本对照后再打开heldout_design组的分析。该门槛目前是配方中的显式依赖，不是已经完成的冻结/验证实验。confirmation和audit没有自动访问。

SHD/HAR只支撑真实分类。真实重建数据和适配仍单独列为阻塞，不以G07/G08冒充。缺代码、数据或时间是未完成/阻塞，不能标为科学上的不适用。

## 时间

若492项都需训练，单按聚合吞吐1.5/3/4项每分钟换算为约5.47/2.73/2.05小时。1.5来自此前六worker混合任务观测，3和4是未测量场景；这些不是此阶段四卡的实测训练时间。

732项共同分析、全部检查点的完整归档和专项适配时间尚未测量，因此不承诺“整个第一阶段2–6小时结束”。服务器先输出真实复用数量，并分别标定短向量、长序列与空间作业的训练、分析和存储成本，才能形成截止时间安排。

English: The compiler emits explicit training identities and all-layer dependencies without touching running work. It retains 40 master cards and 92 reconciled subsection identifiers. Shared analysis recipes and available source files are not completion of specialized protocols. Training-only count scenarios are not a measured end-to-end runtime forecast.
