# 真实数据、典型骨架和领域模型扩展

2026-10-29，v0.3.0。上一版简表只列了受控任务与参考模型，遗漏了已经登记的真实来源。真实数据一直是研究必需部分；本轮把新增适配、近期基准和作者模型入口一起补齐。

本轮交付：32种公共参考骨架、12种可注入单元、9套作者代码运行入口；112项来源/任务/素材加G01—G09，共121个目录条目；24个直接下载文件，其中5个作者原生预处理包默认不下载。121不能写成121个真实动态数据集，32也不能写成32篇官方模型已复现。

## 1. 三类数据的作用

| 类别 | 代表 | 在论文中的作用 | 不能推出什么 |
|---|---|---|---|
| 受控生成任务 | 原16、tagged、G01—G09、Adding/Copy/MQAR | 知道生成因子，拆开年龄/写入/需求/频率/采样 | 不能仅凭合成结果声称实际应用优势 |
| 真实来源的派生序列 | sMNIST、PS-MNIST、静态重复、真实图像人工缺失 | 与旧基准衔接；控制信息到达与重复 | 像素扫描与仿真步不是物理动态时间 |
| 真实动态记录 | 音频、事件、惯性、EEG/ECG、无线、气象/负载、视频/遥感 | 检验机制解释与应用失效是否对应 | 原始数据真实不等于任意切分/归一化都支持泛化主张 |

## 2. 真实数据优先集合及其状态

“专用适配”指代码与人工格式fixture已验收；本轮没有下载并训练这些完整语料。来源及全部直接文件见 [总表](DATASETS_ALL.md) 和 [机器JSON](../snnbench/resources/datasets.json)。

| 来源/任务 | 获取链接 | 原生输入与目标 | 当前接口与下一项验收 |
|---|---|---|---|
| SHD / SSC | [作者发布](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) | 脉冲事件→数字/词类别 | `prepare spikes`；真实录音级分组，尚不声称speaker隔离 |
| Speech Commands v2 | [官方说明](https://www.tensorflow.org/datasets/catalog/speech_commands) | WAV→35词 | `prepare gsc`；speaker隔离、右边界因果FFT，特征不是Mel |
| DVS Gesture | [Tonic数据说明](https://tonic.readthedocs.io/en/latest/datasets.html) | 极性/位置/时间→手势 | `prepare gesture`；Tonic预分段NPY，user隔离，时间窗口另登记 |
| UCI HAR | [UCI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | 128步9通道惯性→6活动 | `prepare har`；用原始Inertial Signals，保留subject |
| WISDM2019 | [UCI](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) | watch/phone、gyro/accel→18活动 | 新`prepare wisdm`；subject先划分，窗口不跨活动/时间间断 |
| OpenBMI MI | [GigaDB](https://gigadb.org/dataset/100542) | EEG→左右运动想象 | 新`prepare openbmi`；读取作者保留subject的MAT，采样率必须显式给定 |
| ESC-50 | [作者库](https://github.com/karolpiczak/ESC-50) | 5s WAV→50环境声 | 新`prepare esc50`；官方外层fold、原Freesound recording分组，五折分别执行 |
| ETTh1 / ETTh2 | [原作者数据](https://github.com/zhouhaoyi/ETDataset) | 小时7变量→未来H步 | 新`prepare forecast-csv`；完整多步预测，CSV时间戳检查 |
| ETTm1 / ETTm2 | [原作者数据](https://github.com/zhouhaoyi/ETDataset) | 15分钟7变量→未来H步 | 同上；不能根据部分README文字误称逐分钟采样 |
| Electricity / Weather / Traffic | [TSL数据发布](https://huggingface.co/datasets/thuml/Time-Series-Library) | 多变量过去→未来 | 带真实date列的CSV可用新接口；仍需实际release/缺失审计 |
| Solar / Exchange / ILI | [TSL](https://huggingface.co/datasets/thuml/Time-Series-Library)、[原TXT来源](https://github.com/laiguokun/multivariate-time-series-data) | 能源/汇率/周监测→未来 | 有真实时间戳才可转CSV；不能替无日期TXT编造采样时刻 |
| METR-LA / PEMS-BAY | [DCRNN原作者](https://github.com/liyaguang/DCRNN) | 路网速度→未来 | HDF5/邻接图待原作者加载后导出；不能当成已接通图模型 |
| DVS-Lip / SLoClas / N-DNS / AISHELL-1 | [NSA下载指南](https://github.com/liyc5929/neuroseqbench/blob/af3c3204df0c3bedc9abfde3a01cc62f42effde5/experiments/neuromorphic_sequential_arena/README.md) | 唇读/定位/语音增强/ASR | 作者原生轨道；公共帧分类器不能替代SI-SNR、CER及原生标签合同 |
| UT-HAR / NTU-Fi / Widar | [Soul无线说明](https://github.com/yudi-mars/Soul/blob/e3dc3237bfbe1876886f3beb77b6d53c6c016232/doc/source/datasets/wireless_en.rst) | CSI或BVP→动作/身份 | 作者原生轨道＋显式tensor manifest；需原始subject/session/room分组 |
| mmWave家族 | [Soul](https://github.com/yudi-mars/Soul)、[mmFiT原始点云](https://data.mendeley.com/datasets/d3dt5tb74h/1)、[ICML2026](https://proceedings.mlr.press/v306/liao26e.html) | 点云/热图/轨迹→活动 | mmFiT原始点云只含2位参与者；D63历史热图另版，不能混入subject泛化；MM-Fit/MM-Fi也不是同一数据 |
| PTB-XL / PTB ECG | [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)、[PTB](https://physionet.org/content/ptbdb/1.0.0/) | ECG→诊断 | 新增来源与SpikeMamba轨道；公共原始WFDB适配、患者协议/临床指标待做 |
| ADFTD / MIT-BIH / CHB-MIT | [ADFTD](https://openneuro.org/datasets/ds004504)、[MIT-BIH](https://physionet.org/content/mitdb/1.0.0/)、[CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) | EEG/ECG→诊断、心律失常、发作预测 | 同上；按患者/发作事件切分，预测与检测分开 |
| EarthNet / U-TILISE | [作者数据准备](https://github.com/prs-eth/U-TILISE) | 多时相反射率→当前干净帧 | 已有`prepare earthnet`；真实数据上的人工矩形缺失，不是原生真实云评估 |
| SEN12MS-CR-TS / AllClear | [SEN12MS](https://github.com/PatrickTUM/SEN12MS-CR-TS)、[AllClear](https://github.com/Zhou-Hangyu/allclear) | SAR/光学时序→重建 | 原始GeoTIFF、配准/云mask/时刻/地理隔离待作者转换与审计 |
| Vimeo-90K / REDS / MVSEC / DSEC / Diving48 | [Vimeo](https://github.com/anchen1011/toflow)、[REDS](https://seungjunnah.github.io/Datasets/reds.html)、[MVSEC](https://daniilidis-group.github.io/mvsec/)、[DSEC](https://dsec.ifi.uzh.ch/) | 视频重建/事件重建/动作 | 显式tensor manifest；保留父视频、标定、目标时刻与未来帧权限 |

已经可直接转入公共链的专用来源是SHD、SSC、GSC35、Gesture、UCI HAR、WISDM、OpenBMI的subject-MAT、ESC-50、ETT四版本、MNIST三种派生协议及EarthNet作者转换HDF5。每种“可转”都受到上表格式与协议限制。其他来源是有链接、有用途、有缺口的待适配项目。

## 3. 新增原始适配的具体合同

### 3.1 sMNIST / PS-MNIST / 静态重复

读取原始IDX及gzip，检查magic、rank、载荷长度与28×28形状。扫描为`[784,1]`；置换使用一个固定784维排列，写入数据合同；重复为`[T,1,28,28]`。均按原始image分组并保留官方train/test来源。没有把按图像分组夸大成按书写者分组。

### 3.2 ESC-50

读取作者CSV与44.1kHz单声道PCM16音频。外层五折中的一折仅进入confirmation/audit；其他fold中的同一Freesound父录音不跨研究角色。特征窗只用当前/过去样本，时间戳是窗的右边界；64个线性频带log-power，明确不是Mel。需要原生五折分数时，另跑作者/领域原生完整协议。

### 3.3 WISDM

分别选择watch/phone和gyro/accel，读取真实timestamp。默认200点窗、100点stride；不跨subject、活动变更、时间倒退或>1秒采集间断。窗口重叠只发生在同一个subject角色内。按训练角色窗口计算通道均值/尺度，记录参与subject；不会用测试subject估计归一化。

### 3.4 OpenBMI

要求先保留原始subject。作者导出的`data_N.mat`采用`N,L,C`，对应`label_N.mat`的1/2标签；不能把已混合的`x_train.npy`反推成subject。非重叠块平均作降采样，观测在块右边界可用；输入采样率由命令参数明确给出。原始`*_MI.mat`到这些MAT的作者预处理仍须执行，MATLAB v7.3等格式需按实际release转换，不宣称通吃所有MAT格式。

### 3.5 真实多步预测

每个输入`[L,F]`，原生目标`[H,F]`；诊断探针可以有预算，训练损失始终覆盖全部H×F。模型头已经支持二维连续目标。

默认十个角色按时间顺序分别占60/5/3/8/3/6/3/4/4/4%。每个角色内部划为不重叠时间块，完整输入和未来目标必须留在同一块；跨角色没有共享原始时间索引，可额外设置embargo。边界不足以容纳L+H直接报错。

长预测可显式用`--role-allocation minimum_first`：先给每个角色保留至少两个窗口及embargo，再按上述权重分配剩余行。实际角色比例与归一化范围全部写入合同；不是暗中修改原生LTSF切分。若总长度连最低要求也不够仍报错。ETTh的H720不能强套默认3%小角色，可使用这一派生协议或留在作者原生轨道。

标准化仅拟合train原始行，写出均值/尺度和行范围。报告标准化MSE/MAE、原单位MSE/MAE、last-value persistence对照。原单位混合异量纲变量的均值只是描述，不当作无量纲排名。未规则采样不被插值成规则采样；原始缺失/非有限值直接拒绝，修复规则需另行版本化。

时间块只是统计分组，不保证过程独立；需报告独立块数与块长敏感性。单块不生成虚假置信区间。短ILI等任务可能不适合十角色长上下文，不能强行复制数据凑角色。

## 4. 公共模型工厂已经增加的骨架

全部32种都有明确接口；完整描述见 [模型来源JSON](../snnbench/resources/model_sources.json)。这些是本文独立实现。`_ref`以外的旧名字也不自动获得官方复现身份。

| 家族 | 原有 | 本轮新增 | 要回答的问题 |
|---|---|---|---|
| 向量脉冲 | MLP、recurrent MLP、3-tap temporal conv、CNN1D | dilated spiking TCN、causal temporal spike-attention reference | 单元记忆、显式延迟、反馈和KV累积能否互相替代 |
| 空间脉冲 | CNN2D、SEW-ResNet18、逐帧spike attention、U-Net | SEW-ResNet34、MS-ResNet18 reference、VGG11-stage reference、ConvLIF-RNN | 残差/深度/空间结构与时间状态如何交互 |
| 经典连续递归 | GRU、LSTM、实数对角SSM | Elman tanh RNN | 门控、线性状态与脉冲单元的公平参照 |
| 时间卷积/attention | 无位置单层causal attention旧控制 | dilated TCN、多层位置编码Transformer、Conformer-style reference | 历史保存在固定缓存、可增长KV还是局部卷积中 |
| 结构化/选择性SSM | 简单对角SSM | S4D-style、LRU-style、Mamba-1-style references | 多时间尺度、复数旋转、输入选择性如何影响保留/读取 |
| 空间连续/视频 | ConvGRU | CNN-LSTM、causal Conv3D、ANN U-Net、ResNet18/ConvNeXt/ViT references | 真正跨帧状态、短视频缓存和逐帧强空间表示分别贡献多少 |

S4D参考使用复数稳定对角递归与ZOH离散输入；LRU参考使用稳定复数极点与幅值归一化；Mamba参考包括输入依赖delta/B/C及因果depthwise卷积。没有使用官方融合核，不测得异步稀疏加速，也不是官方Mamba-2/3。Conformer参考不是完整ASR Conformer；causal Conv3D不是I3D/SlowFast；逐帧ViT不是Video Swin。

12单元可注入14骨架，形成168个组合，其中包括continuous/tanh等非脉冲机制控制。新增18种连续骨架总数中包含原有5种。重建只允许6种有原生空间输出的骨架：CNN2D、SNN U-Net、ConvLIF-RNN、ConvGRU、causal Conv3D、ANN U-Net。ResNet/ViT分类头不能被静默当成像素重建头。

## 5. 状态与时间权限必须准确

- SNN单元：膜电位、适应/突触状态等按实际实现记录；不是只看最终发放率。
- TCN/Conv3D：真实延迟缓存属于记忆；状态干预不能只清神经元。
- Transformer：完整KV是未来会使用的状态，随上下文增长；默认固定坐标探针只观测输出，不据此宣称完整KV没有信息。绝对位置由公共stream cursor派生；NoTD清记忆时不会顺便把位置重置到0。
- S4D/LRU/Mamba：复数实/虚分量、选择性scan状态及卷积缓存分别保留。坐标预算与完整状态大小分别报告。
- 逐帧ANN ResNet/ConvNeXt/ViT/U-Net：骨架本身没有跨帧递归；mean/sum/trajectory读出仍可保存时间历史，因此“无骨架递归”不等于“整个系统无记忆”。

完整状态snapshot/replay与probe所能读到的表征不是同一集合。跨模型比较还需报告实际状态大小、上下文长度、位置/时间戳、累计观测数、参数量与调参预算；相同width不保证相同参数或计算成本。

## 6. 官方模型和领域模型如何增加

近期文献逐篇判断见 [2025—2026复核](BENCHMARK_REVIEW_2025_2026.md)。固定21个上游仓库版本；登记24组作者模型来源，另列视频识别/恢复领域候选。9个suite已有代码运行与回执入口，不表示9套完整实验已经复现。

| 应用/机制 | 作者模型优先集合 | 状态 |
|---|---|---|
| 长序列脉冲机制 | GSN、GSU、BinaryS4D、DCLS-Delays、SpikingTCN、SpikingSSMs | NeuroSeq/SDN原生入口；公共状态适配仍待逐模型做 |
| 重要单元 | PLIF/ALIF/GLIF/CLIF/TC-LIF/LTC/LM-H/DH-LIF/CE-LIF/SPSN/PMSN、TS-LIF | 原作者代码单列；不拿简化参考类冒充 |
| 脉冲视觉 | VGG、SEW/MS-ResNet、Spikformer、Spikingformer、Meta-SpikeFormer、QKFormer、SpikingResformer、SpiLiFormer | Soul移植轨道＋原作者固定来源；两种实现不能混名 |
| 时序预测 | SpikF、iSpikformer、TS-GRU/TCN、CPG、TS-LIF、SpikySpace | 前五类有固定来源；SpikySpace匿名代码未固定 |
| 强预测ANN | DLinear、PatchTST、iTransformer、TimesNet、TimeMixer、TimeXer、S4/S4D、Mamba、MambaSL | TSL原生入口及SSM固定来源；不是只用GRU作参照 |
| 语音 | 原生Spiking-FullSubNet、SpikCommander、相应领域ANN | 增强与ASR沿NSA专用轨道；SpikCommander未核到代码 |
| 运动/无线 | DCNN、DCL、SenseHAR及Soul跨模态骨架 | 保留物理时间和表示差异；不能复制图像设置就称公平 |
| ECG/EEG | SpikeMamba、领域ANN/Medformer来源 | SpikeMamba原生入口已接；患者协议/标签与指标需单独验收 |
| 事件重建 | EVSNN、PA-EVSNN | 固定作者最小推理代码；完整训练代码未由此获得 |
| 视频/遥感重建 | ConvLSTM、EDVR、BasicVSR++、RVRT、U-TILISE | 领域候选/固定来源；未来帧、配准与预训练权限逐项记录 |

作者原生模型先建立合理结果锚点，再接公共观测/状态hook。无法实现的干预应写“不适用/待适配”，不能用清一个无关tensor来假装完成L4归因。

## 7. 可直接使用的服务器命令

```bash
python -m pip install -e '.[data]'
python -m snnbench models
python -m snnbench models --sources
bash scripts/download_real_priority.sh /data/snn/raw
```

WISDM的UCI外层ZIP可能包含内层`wisdm-dataset.zip`，需再解压；OpenBMI保留subject的MAT与EarthNet作者转换HDF5按原作者流程获取。不要把NSA混合四NPY误交给subject适配。

```bash
python -m snnbench prepare mnist --root /data/snn/raw/D89/extracted --mode permuted --permutation-seed 20261029 --out /data/snn/canonical/psmnist
python -m snnbench prepare esc50 --root /data/snn/raw/D41/extracted --test-fold 5 --out /data/snn/canonical/esc50_f5
python -m snnbench prepare wisdm --root /data/snn/raw/D47/extracted --device watch --sensor gyro --out /data/snn/canonical/wisdm_watch_gyro
python -m snnbench prepare openbmi --root /data/openbmi/eeg_data --sampling-hz 1000 --downsample 10 --out /data/snn/canonical/openbmi_subject
python -m snnbench prepare forecast-csv --csv /data/snn/raw/D96/ETTh1.csv --dataset-id D96 --context 96 --horizon 24 --stride 24 --embargo 96 --out /data/snn/canonical/etth1_L96_H24
python -m snnbench verify-data --root /data/snn/canonical/etth1_L96_H24
```

OpenBMI示例的1000Hz仅适用于输入确为该采样率的导出；已重采样导出必须填写其实际采样率。每份canonical都运行verify-data。预测H96/192/336/720需分别导出；长预测可显式设置minimum_first，不足不能偷偷跨边界凑窗口。公共编码器也需纳入公平性审查：auto的向量MLP压缩与none/作者原生编码对照，不能把输入投影损失误认成单元记忆不足。

```bash
python -m snnbench compile --profile expanded --seeds 11 22 33 44 55 --archives /data/snn/canonical/etth1_L96_H24 --out manifests/expanded.jsonl
python -m snnbench preflight --manifest manifests/expanded.jsonl
python -m snnbench queue-init --db expanded.sqlite --manifest manifests/expanded.jsonl
python -m snnbench worker --db expanded.sqlite --device cuda:0 --analyze
```

没有真实archive时，expanded是62种任务设置、每seed 4954份配置；5seed共24770份。它覆盖32骨架，但不执行不兼容的输入/头。新增profile统一使用output读出和gain=0.5；旧screen/full等配置保持旧比较口径，内部pre/post访问留给明确分析。gain只是起点，正式比较仍需相同的调参预算和工作点审查。

架构名单与任务设计也可分开：`compile --profile full --model-roster expanded ...`会把新增骨架放入full的年龄/负荷/写入/频率/采样等全部受控轴；paired/learning同样接受该选项。paired/learning中特殊机制网格仍只施加到明确的MLP/CNN2D锚点，不暗示每个新模型已经具有所有专用干预。默认不指定roster时，旧profile继续保留旧名单。

这不是建议不加筛选地一次性启动24770份：先GPU短验收、真实数据合同和代表模型收敛，随后扩大；原16/G01—G09是生成任务，不需要额外下载。单机队列、多GPU/跨节点Slurm规则继续按 [服务器说明](SERVER_RUNBOOK.md)。

## 8. 作者原生运行入口

```bash
python -m snnbench upstream-checkout --suite spikingtsf --out /data/snn/upstream/spikingtsf
# 在独立环境按固定作者源码安装依赖，准备其原生数据和配置。
python -m snnbench upstream-run --suite spikingtsf --source /data/snn/upstream/spikingtsf --config native_plan.json --out /data/snn/native_receipts/tsf_001
```

`native_plan.json`的argv是Python脚本后的参数，不是shell字符串。可参考 [TSL原生命令示例](../examples/tsl_native_plan.json)；所有路径改成实际服务器路径。入口校验commit与tracked源码，固定输入配置/清单hash，记录实际命令、包版本、stdout/stderr与退出状态。原始语料仍应另有完整校验清单，单个配置hash不能证明全语料未变。

作者代码依赖、GPU核与完整训练尚未在这里跑通；接口不会把一个成功退出自动标成“复现论文”。原生结果和公共协议结果分别入表。

另外核到TSL固定版本的`run.py`在解析参数前硬编码全局seed=2021，同时另有默认值不同的`--seed`。因此示例只登记原生固定seed=2021锚点；仅修改model_id或传`--seed`不能证明全局初始化已改变。正式多seed需建立明确的新源码patch版本、记录补丁及结果文件hash，再注册为派生实现；不能绕过原生入口的tracked源码检查。

## 9. 正式实验应先补的比较，及其论文用途

1. **真实时间是否必要**：完整上下文、仅末步、截断前缀、观测数量匹配和强静态聚合；保留原始标签只是预测同一来源目标的缺失观测实验，不自动解释为语义不变的世界反事实。
2. **保留与利用能否分离**：固定checkpoint，在相同可用内容上比较原生读出、线性/非线性probe与有限预算读出；probe可解码不等于原下游正在使用。
3. **保留是否有害**：在有明确生成因子的任务上固定有用内容并改变干扰写入；真实数据的“无关历史”不能凭时间早晚定义，需标注/实验依据。
4. **何种架构机制解释现象**：单元、显式缓存、KV、scan状态、反馈、读出逐项干预并恢复。观察到排名改变后，先排除编码、位置、观测数量和工作点差异。
5. **给后来者什么建议**：用验证集得到的能力/机制指标预测未见任务条件下的失效或改进；独立confirmation/audit检验，避免在同一批结果上找故事又证实故事。

数据与模型扩展提供检验范围，主要研究贡献应来自这些判断和可证伪证据。尚未完成的跨领域适配和官方复现单列，不用目录数量替代实验结果。
