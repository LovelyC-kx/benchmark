# 近期基准、作者代码与本研究的衔接

核对日期：2026-10-29。本文区分“基准论文”“模型论文”“开源实验库”。下面的来源链接、固定代码版本及具体适配状态共同构成证据；不能把一个库收录的模型数量、一个摘要的任务数量和我们已经跑过的数量混为一谈。

本轮结论：真实数据必须进入主体实验；典型脉冲模型和强连续模型必须按领域补齐。但增加名单本身不构成本文的主要创新。我们的研究问题仍是：任务究竟需要什么历史，模型保留了什么，下游使用了什么，哪些残留内容产生干扰，以及改变哪一个载体能改变结果。

## 1. STP：Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects

- [2025预印本](https://arxiv.org/abs/2502.09449)、[2026-10-08 TNNLS记录](https://pubmed.ncbi.nlm.nih.gov/42848555/)、[作者代码](https://github.com/liyc5929/neuroseqbench)。固定commit：`af3c3204df0c3bedc9abfde3a01cc62f42effde5`。
- 已做到：STBP/SDBP/NoTD拆分学习中的时间路径与前向时间状态；不只比较最终准确率，也问任务是否真正依赖时间。作者README结果表覆盖PTB、PS-MNIST、Binary Adding。
- 版本差异：2025摘要与README写三项；2026期刊摘要写四项；当前代码的STP目录还包含SHD、SSC、TIMIT。没有取得最终期刊全文时，不能自行指定第四项，也不能把目录存在直接当作最终论文结果。
- 本文复用：保留三种梯度/状态模式及原生作者实验作为锚点。新增sMNIST/PS-MNIST原始IDX适配；固定置换，不把像素扫描称为自然动态时序。
- 本文推进：同一训练完成模型上的状态、输出消息、读出、选择性子空间和恢复实验。重新训练一个NoTD模型，不能单独定位已训练模型实际使用的历史载体。两种问题都需要，结果分别解释。
- 尚缺：PTB语言模型的分词、词表、序列损失/perplexity合同没有接入当前公共分类/回归头；继续使用作者原生轨道，不能以普通分类替代。

## 2. NSA：Neuromorphic Sequential Arena（IJCAI 2025）

[论文全文](https://arxiv.org/html/2505.22035v1)、[作者数据与预处理](https://github.com/liyc5929/neuroseqbench/blob/af3c3204df0c3bedc9abfde3a01cc62f42effde5/experiments/neuromorphic_sequential_arena/README.md)。与STP共享上述代码版本。

| 作者任务 | 数据来源 | 本文处理 |
|---|---|---|
| AL | 按动作生成的自主定位 | 合成应用任务，不能计入真实传感器数据 |
| HAR | WISDM2019手表陀螺仪 | 新增原始文本适配、subject划分、train归一化 |
| EEG-MI | OpenBMI | 新增保留subject的MAT适配；混合四NPY另留原生轨道 |
| SSL | SLoClas | 登记原始录音和作者MAT包；方向标签、麦克风布局、父录音须保留 |
| ALR | DVS-Lip | 登记完整事件唇读；原生加载/文本标签协议尚未迁入公共管线 |
| AD | Intel N-DNS | 原生语音增强、SI-SNR轨道；不能用图像MSE代替 |
| ASR | AISHELL-1 | 原生识别、CER轨道；不能用音频类别预测替代 |

NSA同时比较神经元和架构、速度、内存和操作级能耗，应用覆盖明显强于只跑SHD。我们需要复用其任务与作者模型，不能只引用其动机。

实际代码审计发现：`electroencephalogram_motor_imagery.py`在各subject内先归一化、合并后打乱trial再划分；因此该原生协议不能直接支持跨受试者泛化。本文新适配把subject分配提前到归一化之前，且只拟合真正的train角色。改变后必须使用新的协议名，不能直接拿分数宣称超越其原表。

WISDM作者原始加载器把文件拼接后进行滑窗；我们的窗口明确不能跨subject、活动、时间倒退或较长采集间断。归一化、分组改进是可复查的实验设置，不是凭空声称原作者结果无效。

## 3. Benchmarking spiking neural networks across sensing modalities on edge devices（2026预印本）

[预印本](https://arxiv.org/abs/2609.00026)、[Soul作者库](https://github.com/yudi-mars/Soul)，固定commit：`e3dc3237bfbe1876886f3beb77b6d53c6c016232`。本文核对了README、分模态数据说明、模型注册表、Spikformer及无线加载代码；本轮未完成其全部原生实验复现。

作者仓库当前About使用题名“A modality-aware benchmark reveals when spiking neural networks benefit edge sensing”；本目录保留用户给出的预印本题名，并记录此版本差异，不把两个标题自动算作两篇独立基准。

贡献是跨感知模态、架构/神经元组合和边缘设备约束的系统比较。真实数据包括惯性、声音、无线及事件/视觉来源；模型覆盖VGG、SEW/MS-ResNet、多个脉冲Transformer，以及运动领域的DCNN/DCL/SenseHAR。因此本文原来的八骨架确实不足以代表该覆盖。

具体适配规则：

1. Soul移植版、原作者模型、本文参考骨架分别登记。通过Soul调用Spikformer，仍应标明Soul版本。
2. UT-HAR加载代码读出`N,1,250,90`并作整体归一化，且把原val合并入test。本文不能悄悄沿用这些操作后声称独立验证。
3. Widar的BVP张量和原始CSI不是同一种表示；不能把其中任一轴不加核对地叫作物理时间。
4. 工具库部分文档存在类别/数量说明错配；数据长度、类别和轴以原数据及实际加载器验收，不照抄说明表。
5. 静态输入重复T次与连续采样T个新观测不是同一种能力。两者要配G05/G09的观测数量控制。

本研究希望进一步解释跨模态排名为什么改变：来自输入频谱/信息到达，还是状态保留、内容可访问性或读取方式。这是需要实验支持的目标；现在不能宣称已经超越其边缘设备实测。

## 4. The neurobench framework for benchmarking neuromorphic computing algorithms and systems（Nature Communications 2025）

[论文](https://www.nature.com/articles/s41467-025-56739-4)、[官方库](https://github.com/NeuroBench/neurobench)，固定commit：`e521c28f5798cb40bb9423758bf066dad53fe070`。

其价值在于共同的任务和资源评估合同，以及algorithm/system两条轨道；初始算法任务包括少样本增量关键词、事件检测、非人灵长类运动预测及混沌预测。它提醒我们真实连续输出与系统成本都不能被准确率表取代。

本文采用其“先定义测量对象，再报操作/内存/稀疏性”的原则。操作数估计、稠密GPU运行、芯片实测分别报告。现有`resources()`不是NeuroBench完整指标复现；非人灵长类数据的下载、按session划分和原生decoder尚未加入，因此不列入已可运行真实集合。

## 5. SpikingTSF（2026开源预测基准库）

[库及协议](https://github.com/spikora/SpikingTSF)，固定commit：`6154af0585bf615b4a47ce77c6869117b40178d0`。这是开源基准库；未据此虚构顶会接收身份。

它把SpikF、iSpikformer、SpikeRNN/SpikGRU/SpikTCN、多种脉冲Transformer及DLinear/iTransformer带入真实预测任务。ETTh1/2、ETTm1/2应立即成为本文真实数据入口，而不是只在合成回归上谈时序能力。

代码/说明也有版本差异：README模型总数和当前模型目录不完全对应；DATASETS.md说不分发数据，但固定树中包含部分ETT CSV。我们的版本依据是具体源文件和commit，不把仓库宣传数字当完成数。

本文新增`forecast-csv`：完整未来H步、多变量目标、train-only标准化、无跨角色/块窗口、MSE/MAE和持久值基线。它是本文严格的十角色研究协议，分数不等于原生LTSF排行榜。作者原生12/4/4月等协议通过独立入口保留。

## 6. SeqSNN：2024—2025预测、CPG与相对位置编码

[微软作者框架](https://github.com/microsoft/SeqSNN)，固定commit：`15cbacb123362c97c0aa0f6f861c18f4f77e3e8f`。它收录ICML2024时序预测、NeurIPS2024 CPG及NeurIPS2025相对位置编码相关实现。

要复用的不只是模型名字：需核对物理时间到脉冲仿真步的映射、patch/window编码、变量token、预测头和位置编码。Electricity、Solar、METR-LA、PEMS-BAY因此加入来源目录。CSV适配不等于已读取原生HDF5路网图；图结构、时间戳和变量选择另列。

本研究可比较“编码先压缩整段历史”和“递归状态在线压缩历史”的差异。整段已观测上下文内双向处理在预测终点可以合法，但不能直接拿来证明逐时刻在线读取能力。

## 7. 与基准扩展直接相关的重要模型论文

| 论文 | 已有内容 | 本文需要复用/检验 | 当前状态 |
|---|---|---|---|
| [TS-LIF，ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45bd303ea32a8193d3c19de5d5eaf138-Abstract-Conference.html) | 双区室、多尺度频率响应、稳定性 | 按作者实现区分滤波/保留/区室传输；不能用two_state_ref替代 | [作者代码](https://github.com/kkking-kk/TS-LIF)已固定；真实训练待执行 |
| [SpikF，ICML2025](https://proceedings.mlr.press/v267/wu25m.html) | patch编码、Fourier选择、无attention预测 | 频谱选择与历史载体是否产生不同失效；整段上下文权限单列 | [作者代码](https://github.com/WWJ-creator/SpikF)已固定、运行入口已接 |
| [SpikingSSMs，AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/34245) | 稀疏脉冲SSM、并行长序列学习 | 训练并行与推理状态分开，核对reset/scan和稀疏统计 | [作者代码](https://github.com/shenshuaijie/SDN)已固定、入口已接 |
| [Rhythm-SNN，Nature Communications2025](https://www.nature.com/articles/s41467-025-63771-x) | 节律门控，多个时序应用 | 跳过更新何时保留有用内容、何时错过新信息；变采样速度 | [作者代码](https://github.com/YinsongYan/Rhythm-SNN)已固定，尚无公共状态hook |
| [SpiLiFormer，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.html) | 侧抑制与脉冲attention | 抑制冗余/抑制有用历史分别检验；不能只凭稀疏率解释改进 | [作者代码](https://github.com/KirinZheng/SpiLiFormer)已固定；Soul移植版另标 |
| [SpikCommander，AAAI2026](https://ojs.aaai.org/index.php/AAAI/article/view/37194) | 多视图语音时间attention、SHD/SSC/GSC | 语音领域强模型；多视图新增观测/计算权限需对齐 | 原论文核对；未核实代码，不猜URL |
| [SpikySpace，2026预印本v2](https://arxiv.org/abs/2601.02411) | 脉冲selective scan、近似非线性 | 活动依赖更新/阈值策略与状态可用性，逐步因果性 | [匿名代码](https://anonymous.4open.science/r/SpikySpace)未固定，不标已复现 |
| [SpikeMamba，MICCAI2026](https://papers.miccai.org/miccai-2026/0990-Paper4698.html) | 脉冲SSM、状态依赖门控、5种ECG/EEG来源 | 活动门控与有用/干扰状态保留；跨患者协议必须独立 | [作者代码](https://github.com/Brain-Inspired-Computing-Lab/SpikeMamba)已固定、入口已接 |

SpikeMamba具体来源：PTB-XL、PTB ECG、ADFTD、MIT-BIH、CHB-MIT。作者CHB-MIT入口明确使用segment-mixed切分；不能据此推出未见患者泛化。PTB-XL的原生多标签诊断与作者单标签预处理也不能混称相同任务。

EVSNN/PA-EVSNN虽较早，却是事件重建必须有的领域对照。[作者仓库](https://github.com/LinZhu111/EVSNN)目前提供最小推理代码和预训练权重，不能标成已提供完整训练复现。U-TILISE是遥感序列重建参照，但原生全序列到全序列权限与本文因果当前帧重建不同。

## 8. 最新机制论文对“创新”的约束

[Frequency Matching in Spiking Neural Networks for mmWave Sensing，ICML2026](https://proceedings.mlr.press/v306/liao26e.html)已经研究LIF低通机制与毫米波判别频谱的匹配，并提出衰减配置原则。因此“LIF是低通、匹配频率更好”不能作为本文首次提出的结论。

[预印本全文及附录](https://arxiv.org/html/2605.09983v1)还明确给出判别频谱而非只看输入总功率、beta扫描、非线性适用边界和低通LeNet对照。其对照限定为逐帧空间低通，主动排除了跨帧时间滤波；因此本文需要把这两种滤波轴、物理帧数和SNN仿真步进一步分开，而不是笼统补一个“低通基线”就称为新发现。四个实际来源是AOPHand、mmFiT、Pantomime、MMActivity，且存在完整数据与作者子集的区别。

我们应继续追问：在既有判别频谱分析之后，历史内容的绑定、持续写入和下游使用还能解释哪些差异？滤掉噪声的收益与保留可读取内容能否分离？时间低通与空间低通是否给出不同结论？reset、适应和选择性读出分别贡献多少？总功率相同、相关频段不同本身只能检验既有频谱解释；若要提出超出该解释的证据，还需控制其已有判别谱指标并构造相位/跨频关系等反例。这些是G06、L2、L3和L4的连接点，新增相位协议尚未实现，不计入可运行任务。

在线性无reset近似下，可分析`h_t=beta*h_(t-1)+u_t`的频率响应；它不能直接证明带阈值/reset网络的完整任务表现。非均匀采样还需把beta与实际delta-t对应。理论先陈述适用模型，再用真实毫米波和受控频谱任务检验外推边界。

## 9. 2026新预印本：列入观察，不自动扩大本文任务主线

- [ANTShapes Benchmarking Datasets for Event-Based Neuromorphic Object Classification](https://arxiv.org/abs/2608.27150)：新增模拟事件形状任务，适合几何/运动控制。它属于模拟数据，不能用来增加真实采集数据数量。本轮核到arXiv元数据，尚未核到完整代码，暂不加入可运行集合。
- [Benchmarking Fairness in Spiking Neural Networks: Data Bias, Spurious Features, and Hardware Effects](https://arxiv.org/abs/2605.27407)：提示数据伪特征与部署约束的交互。本文借鉴“来源组和干扰维度必须显式登记”的设置思想；其匿名代码和硬件实现本轮未审计，不复述其数值结论或声称加入了硬件公平性基准。

## 10. 从文献到本文实验的组织

1. **原生锚点**：固定作者代码、数据预处理、任务指标与训练设置，先确认能得到合理原生结果。
2. **共同条件下比较**：相同上下文/观测/目标/拆分/调参预算；独立参考实现可替换单元和载体，官方模型不可无说明地被“缩成同名小网络”。
3. **受控任务定位**：同一底层场景分别改变年龄、持续写入、需求、频率、采样、读取，不用单一任务成绩代替能力。
4. **真实数据检验**：原生风险、训练轨迹、保持/读取证据是否对应真实失效。真实数据没有已知独立生成因子时，不伪造H1/H2标签。
5. **在固定模型上干预**：状态/消息/读出分开，配无操作、损伤、恢复、重放和独立确认，回答可行动的设计问题。

“超越已有benchmark”应最终体现为更有解释力、可复核、能指导模型改进的证据，而不是更长的数据集和模型列表。这一目标尚待正式实验验证。
