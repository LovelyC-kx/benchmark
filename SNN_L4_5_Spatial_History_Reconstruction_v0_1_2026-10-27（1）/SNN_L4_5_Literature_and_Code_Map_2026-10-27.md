# L4.5文献与代码调用：已有工作、适配与待验证增量

2026-10-27，v0.1。本稿先覆盖Master原四条调用，随后接入真实重建、空间表征及2025/2026直接近邻。区分原文阅读、固定源码、选定定义执行、完整模型复现；本轮没有作者全模型或数据集复现。

## 1. 原计划覆盖与论文责任

| 原调用 | 已有工作确实提供什么 | 进入本节哪里 |
|---|---|---|
| BEN-DNS | 连续输出、因果实时处理及质量/延迟/能耗/资源评价 | 权限、读出和资源口径；不是图像模型 |
| BEN-SOUL | 多模态/编码/Cell/骨架、频谱/表征及共同解码器重建分析 | 保留已有重建与机制分析，扩展来源/区域/损伤恢复 |
| MEM-DMP | 显式低维慢状态、快速spike及算法硬件协同 | 状态所有者和总成本；不把aux或双通路本身称新 |
| GEN-MUSPIKE | 五类SNN音乐生成及多维客观/主观评价 | 不把生成覆盖当首次贡献；不同目标指标分开 |

原早期图像计划首选EarthNet2021/U-TILISE，再接SEN12MS-CR-TS/AllClear；当前Master另有MVSEC/DSEC。两条分支均保留，不用新检索结果替换旧计划。REPA/REG/RAE的整个L4数学线也保留。

## 2. BEN-SOUL：A modality-aware benchmark reveals when spiking neural networks benefit edge sensing

**版本与阅读。** 本轮重新读取用户上传24页主稿的无线频谱、FMS、表征重建方法、图3说明及代码可用性。上传文件Library身份为`libfile_f3056fc0bbb081918b688a3c270085fa`。它与公开[Benchmarking spiking neural networks across sensing modalities on edge devices](https://arxiv.org/abs/2609.00026)题名不同，不假设实验版本完全相同。

**他们做到了什么。** 已有多模态系统比较，并通过时域频谱、判别频谱、特征可分性及共同decoder骨架的代表性重建解释无线优势。不能再概述成只有排行榜或没有重建/机制分析。

**代码。** 原稿给出[官方仓库](https://github.com/yudi-mars/Soul-NeuSim)。API路由返回迁移信息，随后通过公共Git读取固定HEAD `e3dc3237bfbe1876886f3beb77b6d53c6c016232`，固定README与`soul/neuron/LIF.py`。未运行全框架、边缘设备或图3l重建实验；不把未取到的具体实验脚本说成作者没有实现。

**我们的复用与增量。** 继承多骨架/单元、表征与重建、资源分开评价。增加已知目标来源、C/H/U_vis覆盖、过期U_src、目标重算、独立内容验证、内部四组及区域/频率误差恒等式。本文希望把总体谱/表征解释推进到可干预、可预测的时序来源使用；跨模型及设计收益仍待正式实验。

## 3. BEN-DNS：The Intel Neuromorphic DNS Challenge

**来源。** Timcheck等，[原文v3](https://arxiv.org/html/2303.09503v3)，2023；[正式DOI](https://doi.org/10.1088/2634-4386/ace737)。本轮读任务、评价和基线部分。

**已有内容。** 面向实时语音去噪，联合质量、算法延迟、能量与芯片资源；区分算法和Loihi2硬件路线。连续信号恢复和部署评价已有成熟先例。

**代码状态。** 原文官方`IntelLabs/IntelNeuromorphicDNSChallenge`访问404；搜索得到公开fork，但本轮未把fork改称官方或执行其训练。代码可访问性缺口保留。

**怎样融入。** 本节把因果当前目标与离线双向重建分开，把状态/摘要/解码成本计费；没有把图像MSE改名语音质量，也没有把SNN消息统计直接当硬件能耗。N-DNS保留为后续跨模态连续任务。

## 4. MEM-DMP：Algorithm–hardware co-design of neuromorphic networks with dual memory pathways

**来源与身份。** Sun等，[Nature Machine Intelligence正式文](https://www.nature.com/articles/s42256-026-01255-3)，2026-06-16；[原预印本](https://arxiv.org/abs/2512.07602)。正式出版身份已核，不沿用未确认状态。

**已有内容。** 快速spike之外加入共享低维慢状态，缓慢线性动态反馈为输入电流，并联合研究状态大小、更新间隔、信用分配及硬件数据流。

**代码阅读。** [官方仓库](https://github.com/sunpengfei1122/Dual_memory_pathways)，固定`05cc0a3306b311e37ef9bc439ba7ac5186242672`；选读`psmnist/src/dual_pathway.py`。该路径以FFT计算LMU卷积，并保留memory gate和记录位置。批量卷积实现的缓存/执行方式不能直接当成已测在线常量内存成本。

**本研究适配。** 逐一计膜、aux、缓存与读出所有者，检验联合恢复和mean历史缺口。ALIF的每通道a不等于DMP共享m，当前六核心没有冒充作者DMP模型。后续应将作者模型作为有独立状态账的机制基线，比较诊断指定的慢量/空间量分配；双通路这个概念本身不作为创新。

## 5. GEN-MUSPIKE：MuSpike: A Benchmark and Evaluation Framework for Symbolic Music Generation with Spiking Neural Networks

**来源。** Liang、Tang、Zeng，[2025预印本原文](https://arxiv.org/html/2508.19251v1)。本轮读方法、架构、评价与客观/主观关系。

**已有内容。** 比较SNN-CNN/RNN/LSTM/GAN/Transformer的符号音乐生成，结合多类统计指标与听感研究。生成能力及人类感知评价均不是本项目尚无人覆盖的领域。

**代码。** [官方仓库](https://github.com/lqnankai/MuSpike)，固定`520fa0de26052d064d6450182c2744ac050c2f22`，固定mgeval/utils.py；原Euclidean距离定义在人工向量执行。没有MIDI数据、生成模型或听感实验。

**本研究适配。** 目标对应的像素/区域/频率误差与辅助分类、感知量分别记录；不把音乐分布距离当像素恢复，也不把漂亮图像当过去内容已找回。它为完整评价提供先例，区域来源与机制恢复由本研究自己的实验兑现。

## 6. U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series

**来源与身份。** Stucker、Sainte Fare Garnot、Schindler，TGRS2023；[原文记录](https://arxiv.org/abs/2305.13277)，[正式DOI](https://doi.org/10.1109/TGRS.2023.3333391)，[官方仓库](https://github.com/prs-eth/U-TILISE)。本轮进一步读官方数据/预处理、插值与模型/指标选定实现，不宣称重跑作者全部实证。

**他们做到了什么。** 空间encoder、时间attention和空间decoder构成seq-to-seq重建。EarthNet2021四波段与云缺失模拟具有直接复用价值；删去实际缺失帧后，有效时间间隔可不再均匀。

**代码固定与执行。** commit `1ea50fe1e4f1840a14b3f23062334077fc5f943b`；4文件：ImageSeriesInterpolator、UTILISE、metrics、EarthNet2021Dataset。执行原SAM与last-visible函数：SAM与本实现差约1.12e-16；零范数返回None；H最后见于2、C见于5、未见区为NaN的索引均核验。

**重要适配。** 原cloud mask是1表示遮挡，本参考输入是1表示可见；明确反转。原seq-to-seq不自动满足只给过去/当前的因果权限；pad mask、云mask、attention权重不是同一个对象。我们的adapter检查目标时点、真实可用时间、band与ROI，并在缺少内容兼容证据时不自动把past-seen叫H。

**复现边界。** 14GB官方演示是iid测试，不能当训练/调参数据。未下载它或作者checkpoint；本轮四波段toy不是EarthNet实验，也没有实现完整真实地理配准与邻近ROI泄漏审计。

## 7. AllClear: A Comprehensive Dataset and Benchmark for Cloud Removal in Satellite Imagery

**来源与身份。** Zhou等，[NeurIPS2024 Datasets and Benchmarks正式页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60095e1d7ebb292dbba93c4d8f7b2463-Abstract-Datasets_and_Benchmarks_Track.html)，[作者项目](https://allclear.cs.cornell.edu/)。项目页仍保留旧“submitted”字样，出版身份以正式会议记录为准。

**已有内容。** 全球多ROI、多时相、多传感云去除，包含时间长度和模态等分析，为真实分布与规模评价提供直接数据基础。

**代码。** [官方仓库](https://github.com/Zhou-Hangyu/allclear)，固定`3db4f86c1c58f106462f3a57ef0bc74da7ec455b`；选读dataset与utils两个文件。未下载数据、运行其基线或把其跨时相参考视为无误差同刻真值。

**我们的使用。** 在合成/受控人工缺失后做地域和观测外部检验；先保留S2分支，再单列SAR。关注过期历史、真实变化与参考时间差，采用ROI/source组划分。不是通过再增加一个数据集产生机制新意。

## 8. RAE：Diffusion Transformers with Representation Autoencoders

**来源与身份。** Zheng、Ma、Tong、Xie，[原文](https://arxiv.org/html/2510.11690v1)，[作者项目](https://rae-dit.github.io/)，[ICLR2026正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3c4141c12660ad3625eb4ae845e0a6f9-Abstract-Conference.html)。仅登记核实的主会身份，不推测oral/spotlight。

**已有内容。** 冻结的预训练表征编码器配训练解码器可以高保真重建；分类式表征目标不自动意味着低层细节一定不可恢复。它是“先检查解码器再判信息丢失”的关键近邻。

**代码。** [官方仓库](https://github.com/bytetriper/RAE)，固定`a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`，读取rae.py和decoder.py。原unpatchify方法精确恢复人工patch网格，错误patch数明确拒绝；没有运行预训练encoder、ViT decoder、扩散或作者图像实验。

**怎样适配。** 同一冻结空间核心比较1×1、3×3及GAP全图解码；独立内容probe与像素误差分开，追加同解码参数摘要比较。RAE的预训练权限和成本单列；不将其结论直接替代未训练SNN。

## 9. iREPA：What matters for Representation Alignment: Global Information or Spatial Structure?

**来源与身份。** Singh等，[2025-12预印本](https://arxiv.org/html/2512.10794v1)，[ICLR2026正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html)，[官方代码](https://github.com/End2End-Diffusion/iREPA)。主会身份已核，不赋予未核的报告形式。

**已有内容。** 比较全局任务表现与空间自相似对生成的解释力，并通过卷积projection和空间归一化改善对齐训练。这不是一般证明“全局语义无用”。

**代码固定与执行。** commit `99ad4ac234efe8de52ce157120f72856e836d09f`；spnorm、projectors、spatial_metrics三文件。原LDS模块与手工4邻域定义误差约3.73e-9；原spatial_zscore的平移不变性、常量与单token边界均检查。

**适配与推导。** 自相似只作描述，配合内容/像素风险和可逆置换控制。本研究的宽邻域描述量与作者4邻域代码分别登记。进一步提出绝对强度任务的条件分析：若唯一代码去掉均值，则偏移不同的样本不可区分；保存均值/尺度可逆但要计状态。这是任务适配，不是反驳其辅助对齐损失。

## 10. EVSNN：Event-Based Video Reconstruction via Potential-Assisted Spiking Neural Network

**来源与身份。** Zhu等，[CVPR2022正式原文入口](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Event-Based_Video_Reconstruction_via_Potential-Assisted_Spiking_Neural_Network_CVPR_2022_paper.html)，[作者项目](https://sites.google.com/view/evsnn)，[官方代码](https://github.com/LinZhu111/EVSNN)。

**已有内容。** LIF与膜电位单元实现事件重建，PA-EVSNN进一步利用自适应膜电位；已经有直接的SNN时序记忆与重建工作。

**代码。** 固定`9404c0a8dfc4a692ce7332774f36575b3213cb7c`，神经元与网络两文件。原MP charge及last_mem交接方法在人工状态fixture执行。不是完整SpikingJelly类或官方PA-EVSNN模型复现。

**重要区别。** 其MpNode.forward返回电位，不自动触发fire/reset；本参考continuous与post-LIF也不能随意与之等同。后续适配须对齐event voxel窗、内部步数、状态reset和模拟/物理时间。本文增加来源选择、区域任务效应、对应恢复及读者预算，而不是首次让SNN重建视频。作者效率数字没有转成我们的硬件成绩。

## 11. SSMT：Spike Stream Memory Transfer for Dynamic Scene Reconstruction

**来源与身份。** Dong等，[AAAI2026正式9页PDF](https://ojs.aaai.org/index.php/AAAI/article/view/37364/41326)，[作者主页](https://csycdong.github.io/)及官方oral日程核身份。

**已有内容。** 将先前估计的强度和运动跨片段传递，配合运动及间隔引导对齐，改善spike camera长程重建。读取方法/框架及初始/运动网络冻结的训练步骤。不能把它简化成仅用短窗。

**代码状态。** 原文给出`https://github.com/csycdong/SSMT`，本轮官方commits入口404；未取得可固定源码或执行作者网络。不是永远无开源的判断。

**怎样融入。** 过去强度与运动作为不同来源/所有者检查，控制错来源、错时间和对齐。当前周期整数逆变换只是合法坐标验收，不等于实现其估计运动或间隔模块。输入是spike camera并不自动说明整个网络是SNN，传感器与模型分别登记。

## 12. EvDiff：Event-Based Video Reconstruction using One-Step Diffusion Models

**来源与身份。** Li等，[2026-08 v2原文](https://arxiv.org/html/2511.17492v2)，[官方代码](https://github.com/LiWeitu/EvDiff)及ECCV2026出版记录。v1较早题名与v2/正式题名分别保留，不混作两个模型。

**已有内容。** EvEncoder结合一步扩散和预训练图像先验，使用代理训练扩大可用图像数据，关注时间一致性、像素保真与视觉质量；绝对亮度的观测歧义是其问题动机。

**代码。** 固定`3a510d18549704313b2d626dde495e04244cc98b`，选读DSEC RGB对齐转换及voxel_model。未运行SD3、事件encoder、生成模型或DSEC数据。

**我们的适配。** 区分历史测量提供的内容与预训练先验生成的合理细节；明确外部预训练、推断权限、目标对齐和不同指标。DSEC的event/RGB原生坐标不能默认相同，转换后的空白区域须排除。对观测本来不唯一的内容，不用“图像好看”推导已恢复真实历史。

## 13. 继承REPA、REG与基础E2VID的角色

REPA《Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think!》与REG《Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think!》的正式身份、原作者源码及先前选定执行已在L4.2登记。本节继续用它们组织“同一冻结表示、学习信号改变、语义/细节协作”三种比较，不将旧执行重新计入19新文件，也没有在未训练核心上宣称重跑对齐训练。

E2VID《Events-To-Video: Bringing Modern Computer Vision to Event Cameras》（CVPR2019）提供事件重建的递归CNN基础，[正式原文](https://openaccess.thecvf.com/content_CVPR_2019/html/Rebecq_Events-To-Video_Bringing_Modern_Computer_Vision_to_Event_Cameras_CVPR_2019_paper.html)。本轮只作为基础来源定位，具体作者模型/数据执行未做。

## 14. 每小节怎样调用

| 小节 | 方法近邻 | 本研究的具体适配 | 还缺什么 |
|---|---|---|---|
| 4.5.1任务/权限 | U-TILISE、DNS、RAE | 当前因果/离线分开，模型字段及预训练计费 | 原生训练与真实时间输入模型 |
| 4.5.2来源/覆盖 | U-TILISE、SSMT、EvDiff | 覆盖与内容新旧两轴，目标重算、坐标控制 | 真实运动/变化/目标兼容证据 |
| 4.5.3读者/预算 | RAE、SOUL、REPA/REG | 空间/GAP读者、独立probe、追加同解码参数摘要 | 非线性/训练预算族、等总成本 |
| 4.5.4损伤恢复 | EVSNN及L4.3/4.4近邻 | 真实post状态、原任务四组、独立内容验证 | 训练后选择性与复杂通信路径 |
| 4.5.5几何/频率 | iREPA、RAE、SOUL | 区域频率交叉项、可逆坐标、绝对量诊断 | 对未见条件/改动收益的预测 |
| 4.5.6干扰/所有者 | DMP、SSMT、DNS | 过期来源、v/a联合、mean累计和状态账 | 可靠性门控与实际部署成本 |
| 4.5.7真实分支 | U-TILISE、AllClear、EVSNN/EvDiff | 输入schema、可用时间、ROI与干净目标检查 | 用户下载后的实际内容审计 |
| 4.5.8论文/设计 | 上述全部 | 由诊断指定可检验改动 | 多seed、外推和正向设计收益 |

## 15. 固定源码、执行范围和引用责任

`references/selected_author_sources.json`保存9仓库19文本文件的commit、Git blob、SHA256、字节数和URL。`fetch_pinned_sources45.py`只获取这些固定文本；新包不重新分发完整作者源代码。`audit_spatial_authors45.py`验证全部字节，执行U-TILISE、RAE、iREPA、EVSNN及MuSpike的选定原定义/方法；DMP、SOUL、AllClear、EvDiff在本轮为代码选读，不冒充模型执行。

原作者单token/零范数等人工边界只校准接口，不据此否定其正常实验。原文缺口、不可访问仓库和未执行模型均有记录。贡献目标是SNN来源—空间任务—状态—读取—设计之间的新辨别力与可迁移收益，而不是文献工具名称的重新组合。
