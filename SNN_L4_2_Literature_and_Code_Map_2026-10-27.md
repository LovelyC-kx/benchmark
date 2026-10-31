> **执行状态更新（2026-10-29）**：本文保留原研究记录；当前代码入口、训练/验收状态、数据清单以[当前状态](docs/CURRENT_STATUS.md)和[执行手册](docs/SERVER_RUNBOOK.md)为准。旧“完成”不自动表示全量训练完成。

# L4.2文献、报告身份与代码使用地图

核查日期2026-10-27。目标是让每篇文献进入具体研究判断，不按年份堆叠名单。原L4.2七条路由全部保留，新增七篇理论/表征/潜空间近邻。共有14篇主条目，其中4篇的oral身份已从会议或作者原始记录核实；2026两篇ICLR工作只登记已核实的主会身份，不沿二手列表推定oral。

“选读正文”不表示整篇/附录逐行审计；“执行选定函数”不表示作者完整复现。8个固定源码文件的出处、commit和blob见references/selected_source_manifest.json，实际运行见evidence/selected_author_checks.json。作者源文件未在本包再次分发，可选下载器只取声明的8份代码。

## 1. 先按论文需要的逻辑组织

| 逻辑位置 | 文献 | 对应分析 |
|---|---|---|
| 已有信息与受限读者之间有差距 | Usable Information；Wakhloo；Layer by Layer | 4.2.2—4.2.3、4.2.5 |
| 潜变量方向应联系实际输出 | 概念几何；2026 Latent Concept Disentanglement | 4.2.1、4.2.4，继续4.3—4.4 |
| 表征、任务解码器与细节可能不匹配 | REPA；REG；RAE | 4.2.6—4.2.7及4.5 |
| 内容仍编码但原行为读取有偏 | Park；Amnesic Probing | 4.2.3、4.2.8及4.3 |
| 活动、优先级、维持与可访问分开 | Feature Matching；Alpha；Access；Monkey候选 | 原七条路由的认知参照，不能直接指定SNN机制 |

## 2. 新增顶会与核心理论：逐篇如何使用

### N1. Layer by Layer: Uncovering Hidden Representations in Language Models

Oscar Skean等，**ICML2025 Oral**。正式论文：[PMLR](https://proceedings.mlr.press/v267/skean25a.html)；[会议oral页面](https://icml.cc/virtual/2025/oral/47214)；[所读arXiv v2](https://arxiv.org/html/2502.02013v2)；[官方代码](https://github.com/OFSkean/information_flow)。oral身份同时由会议页面和作者出版页确认。

**做到了什么。** 比较中间层与最后层的下游表示，组织信息论、几何和扰动不变性指标，并在多种架构/任务上进行评价。它直接支持“表示所在层与最终头位置需要区分”的问题设置。

**本轮读/跑到哪里。** 选读v2的表示度量、谱熵、有效秩及附录相关证明；读取Metric-Harness.py和metric_functions.py；仅执行normalize、entropy_normalization等选定定义。没有运行原模型、MTEB、ImageNet、repitl或全部指标。

**如何适配。** 评价轴由层扩展为内容×时刻×状态载体×真实观察；把通用谱度量与指定任务风险、原头及匹配随机方向一起看。静默SNN的零范数、缺失样本和单位/中心化规范显式报告。

**值得深挖的边界。** 谱熵描述特定Gram矩阵，并非任意标签的Shannon互信息或任务充分性保证。增加无关高方差因素就能改变谱；本研究以可计算反例和下游风险校准其解释范围。

**数学核查框。** v2展示EffRank(Z)≤exp(S₁(Z))，其证明先分别归一化奇异值与平方奇异值，随后仍按逐项平方关系比较。按Roy–Vetterli型奇异值归一化定义与其式(1)的协方差谱定义，取中心化、逐行单位范数矩阵Z=[[1,0],[.6,.8],[-1,0],[-.6,-.8]]，两值分别约1.88988与1.64938，所示不等式不成立。该检查针对明确版本和定义，正式PDF是否修改尚未核齐；它不否定论文的中间层实证发现。我们的代码将两种谱秩分开命名，不把待核不等式作为保证。

### N2. The Geometry of Categorical and Hierarchical Concepts in Large Language Models

Kiho Park、Yo Joong Choe、Yibo Jiang、Victor Veitch，**ICLR2025 Oral**。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/be7430d22a4dae8516894e32f2fcc6db-Abstract-Conference.html)；[所读v3](https://arxiv.org/html/2406.01506v3)；[作者oral记录](https://yjchoe.github.io/research/park2024geometry/)；[官方代码](https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations)。

**做到了什么。** 在明确的embedding/unembedding、softmax与概念定义下，联系概念层级和表示几何，并验证指定概念操作的目标与非目标输出效应。它已经结合几何和干预。

**本轮范围。** 选读因果内积、概念可分离、LDA估计及干预段；读取from_models.py和category.py。执行get_g时仅将模型embedding加载器替换成人工张量；完整LLM、WordNet及sklearn原LDA流程未运行。

**如何适配。** 不将其softmax下的正交定理直接套到任意内部SNN状态。使用已知原头构造输出任务度量、row/null与查询行选择；内容方向、原输出效应和未来动态分别验证。秩亏spike代码使用range白化，明确丢失的零方向。

**研究推进。** 将“可读方向”和“真正进入指定输出的方向”联系到具体历史内容及实际状态所有者，为后续选择性操作与恢复提供候选。

### N3. Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think

Sihyun Yu等，**ICLR2025 Oral**。[论文](https://arxiv.org/abs/2410.06940)；[作者项目](https://sihyun.me/REPA/)；[官方代码及oral声明](https://github.com/sihyun-yu/REPA)。本轮使用摘要/作者方法说明和loss.py，HTML全文未取得，不宣称整篇复现。

**做到了什么。** 将噪声输入的扩散隐藏表示对齐到外部视觉编码器的干净表示，用于改善生成训练；作者还区分较早语义对齐与后续细节生成。

**实际代码复用。** AST执行原SILoss定义，以人工模型/张量核对cosine投影损失：同向−1、反向+1、零特征0；denoising与对齐损失分别输出。无扩散训练或原预训练特征下载。

**如何用于我们。** 作为“重塑表示以便下游利用”的C类候选，而非冻结表示的新头。若后续用于SNN，只对合法内容/时间/区域做任务条件对齐，保留语义、位置和细节任务；外部特征、额外监督和训练成本单列。

### N4. Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think

Ge Wu等，**NeurIPS2025 Oral**。[正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0b99315234cc95e6ef281f9155b68832-Abstract-Conference.html)；[所读v2](https://arxiv.org/html/2507.01467v2)；[官方代码及oral声明](https://github.com/Martinser/REG)；会议录制记录为Oral 1B Generation/Simulation。

**做到了什么。** 将低层图像latent和高层class token共同纳入生成/去噪，使语义与细节协同参与推理。它提供“分开解耦”和“有目的联合使用”之间的直接竞争思路。

**本轮范围。** 选读方法和联合加噪目标，读取固定loss.py；检查图像与class-token denoising及projection loss的分离。未运行REG模型或训练。

**如何适配。** SNN中比较细节/身份/位置因素分别编码、共享载体和合法联合读出；不能预设所有因素越解耦越好。对重建中C/H/U的协作与冲突保留不同查询；共同噪声配对和训练/部署访问需一致登记。

### N5. Diffusion Transformers with Representation Autoencoders

Boyang Zheng等，**ICLR2026主会，oral/spotlight未核实**。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3c4141c12660ad3625eb4ae845e0a6f9-Abstract-Conference.html)；[所读预印本v1](https://arxiv.org/html/2510.11690v1)；[项目](https://rae-dit.github.io/)；[官方代码](https://github.com/bytetriper/RAE)。正式PDF较大，方法选读来自所标明预印本版本。

**做到了什么。** 冻结表征编码器、训练图像解码器，检验高维表示的重建潜力，并分析下游生成网络的宽度、噪声尺度及解码训练适配。不能由一个较弱解码器失败就判断代码已丢细节。

**本轮范围。** 选读冻结编码重建、宽度与训练适配部分；固定src/stage1/rae.py，执行原encode/decode的人工identity编码器/解码器控制，核对空间reshape、归一化反变换和no_grad。原ViT、GAN、LPIPS、ImageNet及完整RAE未运行。

**如何适配。** 在SNN相同冻结空间状态上比较受控解码器、空间观察与训练预算。代码中的训练noising仍会改变观察分布，必须与参数冻结分开；本轮矩阵probe使用明确固定的捕获表示。

### N6. Latent Concept Disentanglement in Transformer-based Language Models

Guan Zhe Hong等，**ICLR2026主会，oral/spotlight未独立确认**。[正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2dc5a0faac8102fd47363795f71126ee-Abstract-Conference.html)；[所读v1](https://arxiv.org/html/2506.16975v1)。最终PDF获取受大小限制；本轮未核实官方代码仓库。

**做到了什么。** 在受控离散桥接概念和连续参数任务中，研究模型怎样形成、组合并使用潜概念；所读版本包含跨类型patch和轨迹参数的几何/操作证据，而非仅有PCA图片。

**如何适配。** 在SNN任务中分别控制历史项目、绑定关系、规则和连续因素，检验内容→表示→下游组合；读者拟合与实际原头操作分开。借用其实验逻辑，不将LLM注意力头视作SNN细胞，也不宣称我们首次将潜空间与干预连接。

**后续缺口。** 正式版本和预印本差异、官方代码及完整任务构造尚需继续核查。当前适配只依赖明确选读的方法，不补写未证实的实现。

### N7. A Theory of Usable Information under Computational Constraints

Yilun Xu等，**ICLR2020，理论基础，非近期论文**。[主记录](https://arxiv.org/abs/2002.10689)；[所读v1](https://arxiv.org/html/2002.10689v1)。

**做到了什么。** 将预测能力与可用函数族联系，区分不受计算约束的信息和受限预测者可利用的信息。

**如何使用。** 为本节函数族风险曲线和非线性特征变换提供概念依据。我们没有把MSE或准确率差直接命名为V-information；Shannon DPI与受限可用性分别陈述。实际代码是自己的解析控制与reader ladder，未声称执行作者原代码。

## 3. 原L4.2七条路由：逐项保留与调用

### O1 / GEO-WAKHLOO. Neural population geometry and optimal coding of tasks with shared latent structure

Albert J. Wakhloo、Will Slatton、SueYeon Chung，**Nature Neuroscience2026**，正式发表于2026-02-04；预印本始于2024。[正式记录](https://www.nature.com/articles/s41593-025-02183-y)；[本轮可读的方法v2](https://arxiv.org/html/2402.16770v2)；[官方代码](https://github.com/awakhloo/population_geometry_optimal_coding)。

**用途。** 连接共享潜变量随机任务、表示几何与指定Hebbian读者的学习曲线，继续沿3.x的校准计划。

**本轮实际复用。** 固定code/utils.py，执行theory_part、get_geometry等选定定义；合法Gaussian控制下两种表达一致。emp_error的零代码人工边界返回0错误，因为零margin未按半错误计；本研究对二分类零预测显式计.5。该退化案例不证明作者原数据包含这种错误。

**适配与边界。** 其读者规则、任务分布和近似条件须保留；不直接预测ridge/MLP或PSNR，也不把四个统计量声称为任意任务通用充分条件。正式全文本轮请求存在访问限制，方法读本与代码版本已分开标记。

### O2 / READ-PARK. Confirmation bias through selective readout of information encoded in human parietal cortex

Park等，**Nature Communications2025**。[原文](https://www.nature.com/articles/s41467-025-61010-x)。

**用途。** 人类序列证据判断中，信息编码与最终报告读取可能不同，为“模型错误来自编码不足还是原读出偏置”提供直接认知参照。

**本轮范围。** 重核正式记录与摘要；旧阅读卡的MEG/行为与信息分解方法记录保留。本轮未执行作者代码，未将观察性统计直接说成通路干预。

**适配。** 保留同一内容的原输出、同权限新读者、历史—目标关系操纵；具体内容作用交4.3确认。

### O3 / CAUSE-AMNESIC. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals

Elazar等，**TACL2021**。[论文](https://aclanthology.org/2021.tacl-1.10/)。

**用途。** 可解码与行为依赖分开，并把属性操作后的原任务行为作为检验对象。

**本轮范围。** 原始记录/摘要与既有方法卡；未执行其完整消除代码。L4.2据此明确自己的停止位置：新probe优胜尚未证明原模型使用F。

**后续。** 与2025 Mean Projection/LEACE及干预可靠性论文共同进入4.3的内容命中、非目标保护和风险控制，后两篇已有L4整体路由，不从本节重复认领完成。

### O4 / REC-FEATUREMATCH. Behaviorally Irrelevant Feature Matching Increases Neural and Behavioral Working Memory Readout

Karabay等，**2025**。[PubMed原始记录](https://pubmed.ncbi.nlm.nih.gov/40013416/)；DOI10.1111/psyp.70020。

**用途。** 任务无关特征的匹配也可能改变神经/行为读出，为“读出改善是否意味着目标存得更强”提供竞争解释。

**本轮范围。** 核读原始摘要中的任务、匹配与主要结果；PMC全文请求受验证页面阻挡，代码未核实。

**适配。** 在SNN中将目标内容保持与观察/扰动匹配分开。不要预设所有非目标因素必须删除；匹配是否帮助原任务要由风险判断。

### O5 / REC-ALPHA. Sustained alpha oscillations serve attentional prioritization in working memory, not maintenance

**2026**。[PubMed原始记录](https://pubmed.ncbi.nlm.nih.gov/41958633/)；DOI10.1162/IMAG.a.1199。

**用途。** 检查活动指标是否主要反映优先级，而非等同于全部内容的维持。

**本轮范围。** 原始记录/摘要和既有卡；PMC全文请求受验证页面阻挡。公开资源入口https://osf.io/dtknz仅作为待核分析代码入口，未下载或运行。

**适配。** SNN的活动强度、目标可读性、原输出贡献分别记录。生物alpha不直接对应某个SNN数值状态或衰减常数。

### O6 / REC-ACCESS. Attentional prioritization enhances the accessibility of neural representations during working memory maintenance

Zhang、Akyürek、Kruijne，**2026预印本候选**。[原始DOI](https://doi.org/10.64898/2026.05.04.722761)；[作者出版记录](https://elkanakyurek.com/publications/)。

**用途。** 核查优先级、维持与可访问性的关系；对应4.2.1—4.2.3。

**缺口。** 本轮bioRxiv正文请求403，未取得足够原始方法，官方代码未核实。保留为原计划候选，不由二手摘要补写确定性实验结论，也不将其作为本轮代码已验证的依据。

### O7 / CAND-MONKEY. Linking working memory maintenance and readout in monkey sensory and prefrontal cortex

**2026 bioRxiv候选**。[原始DOI](https://doi.org/10.64898/2026.03.13.711560)；作者机构出版记录可核其身份。

**用途。** 后续比较维持期与读取期的内容/群体关系，延续L3.5到L4的任务阶段问题。

**缺口。** 本轮原始正文请求403，方法与官方代码未齐，继续保留候选身份。没有将题名中的linking当成已完成因果路径验证。

## 4. 代码来源、许可和执行账

| 文件别名 | 所属工作 | 本轮处理 |
|---|---|---|
| layer_harness.py、layer_metrics.py | Layer by Layer | 入口和选定度量读取；仅选定定义执行 |
| concept_from_models.py、concept_category.py | 概念几何 | 白化执行/模型加载替换；LDA代码选读 |
| wakhloo_utils.py | Wakhloo | 选定NumPy函数执行；无原动物/MLP训练 |
| repa_loss.py | REPA | 人工模型下原loss定义执行 |
| reg_loss.py | REG | 选定loss及参数路径读取，未执行模型 |
| rae_stage1.py | RAE | 原encode/decode人工identity与归一化控制 |

固定commit来自本轮作者仓库记录；每份文件Git blob逐字节核验。源码只供本轮私下审阅，没有将未核许可的第三方完整文件再次分发。包内提供适配代码、出处、哈希与可选固定版本下载器。

## 5. 哪些是我们的具体适配与独立贡献目标

1. 以同一历史内容、同一原任务和同一冻结SNN核心为比较单位，建立访问、读者、训练与原输出的联合合同。
2. 将任务充分性、查询族下界、函数族风险和原头任务度量组成可计算的判别链，而非把通用几何指标直接当能力分数。
3. 对SNN静默、离散阈值、重置、aux/cache/读出所有者作实际适配，保留零预测、秩亏、尺度和空间损失边界。
4. 将新读者和真实头子空间内容分析接到后续选择性操作、恢复及设计预测。
5. 正式贡献最终由稳定的条件规律、原生分类/重建收益及对未见条件的指导兑现。当前一个弱旧模型上的读者结果只是开发证据。

这些工具近邻提供研究起点和对照；我们追求它们在SNN系统性时序能力判断中产生新的解释和设计增益，不以“所有公式都是新发明”作为价值前提。
