# SNN系统性时序能力判断与归因：论文逻辑、旧计划与文献调用总册

整理日期：2026-10-24；状态更新：2026-10-27。用途：继续构思、逐节实现、实验解释和论文写作时的共同入口。

更新说明：3.2现有独立v0.2协议与验收。下列40卡/89条等数量仍描述10月24日原归档快照；新增3.2材料单独交付，旧ZIP保留历史版本。

**这份总册先保留原计划，再把文献接到具体研究动作。** 后续不能只读最近一轮讨论，也不能见到一个新指标就重新组织研究。每个小节都要回答：它属于哪层、为什么做、沿用哪份plan、借鉴哪篇文献的什么方法、对应哪段代码、凭什么得出何种结论。

本轮建立40个检索模块、89个重点文献/任务方法条目，保留101份去重来源，其中92份文字来源共28,212行、9份为不同字节的上传PDF。文字分为1,339个连续来源片段，另登记506个不同URL入口。**原文字节和行覆盖已核验；这些数量不表示所有文献本轮都重新全文精读，也不表示所有实验已经完成。** 155个片段与独立小节显式对接，其余路由中有关键词建议和上下文保留，二者身份分别登记。


**L4.4最新交付（2026-10-27）：** 恢复、补丁与方向验证v0.1开发完成，三份深入报告及完整代码/证据包；L4.1—L4.4开发实现已交付，L4.5待逐节执行。新SNN训练0、正式0/435。以下旧进度入口和历史执行记录保留其原日期含义。

**2026-10-27最新进度入口：** 3.7独立v0.1深入复审已完成：7部分、42测试、真实接口与配对世界重放、12观察及条件噪声/有限消息/精度/FIT。3.6为50测试的独立v0.1，3.5为58测试，3.4为55测试的v0.2，3.3.3为63测试的v0.2；各节验收不混加成模型数。新SNN训练0、正式0/435、audit_test未访问；下一阶段接L4。详见文末3.7执行记录。

## 1. 先固定论文到底要讲什么

我们的中心问题仍是：**在明确任务需求、输入权限和资源条件后，SNN系统怎样保留历史内容、把它变成下游可用的表示，又在什么条件下产生收益、干扰或学习困难？能否利用诊断结果预测未见条件，并指导有效修改？**

它可以同时比较脉冲与连续系统，但不能先预设SNN必胜；也不能退回“有状态所以记得更久”的常识。真正有用的结果，是指出一种看似相同的时间能力结论为何在具体条件下改变，定位它发生在何种内容、载体、时刻和通路，并让解释接受独立检验。

| 读者的问题 | 六层各自负责什么 | 后续依赖 |
|---|---|---|
| 任务到底要求什么？ | L1：目标、输入单位、历史必要性、无序/顺序/度量时间、合法变换 | 没有此合同，后面的“记忆/更新/顺序能力”无明确含义 |
| 历史怎样进入并改变计算？ | L2：写入、泄漏、阈值/重置、辅助状态、骨架与完整运行状态 | 形成可操作、可复核的机制对象 |
| 状态保留了什么、怎样组织？ | L3：内容分解、叠加、干扰几何、指定泛化理论、载体、时间和通信 | 作为机制解释中心，为L4提供有内容含义的对象 |
| 实际输出用了什么？ | L4：观察/读出权限、原头与探针、选择性操作和恢复 | 把“可解码”推进到有限范围的使用证据 |
| 哪些条件下成立、有何用途？ | L5：时长/负荷、更新保护、真实数据、留出预测与成本 | 检验诊断能否超出同一批结果的事后解释 |
| 这些状态与表示怎样学出来？ | L6：信用分配、SG/reset梯度、编码/norm、优化课程、形成过程 | 限定前向表达、学习成功和训练成本的关系 |

六层是研究责任与证据来源，不要求正文写成六个并列排行榜章节。每一层的基础定义和工具可以进入方法/附录；只有被实验证据支持的跨层认识进入主要贡献。

## 2. 沿用三个跨层研究单元组织正文

| 单元 | 要检验的主张 | 主证据与竞争解释 | 什么时候必须收窄 |
|---|---|---|---|
| P1：能力判断的系统条件 | 某些“时序能力”判断依赖任务内容、状态/观察权限和学习条件；不同条件的交互有实质影响 | L1任务认证＋L2完整边界＋L4同权限比较；对照输入不足、读出额外权限、尺度与预算 | 只有已知权限差或不公平基线能解释时，不夸大为新能力规律 |
| P2：内容、载体、时刻与输出作用 | 机制改变特定历史内容的组织和访问，并在实际任务中产生可定位的后果 | L2操作＋L3恢复/几何＋L4目标/非目标检查及恢复；对照工作点、一般损伤、路径替代与训练差 | 只有相关图、探针差或清零损伤时，不写唯一因果归因 |
| P3：条件化规律与设计用途 | 诊断形成的条件规则能预测未见任务/架构/环境，或帮助选择修改 | L5留出选择/验证搜索/随机/强基线，L6训练适应与总成本登记 | 没有额外预测或成本收益时，如实报告边界，不把验证选择换名 |

**候选新颖性不是“我们也测了记忆、读出、干扰”。** 最近邻已经覆盖这些问题的不同部分。我们需要的是相互咬合的设置、可辨别实验与分析，以及最终支持的跨条件发现。设置贡献、经验发现和设计价值分别陈述，不用一项工具校准替代三者。

## 3. 第三层沿用原七节，并形成三条论证

| 原小节 | 在第三层做什么 | 正文论证 |
|---|---|---|
| L3.1 内容与来源表征分解 | 先知道保留的是哪一份内容和关系 | A：内容形成与载体分布 |
| L3.2 时空叠加与资源关系 | 内容维数K、历史负荷L和状态/观察预算N如何共同改变编码 | B：共同编码与任务相关干扰 |
| L3.3 目标—干扰几何 | 方向、信号、噪声和混合怎样联系恢复及损失 | B |
| L3.4 Wakhloo几何与泛化 | 在规定任务/读出/分布条件下检验几何误差预测 | B |
| L3.5 膜状态、脉冲与联合载体 | 区分减弱、消失、载体转移与观测互补 | A |
| L3.6 跨时间几何与重新编码 | 区分跨时读出失配和内容遗忘 | A与C |
| L3.7 跨层传播与通信子空间 | 源端能读出的内容是否传到接收方并影响输出 | C：跨层访问与传播 |

B是候选中心发现所在，不能在结果出现前断言已经发现固定容量竞争、普遍重置优势或时空容量守恒。A先确定内容和载体，B检验组织与任务后果，C检查局部可读性怎样连接整个系统。这样既保留高级表征分析，也避免按PR、夹角、MI各写一节而失去论证关系。

七个题名和上述组织关系来自旧对话恢复摘要。L3.1另有完整v0.1协议；L3.2–3.7的完整旧对话原稿尚未回收到可逐字核对文件，本轮调用卡是依据这些主题与现存计划整理的执行入口，不能冒充所有历史原稿均已完整恢复。

## 4. 论文各部分应该怎样使用这些材料

| 论文位置 | 要使读者相信什么 | 优先调用的旧材料 | 结果与图表职责 |
|---|---|---|---|
| 引言与相关工作 | 现有benchmark已做什么，哪个判断仍需更强证据 | `argument`、`crosslayer`、`referenceaudit`、`expandedaudit`、`supplement` | 给出具体最近邻和有限增量；不靠“缺少系统性研究”空话 |
| 任务与系统测量方法 | 实验对象可比较、可解释、可复核 | L1完成稿及1.1/1.2、数据计划、L2七节、路线图D1–D13 | 任务合同表、完整状态图、三类操作及预算表 |
| 系统条件结果P1 | 能力结论为何依赖观察和任务条件 | E0–E2、L2机制卡、L4读出权限 | 有认证任务和同权限对照的交互结果 |
| 机制解释中心P2 | 哪份内容在何载体、何时、经何路影响输出 | 第三层A/B/C、W0–W3、L4选择性操作/恢复 | 内容恢复→几何解释→目标/非目标作用与恢复的一条证据链 |
| 条件外推与设计P3 | 解释是否预测未见条件、有独立用途 | E3–E6、数据计划、L5与L6规则 | 留出条件、强选择基线、诊断/训练总成本与失败边界 |
| 讨论与限制 | 发现对谁成立、对后来架构有什么指引 | 全部失败/修订记录、共享机制与SNN具体机制区分 | 给出有适用条件的设计规则；保留非SNN也成立的认识 |
| 附录/代码 | 基础工具与所有选择可复核 | 原始计划、全部来源片段、参数、日志、版本与代码入口 | 不把仅能在附录找到的重大权限差异藏起来 |

## 5. 原plan怎样优先，发生变化时怎样处理

计划分四类：完整当前协议、早期方案、跨层论证与文献审计、未验收WIP。独立协议比一句摘要更具体；后来的明确修订只替换其所修正的规则，旧内容保留出处。**不能按日期机械地将旧文件整体作废，也不能把关联项目的授权、训练结果或数值原样移到本benchmark。**

最重要的原文件入口：

| 键 | 原计划职责 | 本轮处理 |
|---|---|---|
| `roadmap` | v0.6阅读台账、D1–D13、理论/代码审计、最小决定性实验与条件式论文稿 | 全文保留；关联每一层，修正纳入单独表 |
| `argument`、`crosslayer`、`intropilot` | 论文问题、P1/P2/P3、竞争结果、图表与阶段 | 作为正文逻辑来源，不压成任务清单 |
| `taxonomy` | 六类识别、固定策略、变体、停止条件 | 作为每节解释与归档的共同骨架 |
| `l1binary`、`l1structure`、`l1complete` | 历史窗口、无序/次序/时间、16任务合同与合法性 | 保留具体小节和已运行范围；真实适配仍待做 |
| `datasetplan` | 93条来源候选、9项G协议、下载与覆盖 | 原编号全部保留，不自动下载、不称全部实现 |
| `l2plan`、`l2complete`、`l2s1`–`l2s7` | 七节前向机制计划、具体实现与探索证据 | 逐节显式对接、附只读代码与旧日志 |
| `l3protocol` | 第三层首个A/B/C协议与参考实现 | 保存完整合同、配置和12项旧NumPy验收 |
| `l3recovery` | 七节原题名、三条论证与已接受选择 | 标明对话摘要身份和原稿缺口 |
| `handoff`、`capacityplan`、`stsepwakhloo` | 历史容量/几何工作线、W0–W3、可读性和任务要求 | 作为关联historySNN分支来源；不合并试验数量 |
| `oldaudit`、`roundaudit`、`deepreview`、`implementationgaps` | 指标错误、失败结论、代码/证据缺口 | 提升为后续读取时必须同时看到的限制 |

完整逐文件覆盖见`COVERAGE.md`；来源别名、SHA、版本、原片段和代码出处可在`registry`查询。正文不依赖用户记住当时对话。

## 6. 当前真的完成到了哪里

| 部分 | 已有产物 | 尚未由这些产物证明 |
|---|---|---|
| L1 | 合同、规则/有限分布参照、配对/区域/可用时间接口和旧验收日志 | 全部官方/真实数据适配、全部任务正式模型结果 |
| L2 | 12种参考单元、8骨架、快照/分组操作、解析控制和旧CPU探索fit | 工作点/预算/训练因素均排除后的普遍机制发现 |
| L3.1 | v0.1协议、数据/配对几何参考、12项NumPy检查 | 拟议45次fit已运行、SNN内容结论已确认 |
| L3.2 | 2026-10-25 v0.3文献/作者代码复盘、28项验收及原3个极小CPU流程的内容操作分析 | 435个拟议正式配置均未运行；尚无容量规律或机制发现 |
| L3.3 | 3.3.1、深化复盘与3.3.2；新增3.3.3组间非线性交互/条件编码/误差协议，53项检查、同一旧checkpoint八视图及同权重加性参照验收 | 组内/具体来源进一步细分、正式SNN机制结果、跨训练稳定性与L4因果确认 |
| L3.4–3.7、L4–L6 | 已有主题、跨层计划和文献方法入口；部分底层接口可复用 | 每个小节都已经独立协议定稿、实现和验收 |
| 本轮总索引 | 原文字节/行覆盖、模块—文献—来源—代码引用一致性校验 | 重新训练SNN、重新复现所有原论文、逐篇重新阅读全文 |

L2旧交付中“60次fit”和不同run标识数量应分别看原日志，不能混为独立确认实验；历史historySNN的校准和训练数字更不能与它相加。2026-10-24归档时未检出PyTorch；2026-10-25检查到CPU版PyTorch 2.14.1并完成3.2功能验收与极小流程smoke。两次记录分开；正式基准拟合仍为0。

## 7. 文献先按它在论文中的作用组织

每条重点文献记录四件事：**做到了什么、我们采用什么、要补哪种验证、不能推出什么。** 下文不是把2025/2026年论文堆在末尾：近期工作与基础理论按同一个论证位置组织。89个条目中有33条年份标为2026、29条标为2025；其中包括预印本、任务/代码条目和题名待核项，不将它们统称为62篇已完成全文审稿的正式论文。

| 文献组 | 在论文里承担什么 | 优先入口 |
|---|---|---|
| 直接benchmark与最邻近机制研究 | 确定已有覆盖及候选增量 | STP、NSA、Soul、NeuroBench、STEP、Temporal Representations、STSep、How Linear Attention Remembers |
| 任务需求与时间信息 | 避免把长序列/绑定/求和全称顺序能力 | Binary Adding原协议、MAD、MQAR、Deep Sets、NeuroGym、Beyond Rate Coding |
| 前向机制与频率解释 | 建可解析参照并确定其失效条件 | Frequency Matching、High-Frequency、重置相关、DMP、HiPPO/ESS |
| 表征容量与几何泛化 | 将内容恢复、混合、噪声和受限读出联系起来 | Dambre、Refined IPC、Inubushi、Wakhloo、Temporal Superposition |
| 载体与动态编码 | 判断内容消失还是活动/载体/编码变化 | Mongillo、Murray、Wolff、持续/静默工作记忆及近期访问研究 |
| 通信与真实使用 | 源端可读和接收/输出使用分开 | Semedo、Weiss/Coen-Cagli、FIT、null-space、Park、Binish |
| 干预与恢复有效性 | 防止一般损伤冒充特定内容归因 | Amnesic Probing、Mean Projection/LEACE、可靠性、概念消除边界、DecodeShare |
| 干扰、学习与用途 | 竞争解释及强外推参照 | 旋转/情境分离、前摄干扰、固定状态关联回忆、Palimpsa/Titans、TSNeuron、TIC、预训练公平性与资源评估 |

文献卡中的“相近”不是缺陷。“采用”也不是已经完成复现。尤其最近邻已有写入—保留—访问、随机方向/能量对照、patchback、频率设计规则、辅助慢状态、生成/重建等内容，不能再次包装成首次。

## 8. 实现前最先读什么

**L3.2时空叠加与资源关系已形成2026-10-25 v0.3复盘与可执行补充。** 本轮补入同输入需求配对、等总需求量、写入/年龄分离、发放/反馈交叉、真实CPU验收及回归/分类/冻结编码器流程smoke。下一步先审阅具体设置，再开展15个开发screen；3.3按原顺序继续。独创性、原文使用及代码以新L3.2文档和包为准。

```bash
python retrieve.py module L3.2 --full --limit 6
python retrieve.py paper GEO-TEMPORALSUPER --full
python retrieve.py paper GEO-WAKHLOO --full
python retrieve.py search "D10"
python retrieve.py dataset D01
python retrieve.py dataset G02
python retrieve.py coverage
```

单元实现按照“读原plan→读文献定义/代码→校准→受控实验→解释与归档”进行。具体操作见`IMPLEMENTATION_READING_GUIDE.md`。下文给出全部调用卡与重点文献卡；更长的原文始终保留在来源快照中。



# 各层逐节调用卡

以下40个模块连接旧计划、用途、代码、对照与文献。L1.1/L1.2、L2七节沿用独立旧协议编号；L3七题名按旧对话恢复。L1其余字母编号及L4–L6细分编号是本轮检索路由。

# L1 研究与实现调用卡

本文件的小节编号是检索路由。独立历史协议的编号身份另在每张卡标明。

## L1.1 历史必要性

**属于什么、要回答什么。** 完成目标究竟是否需要较早观察？

**原计划与当前状态。** 方法/规则校准已完成；官方适配与正式训练待做。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_L1_1_BinaryAdding_Protocol_v1.md](sources/SRC054.md)（SRC054）；[SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016）；[SNN_L1_Task_Information_Revision_2026-10-17.md](sources/SRC017.md)（SRC017）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 固定目标、查询和预测期限，比较完整历史与近期窗口的最优风险；可枚举时先给精确参照，真实数据再给经验基线。

**哪些变体改哪里、为何要控制。** 官方Binary Adding与toy adding分开；同一窗口单位；末帧充分控制；学习失败不能充当信息缺失证明。

**代码入口。** [code_reference/L2/snnbench/l1_contracts.py](code_reference/L2/snnbench/l1_contracts.py)；[code_reference/L2/snnbench/tasks.py](code_reference/L2/snnbench/tasks.py)

**文献怎样进入，而不是只列名字。**

- **BEN-STP** — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects。调用理由：复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。
- **BEN-NSA** — Neuromorphic Sequential Arena: A Benchmark for Neuromorphic Temporal Processing。调用理由：借任务覆盖、时序处理和资源比较的组织逻辑。
- **BEN-LRA** — Long Range Arena: A Benchmark for Efficient Transformers。调用理由：借多任务长序列与强基线组织，逐项审计任务需求。
- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：窗口—风险曲线、近期充分性证据与任务卡。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 只能解释指定观察和分布；解析理想总体不等于官方有限数据集，更不等于SNN成绩。

```bash
python retrieve.py module L1.1 --full
```


## L1.2 历史结构：无序、次序和度量时间

**属于什么、要回答什么。** 历史有用时，究竟需要内容、排列还是实际间隔？

**原计划与当前状态。** 方法与有限分布校准已完成；复杂任务适配待做。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_L1_2_Historical_Structure_2026-10-21.md](sources/SRC015.md)（SRC015）；[SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016）；[SNN_L1_Task_Information_Revision_2026-10-17.md](sources/SRC017.md)（SRC017）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 定义完整项多重集、无度量时间的事件序列和含时间的观察；在真正嵌套的观察下比较风险。

**哪些变体改哪里、为何要控制。** 强Deep Sets参照而非只用原始均值；位置标签泄漏；同内容异顺序；同顺序异间隔；绑定是否已被预处理完成。

**代码入口。** [code_reference/L2/snnbench/l1_contracts.py](code_reference/L2/snnbench/l1_contracts.py)；[code_reference/L2/snnbench/data.py](code_reference/L2/snnbench/data.py)；[code_reference/L2/snnbench/tasks.py](code_reference/L2/snnbench/tasks.py)

**文献怎样进入，而不是只列名字。**

- **BEN-STP** — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects。调用理由：复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。
- **TASK-DEEPSETS** — Deep Sets。调用理由：采用逐项非线性编码后置换不变汇总，而不是只比较原均值。
- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。
- **REP-RATE** — Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks。调用理由：采用计数保持、时机改变等对照区分rate与timing。
- **TASK-MQAR** — Zoology / Multi-Query Associative Recall（项目与任务入口）。调用理由：检验原始符号流上的键值绑定、查询和冲突。
- **TASK-NEUROGYM** — NeuroGym IntervalDiscrimination（任务/源码入口）。调用理由：复用刺激持续时长比较，登记通道、时点与离散化。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：历史结构分类、合法置换和反例证书。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** XOR或求和不自动要求次序；固定起点interval只支持位置计时控制。

```bash
python retrieve.py module L1.2 --full
```


## L1.C 任务合同、目标粒度与模型权限

**属于什么、要回答什么。** 模型输入、监督目标、分析真值是否被混在一起？

**原计划与当前状态。** 已有合同接口；外部数据待实例化。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016）；[SNN_L1_Task_Information_Revision_2026-10-17.md](sources/SRC017.md)（SRC017）；[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）

**执行逻辑。** 每个任务写输入单位、目标函数、查询时机、期限、可见字段、分组划分和最小需求；分类标签与连续恢复对象分别登记。

**哪些变体改哪里、为何要控制。** model_payload白名单；不把字典中的timestamps或潜变量自动喂给模型；分类一比特与完整恢复分开。

**代码入口。** [code_reference/L2/snnbench/l1_contracts.py](code_reference/L2/snnbench/l1_contracts.py)；[code_reference/L2/snnbench/runner.py](code_reference/L2/snnbench/runner.py)

**文献怎样进入，而不是只列名字。**

- **BEN-STP** — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects。调用理由：复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。
- **GEO-WAKHLOO** — Neural population geometry and optimal coding of tasks with shared latent structure。调用理由：调用作者代码定义、校准Gauss/稀疏码，预测新潜变量任务Hebbian学习曲线。
- **MECH-COMPLEXITY** — Impact of Neuron Models … A Complexity Based Classification Approach（原笔记缩写）。调用理由：参考可控统计过程上的神经元与学习规则比较。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：16任务×2目标视图的合同与后续新任务模板。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 32视图不是32独立任务；协议完成不等于所有训练runner已接入。

```bash
python retrieve.py module L1.C --full
```


## L1.D 自然配对与标签合法性

**属于什么、要回答什么。** 改变历史后，原标签还成立吗？

**原计划与当前状态。** L1配对边界已实现；L3.1配对参考已校准。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016）；[PROTOCOL.md](sources/SRC074.md)（SRC074）；[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）

**执行逻辑。** 区分同标签合法样本、重算标签的合法样本、离分布诊断扰动；优先只更换一个独立因素并固定其余内容、噪声与时刻。

**哪些变体改哪里、为何要控制。** stale原配对与跨样本拼接不同；A/B/C重采样重算对应y；不能将人工状态操作冒充自然数据。

**代码入口。** [code_reference/L2/snnbench/l1_contracts.py](code_reference/L2/snnbench/l1_contracts.py)；[code_reference/L3_1/factor_data.py](code_reference/L3_1/factor_data.py)

**文献怎样进入，而不是只列名字。**

- **REC-LINEARATTN** — How Linear Attention Remembers。调用理由：采用内容特异写入/保留/查询路径、负荷与经过长度分离和混合载体对照。
- **CAUSE-RELIABILITY** — How Reliable are Causal Probing Interventions?。调用理由：同时记录completeness、selectivity与原任务变化。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：配对证书、监督许可、分布支持与差分字段。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 同标签输出变化只是敏感性；期望风险改善才支持有害影响判断。

```bash
python retrieve.py module L1.D --full
```


## L1.E 真实时间、模拟步和可用时间

**属于什么、要回答什么。** 增加T改变了测量、计算还是可访问历史？

**原计划与当前状态。** 可用性接口及线性时间参照完成；真实数据待做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016）；[SNN_L1_Task_Information_Revision_2026-10-17.md](sources/SRC017.md)（SRC017）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 区分实际采样、重复内部仿真、确定性插值和更长窗口；记录每个派生观察的availability与预测期限。

**哪些变体改哪里、为何要控制。** 密集空白可泄漏时长；离线插值不得逐点称在线可得；固定物理时长与固定步数分别比较。

**代码入口。** [code_reference/L2/snnbench/physical.py](code_reference/L2/snnbench/physical.py)；[code_reference/L2/snnbench/l2_time.py](code_reference/L2/snnbench/l2_time.py)；[code_reference/L2/snnbench/l1_contracts.py](code_reference/L2/snnbench/l1_contracts.py)

**文献怎样进入，而不是只列名字。**

- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **REP-TEMPORALDESIGN** — Design and Analysis of Temporal Representations in Neural Networks。调用理由：核查编码—激活—解码和时间定义与本框架重叠，作为直接最近邻。
- **LEARN-TSNEURON** — Resolving the Timestep Scaling Paradox in Spiking Neural Networks with a Timestep-Scalable Neuron Model。调用理由：把前向历史与指定STE下的梯度界、周期截断和实际训练效果分开。
- **MEM-HIPPO** — HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models。调用理由：用显式历史表示机制作强连续参照，具体比较内容和时间尺度。
- **LEARN-BULLET** — Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks。调用理由：连续时间事件原生训练是比较轴，不默认所有SNN都time-bin+普通SG。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：时间语义表、可用性检查与采样一致性参照。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** dt-aware解析参考并未使全部八骨架成为物理时间模型。

```bash
python retrieve.py module L1.E --full
```


## L1.F 分类与重建的需求分离

**属于什么、要回答什么。** 正确分类是否意味着取回了历史细节？

**原计划与当前状态。** 合成区域接口完成；真实恢复协议待适配。编号身份：本轮检索路由，不声称旧独立协议已经定稿。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）；[SNN_spatiotemporal_capacity_plan_v1.md](sources/SRC022.md)（SRC022）

**执行逻辑。** 同源目标分别定义类别和完整连续内容；图像逐像素分当前可见C、仅历史可见H、对目标全程未见U。

**哪些变体改哪里、为何要控制。** 保留空间布局；空区域返回null；场景切换后的旧图mask不能算当前目标覆盖；对齐与目标时点明确。

**代码入口。** [code_reference/L2/snnbench/l1_contracts.py](code_reference/L2/snnbench/l1_contracts.py)；[code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)；[code_reference/L2/snnbench/runner.py](code_reference/L2/snnbench/runner.py)

**文献怎样进入，而不是只列名字。**

- **BEN-DNS** — The Intel Neuromorphic DNS Challenge。调用理由：作为连续输出/因果语音恢复与质量—延迟—资源参照。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。
- **GEN-MUSPIKE** — MuSpike（完整题名以原始版本核验为准）。调用理由：生成/连续任务评价已有先例，重建需具体机制需求。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：区域MSE/PSNR、类别指标和内容需求交叉表。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** U区域的恢复可来自先验；重建差不等于标签信息丢失。

```bash
python retrieve.py module L1.F --full
```


## L1.G 现成数据与自建协议覆盖

**属于什么、要回答什么。** 哪条论证有现成任务，哪种严格控制仍需构造？

**原计划与当前状态。** 来源目录已有；用户下载与正式适配待做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）；[SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016）

**执行逻辑。** 沿用D01–D93来源候选与另外G01–G09补充协议，按任务需求、时间合法性和论文证据用途选择，不把下载目录当已验收主表。

**哪些变体改哪里、为何要控制。** 来源版本、原始分组、下载状态；因果/离线恢复；合成补充保留独立G编号；不自动下载用户保留的数据。

**代码入口。** [code_reference/L2/snnbench/data.py](code_reference/L2/snnbench/data.py)；[code_reference/L2/snnbench/tasks.py](code_reference/L2/snnbench/tasks.py)；[code_reference/L2/snnbench/upstream.py](code_reference/L2/snnbench/upstream.py)

**文献怎样进入，而不是只列名字。**

- **BEN-NSA** — Neuromorphic Sequential Arena: A Benchmark for Neuromorphic Temporal Processing。调用理由：借任务覆盖、时序处理和资源比较的组织逻辑。
- **BEN-HEIDELBERG** — The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks。调用理由：沿用SHD/SSC来源与说话者等划分，审计实际时序需求。
- **BEN-DNS** — The Intel Neuromorphic DNS Challenge。调用理由：作为连续输出/因果语音恢复与质量—延迟—资源参照。
- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。
- **TASK-NEUROGYM** — NeuroGym IntervalDiscrimination（任务/源码入口）。调用理由：复用刺激持续时长比较，登记通道、时点与离散化。
- **BEN-LRA** — Long Range Arena: A Benchmark for Efficient Transformers。调用理由：借多任务长序列与强基线组织，逐项审计任务需求。
- **TASK-MQAR** — Zoology / Multi-Query Associative Recall（项目与任务入口）。调用理由：检验原始符号流上的键值绑定、查询和冲突。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：93条来源候选、9项派生协议、下载队列与任务适配工单。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 候选数是目录规模；当前没有完成93条数据适配或正式结果。

```bash
python retrieve.py module L1.G --full
```


# L2 研究与实现调用卡

本层七节与已交付L2独立小节文件逐项对接。

## L2.1 完整状态、路径与更新时间

**属于什么、要回答什么。** 过去能沿哪些运行变量继续作用？

**原计划与当前状态。** 已实现并有历史验收日志。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[section_2_1.md](sources/SRC063.md)（SRC063）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）；[SNN_L2_Complete_2026-10-21.md](sources/SRC062.md)（SRC062）

**执行逻辑。** 登记Cell的v/a/syn、RNN/TCN cache、读出累计与时钟；逐步核对pre→事件→post→辅助更新；用快照继续运行。

**哪些变体改哪里、为何要控制。** 流式/批处理一致；快照克隆与权重指纹；core与system分清；诊断trace不是自主记忆。

**代码入口。** [code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)；[code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)；[code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)

**文献怎样进入，而不是只列名字。**

- **BEN-STP** — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects。调用理由：复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。
- **REC-LINEARATTN** — How Linear Attention Remembers。调用理由：采用内容特异写入/保留/查询路径、负荷与经过长度分离和混合载体对照。
- **BEN-NEUROBENCH** — The neurobench framework for benchmarking neuromorphic computing algorithms and systems。调用理由：继承算法与系统指标边界，分别报逻辑工作量和实测部署。
- **REC-TITANS** — Titans: Learning to Memorize at Test Time。调用理由：把测试时学习记忆与固定参数递归作为不同状态系统登记。
- **REC-COMPLEMENTARY** — A complementary learning system for continual episodic memory in large language models。调用理由：对照事件记忆的互补系统与检索载体。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：状态账本、更新时间表和可恢复接口。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** StreamSession为no_grad推理接口，不能代替BPTT；逻辑张量字节不等于峰值显存。

```bash
python retrieve.py module L2.1 --full
```


## L2.2 写入幅度、编码与工作点

**属于什么、要回答什么。** 泄漏或神经元差异是否混入输入幅度与阈值工作点？

**原计划与当前状态。** 接口与算子校准完成；正式匹配网格待做。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[section_2_2.md](sources/SRC064.md)（SRC064）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 显式写J=g·eta(beta)·x，保留原始写入与DC归一化两轨，记录电流、margin与自然事件比例。

**哪些变体改哪里、为何要控制。** beta=1拒绝DC归一化；共同编码器；发放率为潜在中介，事后匹配不自动识别纯效应。

**代码入口。** [code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)；[code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)

**文献怎样进入，而不是只列名字。**

- **REP-FREQUENCY** — Frequency Matching in Spiking Neural Networks for mmWave Sensing。调用理由：借频率匹配到超参规则的闭环；在留出条件检验规则并对照连续滤波。
- **REP-HIGHFREQ** — Spiking Neural Networks Need High-Frequency Information。调用理由：分别审时间/空间/事件频率轴和残差、DWC、pool的实际作用。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **BEN-RETHINK** — Rethinking the performance comparison between SNNS and ANNS。调用理由：把输入、模型规模、训练和比较权限具体化。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：工作点账、稳态/瞬态校准和输入编码核验。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** DC匹配不等于所有瞬态或任务条件都匹配。

```bash
python retrieve.py module L2.2 --full
```


## L2.3 衰减、时间常数与频率响应

**属于什么、要回答什么。** 线性泄漏能解释到哪里，何时失效？

**原计划与当前状态。** 线性频率和时间校准完成。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[section_2_3.md](sources/SRC065.md)（SRC065）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 以连续无重置递推建立可解析核、幅相和物理时间参照；再检查事件驱动系统的自然响应。

**哪些变体改哪里、为何要控制。** 有限窗与稳态分开；时间/空间频率分开；信号与噪声共同缩放；泄漏不保证整个网络低通。

**代码入口。** [code_reference/L2/snnbench/l2_time.py](code_reference/L2/snnbench/l2_time.py)；[code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)

**文献怎样进入，而不是只列名字。**

- **REP-FREQUENCY** — Frequency Matching in Spiking Neural Networks for mmWave Sensing。调用理由：借频率匹配到超参规则的闭环；在留出条件检验规则并对照连续滤波。
- **REP-HIGHFREQ** — Spiking Neural Networks Need High-Frequency Information。调用理由：分别审时间/空间/事件频率轴和残差、DWC、pool的实际作用。
- **MEM-ESS** — Quantifying Memory Utilization with Effective State-Size。调用理由：把分配状态和实际可利用记忆区分，作为线性算子参照。
- **MEM-HIPPO** — HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models。调用理由：用显式历史表示机制作强连续参照，具体比较内容和时间尺度。
- **THEORY-SSMFREQ** — Tuning Frequency Bias of State Space Models（旧计划题名）。调用理由：用连续状态模型频率偏置作为泄漏机制的竞争参照。
- **THEORY-INUBUSHI** — Reservoir Computing Beyond Memory-Nonlinearity Trade-off。调用理由：用混合动力学和局部扰动对照解释记忆/非线性权衡。
- **INT-PRIMACY** — Emergence of the Primacy Effect in Structured State-Space Models。调用理由：作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。
- **INT-RECENCY** — Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing。调用理由：纳入recency与over-smoothing两种连续状态瓶颈作强替代解释。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：解析—数值对照、有效视窗与频率轴说明。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 不能把亚阈值LTI公式直接当完整硬重置网络的信息定理。

```bash
python retrieve.py module L2.3 --full
```


## L2.4 阈值、发放事件与重置反馈

**属于什么、要回答什么。** 离散输出、反馈截断和训练梯度各改了什么？

**原计划与当前状态。** 分支与交叉控制完成；已有探索pilot，内容确认待做。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[section_2_4.md](sources/SRC066.md)（SRC066）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）；[SNN_L2_Complete_2026-10-21.md](sources/SRC062.md)（SRC062）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）

**执行逻辑。** 使用continuous/threshold/analog_hard_ref/lif_hard二维交叉；emit/reset/aux分支独立；实现路径核、自然扰动和SG分别记录。

**哪些变体改哪里、为何要控制。** 同电流局部校准；冻结闭环重算；训练重训另算；事件回放是否借用未来；reset_strength适用范围。

**代码入口。** [code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)；[code_reference/L2/snnbench/experiments.py](code_reference/L2/snnbench/experiments.py)

**文献怎样进入，而不是只列名字。**

- **MECH-STSEP** — Unleashing Temporal Capacity of Spiking Neural Networks through Spatiotemporal Separation。调用理由：沿去状态与时空分支现象，检验工作点、读出、路径替代和内容竞争解释。
- **MECH-RESET** — Revisiting Reset Mechanisms（旧计划简称；完整题名待核）。调用理由：沿记忆/发放模块和反馈复位的已有区分，设计具体分支控制。
- **REP-RATE** — Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks。调用理由：采用计数保持、时机改变等对照区分rate与timing。
- **LEARN-TSNEURON** — Resolving the Timestep Scaling Paradox in Spiking Neural Networks with a Timestep-Scalable Neuron Model。调用理由：把前向历史与指定STE下的梯度界、周期截断和实际训练效果分开。
- **THEORY-RMP** — RMP-SNN（原笔记简称；Residual Membrane Potential类方法）。调用理由：区分硬/软重置、ANN转换与直接训练。
- **THEORY-INFLOR** — InfLoR-SNN（原笔记简称；完整题名待核）。调用理由：沿MPR前向等价性建立固定计算、改变训练的对照。
- **MECH-COMPLEXITY** — Impact of Neuron Models … A Complexity Based Classification Approach（原笔记缩写）。调用理由：参考可控统计过程上的神经元与学习规则比较。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：分支证据卡、事件时机试验与解释边界。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 11变体非11个单因素开关；事件发生并不等于有选择地删除了无关内容。

```bash
python retrieve.py module L2.4 --full
```


## L2.5 辅助状态、异质时间尺度与耦合

**属于什么、要回答什么。** 清膜以后，历史还能经哪个状态返回？

**原计划与当前状态。** 分组清除与预算探索完成；异质性确认待做。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[section_2_5.md](sources/SRC067.md)（SRC067）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）；[SNN_L2_Complete_2026-10-21.md](sources/SRC062.md)（SRC062）

**执行逻辑。** 分别清v、aux、cell；区分慢突触、阈值适应与电流适应；同宽、同状态量、参数/工作点分别报告。

**哪些变体改哪里、为何要控制。** 双状态阳性控制；state字节按空间尺寸计；plif学习与hetero初始化分开；不把ref单元称原论文复现。

**代码入口。** [code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)；[code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)

**文献怎样进入，而不是只列名字。**

- **MEM-DMP** — Algorithm–hardware co-design of neuromorphic networks with dual memory pathways。调用理由：登记快慢路径、低维慢状态和状态/更新成本。
- **MEM-SYNAPTIC** — Synaptic Theory of Working Memory。调用理由：把低活动/突触载体列入状态边界，做载体分组操作。
- **MEM-HIPPO** — HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models。调用理由：用显式历史表示机制作强连续参照，具体比较内容和时间尺度。
- **MECH-OSCILLATION** — Efficient and robust temporal processing with neural oscillations modulated spiking neural networks。调用理由：作为多时间尺度和振荡机制专项阅读入口。
- **REC-ATTRACTOR** — Internal state dynamically gates task-specific attractor dynamics in prefrontal cortex。调用理由：把内部状态对任务特异动力学的门控列为替代机制。
- **REC-PALIMPSA** — Learning to Remember, Learn, and Forget in Attention-Based Models。调用理由：参考可学习写入/遗忘与存储规则。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：载体路径与资源切片，交给L3测内容。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 清a后任务变差也可能因兴奋性变化，不能直接说特定记忆被抹除。

```bash
python retrieve.py module L2.5 --full
```


## L2.6 骨架路径、传播与局部替代

**属于什么、要回答什么。** 局部去状态为何未必切断历史？

**原计划与当前状态。** 八骨架接口和探索路径验收已有。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[section_2_6.md](sources/SRC068.md)（SRC068）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）；[SNN_L2_Complete_2026-10-21.md](sources/SRC062.md)（SRC062）

**执行逻辑。** 八骨架逐项登记时间cache、当帧残差与跨层Cell；比较cell/cache/core/system清除，必要时按层定位。

**哪些变体改哪里、为何要控制。** spike_attention没有时间KV；cnn1d轴非默认时间；U-Net当帧skip非历史cache；读出累计独立清理。

**代码入口。** [code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)

**文献怎样进入，而不是只列名字。**

- **MECH-STSEP** — Unleashing Temporal Capacity of Spiking Neural Networks through Spatiotemporal Separation。调用理由：沿去状态与时空分支现象，检验工作点、读出、路径替代和内容竞争解释。
- **REC-LINEARATTN** — How Linear Attention Remembers。调用理由：采用内容特异写入/保留/查询路径、负荷与经过长度分离和混合载体对照。
- **BEN-STEP** — STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking。调用理由：借统一组件替换和可复现平台设置；我们的增量必须落到内容/通路证据。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **REP-HIGHFREQ** — Spiking Neural Networks Need High-Frequency Information。调用理由：分别审时间/空间/事件频率轴和残差、DWC、pool的实际作用。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：路径替代实例与多骨架验证范围。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 单种子未匹配预算的扩展不作架构排名；相同准确率不等于状态数值相同。

```bash
python retrieve.py module L2.6 --full
```


## L2.7 机制证据卡与跨层交接

**属于什么、要回答什么。** 本层交给表征层的对象和证据究竟是什么？

**原计划与当前状态。** L2交付完成；后续内容和用途确认未完成。编号身份：沿用独立旧协议编号。完整独立协议优先；原文片段可按下方命令检索。

原计划：[section_2_7.md](sources/SRC069.md)（SRC069）；[SNN_L2_Complete_2026-10-21.md](sources/SRC062.md)（SRC062）；[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）；[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）

**执行逻辑。** 每个操作记录checkpoint、完整范围、切点、事件/工作点、自然输入与人工操作身份，再连接内容和任务结果。

**哪些变体改哪里、为何要控制。** 全部结果留档；解析/校准/pilot/确认分开；试验次数与独立配置数分开。

**代码入口。** [code_reference/L2/snnbench/experiments.py](code_reference/L2/snnbench/experiments.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)

**文献怎样进入，而不是只列名字。**

- **BEN-NEUROBENCH** — The neurobench framework for benchmarking neuromorphic computing algorithms and systems。调用理由：继承算法与系统指标边界，分别报逻辑工作量和实测部署。
- **CAUSE-RELIABILITY** — How Reliable are Causal Probing Interventions?。调用理由：同时记录completeness、selectivity与原任务变化。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：机制卡、已完成/待完成表和可引用方法描述。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 60次CPU fit不是60个独立科学结论；重复run标识不可重复计数。

```bash
python retrieve.py module L2.7 --full
```


# L3 研究与实现调用卡

第三层沿用回收的七个原题名，3.1有完整独立协议，其余原对话仅恢复主题/摘要；不冒充已回收全部原稿。

## L3.1 内容与来源表征分解

**属于什么、要回答什么。** 状态里保留的是当前、有效历史、无关历史还是关系？

**原计划与当前状态。** v0.1协议及12项NumPy检查完成；新SNN训练为0。编号身份：L3原题名由对话摘要恢复；3.1另有完整协议。完整独立协议优先；原文片段可按下方命令检索。

原计划：[PROTOCOL.md](sources/SRC074.md)（SRC074）；[conversation_plan_recovery.md](sources/SRC084.md)（SRC084）；[STSep_LIF_Wakhloo_Research_Protocol.md](sources/SRC023.md)（SRC023）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 沿用A/B/C独立潜变量、内容标签与单因素配对；在独立数据上测目标/非目标恢复和条件/边际内容响应。

**哪些变体改哪里、为何要控制。** 共用无状态非线性编码器；时间位置平衡；basis只在geometry集拟合；rank=0空基；单一因素重采样重算y。

**代码入口。** [code_reference/L3_1/factor_data.py](code_reference/L3_1/factor_data.py)；[code_reference/L3_1/paired_geometry.py](code_reference/L3_1/paired_geometry.py)；[code_reference/L3_1/l2_adapter_example.py](code_reference/L3_1/l2_adapter_example.py)

**文献怎样进入，而不是只列名字。**

- **GEO-WAKHLOO** — Neural population geometry and optimal coding of tasks with shared latent structure。调用理由：调用作者代码定义、校准Gauss/稀疏码，预测新潜变量任务Hebbian学习曲线。
- **GEO-TEMPORALSUPER** — Temporal superposition and feature geometry of RNNs under memory demands。调用理由：把内容维数、年龄需求、稀疏性和网络维数共同纳入，先核线性回忆参照再测非线性系统。
- **MECH-STSEP** — Unleashing Temporal Capacity of Spiking Neural Networks through Spatiotemporal Separation。调用理由：沿去状态与时空分支现象，检验工作点、读出、路径替代和内容竞争解释。
- **MEM-MURRAY** — Stable population coding for working memory coexists with heterogeneous neural dynamics in prefrontal cortex。调用理由：并行测稳定群体读出与单元/跨时动态，避免把变化编码误作遗忘。
- **REP-TRQR** — MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks。调用理由：保留冗余诊断→门控→读出/成本闭环，明确估计样本轴与新增目标证据。
- **REP-TRE** — Temporal Representation Enhancement (TRE): Learning to Forget Dominant Patterns for Enhanced Temporal Spiking Features。调用理由：将类别贡献、互补特征和训练门控与真正新增标签证据分别分析。
- **REP-TMI** — Robust Spiking Neural Networks by Temporal Mutual Information。调用理由：核对MI随机变量、KDE轴与估计器、训练和攻击接口；理论用于其实际对象。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：内容×载体×时间的恢复图和配对响应；作为第三层论证A起点。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 配对敏感性不是语义理解；条件残差几何不能代替边际干扰功率；45拟议fit尚未执行。

```bash
python retrieve.py module L3.1 --full
```


## L3.2 时空叠加与资源关系

**v0.3当前复盘与代码。** [SNN_L3_2_Review_v0_3_2026-10-25.md](../SNN_L3_2_Review_v0_3_2026-10-25/SNN_L3_2_Review_v0_3_2026-10-25.md)；同名ZIP含本节完整代码与验收。R3补核编码器去除方法与作者代码；R4补核论文内实现；R6/R8/R9补核选定源码。新增操作后匹配特征、固定自然探针、单来源配对与即时编码诊断。原v0.2保留作历史；未新增正式训练。共同编码器控制、完整循环RNN参照和需求受限区检查见新复盘的必要条件。

**2026-10-25续稿入口。** [SNN_L3_2_Protocol_v0_2_2026-10-25.md](../SNN_L3_2_Protocol_v0_2_2026-10-25/SNN_L3_2_Protocol_v0_2_2026-10-25.md)；同名ZIP包含文献映射、数据、解析参照、模型/训练/探针和验收。旧WIP保留作来源，不再作为当前实现状态。新增等总需求量对照与Perrinet 2026异质延迟工作记忆参照，详见新包LITERATURE_MAP。

**属于什么、要回答什么。** 内容维数、需要保留的历史份数与可用状态如何共同改变编码？

**原计划与当前状态。** 2026-10-25 v0.3复盘完成：核查九篇文献与选定作者代码，28项验收，原3个极小CPU checkpoint的固定探针/单来源配对/即时编码诊断完成；正式基准拟合0。编号身份：L3原题名由对话摘要恢复；3.1另有完整协议。完整独立协议优先；原文片段可按下方命令检索。

原计划：[conversation_plan_recovery.md](sources/SRC084.md)（SRC084）；[SNN_spatiotemporal_capacity_plan_v1.md](sources/SRC022.md)（SRC022）；[SNN_AI_execution_handoff_v2.md](sources/SRC002.md)（SRC002）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 保留K×L×N三个轴；区分物理状态Ncore、观测Dobs和编码器瓶颈；先建线性可逆/带噪参照，再做学习后编码与冻结操作。

**哪些变体改哪里、为何要控制。** 固定同一输入改变监督L需重训；固定anchor损失防稀释；raw/fixed-RMS两轨；年龄与位置平衡；多视角证据不自动算独立内容负荷。

**代码入口。** 当前以v0.3包的load_data.py、torch_adapter.py、analyze_features.py、intervention_features.py与analyze_interventions.py为准；[WIP/L3_2_load_data.py](WIP/L3_2_load_data.py)保留作历史来源。

**文献怎样进入，而不是只列名字。**

- **GEO-TEMPORALSUPER** — Temporal superposition and feature geometry of RNNs under memory demands。调用理由：把内容维数、年龄需求、稀疏性和网络维数共同纳入，先核线性回忆参照再测非线性系统。
- **THEORY-IPC** — Information Processing Capacity of Dynamical Systems。调用理由：按正交目标族、线性读出和总体投影条件使用容量总和。
- **THEORY-REFINED** — A refined information processing capacity metric allows an in-depth analysis of memory and nonlinearity trade-offs in neurocomputational systems。调用理由：在SNN编码、噪声和观测条件下校准容量计算。
- **MEM-ESS** — Quantifying Memory Utilization with Effective State-Size。调用理由：把分配状态和实际可利用记忆区分，作为线性算子参照。
- **REC-ASSOCWALL** — Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It。调用理由：保留matched-state、干扰墙与课程改变的竞争解释。
- **REC-LINEARATTN** — How Linear Attention Remembers。调用理由：采用内容特异写入/保留/查询路径、负荷与经过长度分离和混合载体对照。
- **MECH-STSEP** — Unleashing Temporal Capacity of Spiking Neural Networks through Spatiotemporal Separation。调用理由：沿去状态与时空分支现象，检验工作点、读出、路径替代和内容竞争解释。
- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。
- **REC-VARIATIONAL** — Variational Linear Attention: Stable Associative Memory for Long-Context Transformers。调用理由：追踪关联存储稳定性与写入规则对负荷的影响。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：负荷—内容恢复—编码条件数/方向关系；第三层论证B候选中心。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 非正交不必然丢信息；高斯致密控制不能代替稀疏superposition机制；本轮实现验收和极小smoke不构成435个拟议实验的正式结果。

```bash
python retrieve.py module L3.2 --full
```


## L3.3 目标—干扰几何

**3.3.1当前入口（2026-10-25 v0.1）。** [SNN_L3_3_1_Decomposition_v0_1_2026-10-25.md](../SNN_L3_3_1_Decomposition_v0_1_2026-10-25/SNN_L3_3_1_Decomposition_v0_1_2026-10-25.md)。已形成逐坐标focal/共同目标/未请求角色、联合编码、二阶诊断、三类条件重复及14项验收。继承3.2原有单个极小checkpoint的C/H1两个视图，未新增模型训练，未访问audit。历史L2 geometry的noise字段仅为联合线性残差归一化，不能代替focal-only读出的全部非目标变异；本节新代码不调用该字段。dPCA与Wakhloo选定作者函数的核查范围见新包；dPCA新增记录不回填为10/24快照已精读文献。

**深化复盘入口（2026-10-25 v0.1）。** [SNN_L3_3_Deepening_Audit_v0_1_2026-10-25.md](../SNN_L3_3_Deepening_Audit_v0_1_2026-10-25/SNN_L3_3_Deepening_Audit_v0_1_2026-10-25.md)。将已有3.2风险函数细化为focal/共同目标/未请求/余项/偏置，接入非线性也成立的条件风险恒等式、有限重复修正、可逆混合、角色与噪声pick-freeze、自然/固定事件及SG对照。23项校准通过；复用同一旧checkpoint检查，没有新增SNN训练或访问audit。补充全零外部输入仍可有核心bias驱动，以及初始直接kernel为零但自然事件相位仍可不同的已知答案，避免把固定路径贡献消失等同内容遗忘。新文献记录明确2025 LIF读出工作与2026 split-trial预印本的使用和未复现范围；不改写10/24文献计数。该复盘时3.3.2尚待实现，现以以下独立协议为准。

**3.3.2当前入口（2026-10-25 v0.1）。** [SNN_L3_3_2_Subspaces_v0_1_2026-10-25.md](../SNN_L3_3_2_Subspaces_v0_1_2026-10-25/SNN_L3_3_2_Subspaces_v0_1_2026-10-25.md)。按“编码/解码/协方差度量→独立拟合和秩选择→重叠与功率参照→条件协方差→实际读出误差→观察限制和学习曲线”组织。新增subspace_geometry.py、readout_geometry.py、sampling.py、run_geometry.py；35项检查通过，另有已知线性系统和同一旧checkpoint的六排列/H1/joint共八视图验收。允许秩0与未定义角度；独立replica、Haar/协方差参照、明确同秩同方差构造均有合同；编码投影可能损失噪声参照。核读RNN作者一个notebook的全部10个代码单元，明确tied读出、二阶矩和示例选择范围；未执行作者训练。没有新增SNN训练或访问audit，正式模型规律和L4因果确认仍未完成。新的文献/代码核查不追补为10/24历史文献计数。

**3.3.3当前入口（2026-10-25 v0.1）。** [SNN_L3_3_3_Nonlinear_v0_1_2026-10-25.md](../SNN_L3_3_3_Nonlinear_v0_1_2026-10-25/SNN_L3_3_3_Nonlinear_v0_1_2026-10-25.md)。按“函数依赖/可读/利用→独立因素合同→完整组子集ANOVA→指定输出与误差→合法非线性读出→排列条件与权限→机制零参照”组织。活动F/G/U/E最多16组合，完整仅指组间；误差函数重算各hybrid真值，保留负估计和成组区间。新增线性/二次/MLP、六排列迁移与共享控制、独立条件重复及可选嵌套条件函数。53项检查、已知有限总体、Gaussian估计器校准、同一旧checkpoint八视图通过；另用同权重continuous前向覆盖验证当前骨架跨写入加性零参照，不计为训练基线。SNN接入中104个probe拟合含32个MLP fit/5120次读出更新，与新增SNN训练0明确分开。核读2025 ANOVA-TPNN及2026 in-context superposition选定作者文件；经典分解/几何/抑制并非原创，需以非线性任务后果和留出预测形成增量。未访问audit，正式benchmark仍0/435；分类/真实图像适配、多训练seed及L4未完成。历史文献/模块注册快照不回填为已精读或已运行。

**属于什么、要回答什么。** 历史变化为何会沿任务敏感方向影响当前目标？

**原计划与当前状态。** 3.3.1、3.3.2及3.3.3 v0.1协议、初版实现和校准完成；3.3.3完整覆盖声明的因素组之间作用，不等于全部组内细节或整个第三层完成。正式SNN实验未运行。编号身份：L3原题名由对话摘要恢复；3.1另有完整协议。完整独立协议优先；原文片段可按下方命令检索。

原计划：[conversation_plan_recovery.md](sources/SRC084.md)（SRC084）；[STSep_LIF_Wakhloo_Research_Protocol.md](sources/SRC023.md)（SRC023）；[SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[historySNN_expert_audit_20261012.md](sources/SRC024.md)（SRC024）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 分别估计目标编码、非目标配对协方差、目标对齐干扰和噪声；结合条件数、噪声与受限读出解释已确认的恢复差。

**哪些变体改哪里、为何要控制。** 独立估计子空间；匹配随机方向/维数；可逆混合零模型；条件与边际统计分开；非线性交互与线性叠加干扰分开。

**代码入口。** 当前3.3.1使用新包roles.py、decomposition.py、repeat_panels.py与run_decomposition.py；[code_reference/L2/snnbench/analysis.py](code_reference/L2/snnbench/analysis.py)为历史描述接口，注意上述残差字段边界；[code_reference/L3_1/paired_geometry.py](code_reference/L3_1/paired_geometry.py)保留作后续配对几何基础。

**文献怎样进入，而不是只列名字。**

- **GEO-WAKHLOO** — Neural population geometry and optimal coding of tasks with shared latent structure。调用理由：调用作者代码定义、校准Gauss/稀疏码，预测新潜变量任务Hebbian学习曲线。
- **GEO-TEMPORALSUPER** — Temporal superposition and feature geometry of RNNs under memory demands。调用理由：把内容维数、年龄需求、稀疏性和网络维数共同纳入，先核线性回忆参照再测非线性系统。
- **INT-ROTATION** — Rotational dynamics reduce interference between sensory and memory representations。调用理由：将感觉—记忆方向旋转与干扰、行为效用并测。
- **INT-FLESCH** — Orthogonal representations for robust context-dependent task performance in brains and neural networks。调用理由：以情境相关表示分离解释泛化与干扰，检查噪声/任务条件。
- **INT-STREAMROT** — Orthogonal-Rotational Dynamics Supports Efficient Encoding and Updating for Streaming Information in Working Memory。调用理由：把流式更新中的旋转/重排作为保持与更新兼顾的候选。
- **REC-SENSORYINT** — Overcoming sensory-memory interference in working memory circuits。调用理由：比较门控、吸引子和动态编码避免感觉—记忆干扰的不同解释。
- **MECH-STSEP** — Unleashing Temporal Capacity of Spiking Neural Networks through Spatiotemporal Separation。调用理由：沿去状态与时空分支现象，检验工作点、读出、路径替代和内容竞争解释。
- **REP-TRE** — Temporal Representation Enhancement (TRE): Learning to Forget Dominant Patterns for Enhanced Temporal Spiking Features。调用理由：将类别贡献、互补特征和训练门控与真正新增标签证据分别分析。
- **CAND-OPERATIONS** — Working memory operations emerge from dynamic changes in neural subspace geometry。调用理由：作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。
- **CAND-HIGHROT** — Efficient Working Memory Maintenance via High-Dimensional Rotational Dynamics。调用理由：作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。
- **READ-TRANSFER** — Measuring Stimulus Information Transfer Between Neural Populations Through the Communication Subspace。调用理由：加入内容方向、接收噪声与有效传输的线性Fisher分析。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：内容特异几何和线性风险分解；为L4选择干预方向。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 夹角、PR或投影方差单独不能证明信息损失、竞争或因果干扰。

```bash
python retrieve.py module L3.3 --full
```


## L3.4 Wakhloo几何与泛化

**属于什么、要回答什么。** 几何统计能否预测新任务的指定读出误差？

**原计划与当前状态。** v0.2按W0–W3完成细化复审与执行：有限p Gaussian精确参照、逐步分布替换、正交化高阶对齐与最优仿射方差控制、任务方向/支持、状态能量与任务谱/harmonic/PCR、连续目标有限p MSE，以及冻结后配对读取收益界。55项测试；13个语义视图×F/G/FG；13连续恢复视图、1248个闭式映射；仍需正式训练模型与多seed检验。既有专项协议和historySNN结果保留原身份。编号身份：L3原题名由对话摘要恢复；3.1另有完整协议。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_AI_execution_handoff_v2.md](sources/SRC002.md)（SRC002）；[STSep_LIF_Wakhloo_Research_Protocol.md](sources/SRC023.md)（SRC023）；[historySNN_expert_audit_20261012.md](sources/SRC024.md)（SRC024）；[SNN_round1_independent_review_20260919.md](sources/SRC021.md)（SRC021）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）

**执行逻辑。** 按W0源码/版本→W1 Gaussian等价校准→W2稀疏码适用性→W3随机潜变量任务执行；与ridge、受限MLP和真实头并行。

**哪些变体改哪里、为何要控制。** c/f/s/PR按原定义；ssf与snf顺序；独立统计与任务方向；ties/奇异协方差/零信号；预测不拟合修正系数。

**代码入口。** 当前为SNN_L3_4_Geometry_Generalization_v0_2代码包：geometry_generalization、finite_gaussian、task_structure、higher_order_controls、task_spectrum、continuous_risk、paired_readouts与run_focused_review等；原code_reference/L2/snnbench/analysis.py是历史入口。

**文献怎样进入，而不是只列名字。**

- **GEO-WAKHLOO** — Neural population geometry and optimal coding of tasks with shared latent structure。调用理由：调用作者代码定义、校准Gauss/稀疏码，预测新潜变量任务Hebbian学习曲线。
- **THEORY-IPC** — Information Processing Capacity of Dynamical Systems。调用理由：按正交目标族、线性读出和总体投影条件使用容量总和。
- **THEORY-INUBUSHI** — Reservoir Computing Beyond Memory-Nonlinearity Trade-off。调用理由：用混合动力学和局部扰动对照解释记忆/非线性权衡。
- **THEORY-REFINED** — A refined information processing capacity metric allows an in-depth analysis of memory and nonlinearity trade-offs in neurocomputational systems。调用理由：在SNN编码、噪声和观测条件下校准容量计算。


- **GEO-RC-MODES-2026** — Organization of computation in reservoir computing（arXiv2607.17858v1，2026-07）。模式能量/任务投影；原文§2.1/2.2；未核实官方代码，按方程独立实现。
- **THEORY-HARMONIC-2026** — Memory Uncertainty Relation and Harmonic Memory in Random Recurrent Networks（PRE2026；arXiv2605.24628v1）。harmonic/OLS间隙及PCR噪声；官方mur仓库固定commit，核读3选定文件，未跑全仓仿真。
- **DIST-COMPLEXITY** — Neural networks trained with SGD learn distributions of increasing complexity（2022预印本/2023ICML/2025JSTAT）。匹配分布clone；固定dist_inc_comp commit，核读3文件，适配为两类不同的训练分布参照。
- **GAUSS-UNIVERSALITY-2026** — Gaussian universality in neural network dynamics with generalized structured input distributions（2024预印本/2025修订/2026JSTAT）。结构和标准化条件；固定GaussianUniversality commit，核读3文件。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：理论—经验Hebbian误差随样本量的校准、失败区和留出增量。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** Hebbian分类公式不预测任意分类器或图像MSE；旧historySNN校准不可冒充当前重跑。

```bash
python retrieve.py module L3.4 --full
```


## L3.5 膜状态、脉冲与联合载体

**属于什么、要回答什么。** 内容减弱是消失还是换了载体？

**原计划与当前状态。** 2026-10-26已完成独立v0.1深入复审和工程执行，完整文档为SNN_L3_5_Deep_Review_and_Execution_2026-10-26.md，配套包为SNN_L3_5_Carriers_State_Timing_v0_1_2026-10-26.zip。六子节覆盖阶段/复位/精度、稀疏歧义、联合增量、时序窗口、完整运行状态、访问预算；58测试和480续跑通过。多训练seed、真实数据与跨模型设计收益仍待正式实验。原主题、下列11条计划文献及六层责任保持。

原计划：[conversation_plan_recovery.md](sources/SRC084.md)（SRC084）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[STSep_LIF_Wakhloo_Research_Protocol.md](sources/SRC023.md)（SRC023）；[SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019）

**执行逻辑。** 固定内容、层和时刻，比较pre/post、spike、aux/cache及合法联合观察；检查膜清除后其他载体能否恢复或影响后续输出。

**哪些变体改哪里、为何要控制。** 相同probe函数族与选择预算；联合观察维数/字节单列；辅助状态清除对工作点影响；不用空间均值代替完整图像状态。

**代码入口。** [code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)；[code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)

**本次可执行入口。** 独立3.5包中的carrier_theory.py、capture35.py、probes35.py、timing_controls.py、event_adapter35.py、slow_path_reference.py、run_state_contracts.py、run_carrier_review.py、run_unrequested_extension.py及tests/。旧code_reference入口保留来源；不得将随机未训练的8骨架续跑覆盖写成已训练性能覆盖。

**文献怎样进入，而不是只列名字。**

- **MEM-SYNAPTIC** — Synaptic Theory of Working Memory。调用理由：把低活动/突触载体列入状态边界，做载体分组操作。
- **MEM-WOLFF** — Dynamic hidden states underlying working-memory-guided behavior。调用理由：比较自然读出与受控探测/恢复，检查潜在载体。
- **MEM-PALUCH** — Unattended working memory items are coded by persistent activity in human medial temporal lobe neurons。调用理由：加入未被注意项目的持续活动竞争解释。
- **MEM-REACTIVATION** — Reactivating and reorganizing activity-silent working memory: two distinct mechanisms underlying pinging the brain。调用理由：将再激活与重组织作为两种待判别路径。
- **MEM-DMP** — Algorithm–hardware co-design of neuromorphic networks with dual memory pathways。调用理由：登记快慢路径、低维慢状态和状态/更新成本。
- **REC-LINEARATTN** — How Linear Attention Remembers。调用理由：采用内容特异写入/保留/查询路径、负荷与经过长度分离和混合载体对照。
- **CAND-MONKEY** — Linking working memory maintenance and readout in monkey sensory and prefrontal cortex。调用理由：作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。
- **REC-ACTIVEPASSIVE** — Feature- and space-based interference with functionally active and passive items in working memory。调用理由：不预设active/passive项目遭受相同特征与空间干扰。
- **REC-ACCESS** — Attentional prioritization enhances the accessibility of neural representations during working memory maintenance。调用理由：把注意优先级与可访问性分开于维护强度。
- **REC-ALPHA** — Sustained alpha oscillations serve attentional prioritization in working memory, not maintenance。调用理由：检查活动指标究竟反映优先级还是维护。
- **REC-COMPLEMENTARY** — A complementary learning system for continual episodic memory in large language models。调用理由：对照事件记忆的互补系统与检索载体。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：单载体/联合载体可恢复性与实际读出权限表；归入论证A。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 联合恢复收益只支持观测互补，不自动证明独立槽位或可部署收益。

```bash
python retrieve.py module L3.5 --full
```


## L3.6 跨时间几何与重新编码

**属于什么、要回答什么。** 跨时探针失效是忘记还是编码改变？

**原计划与当前状态。** 2026-10-27完成3.6 v0.2收尾交付，正文SNN_L3_6_Deep_Review_and_Execution_2026-10-26.md，另有交付入口和正式论文/实验规格；完整包SNN_L3_6_Temporal_Geometry_v0_2_2026-10-27.zip。67测试、120未训练分类/重建接口、540循环、54同lag、25相位比较、576事件/阈值/噪声/量化行及两套集成通过。相位比较没有同时正下界。原16×16和条件均值证据保留；正式训练0/435。编号身份：L3原题名由对话摘要恢复；3.1另有完整协议。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[conversation_plan_recovery.md](sources/SRC084.md)（SRC084）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019）

**执行逻辑。** 逐时刻独立解码与train-time×test-time交叉解码并行；分别分析空白保持和持续输入；再连接不同年龄内容的旋转/重编码。

**哪些变体改哪里、为何要控制。** 按独立序列划分而非时间点泄漏；尺度/样本量/位置控制；不同seed坐标对齐；固定probe与可适配probe分开。

**当前代码入口。** 独立3.6包中的temporal_contracts.py、temporal_probes.py、temporal_alignment.py、temporal_theory.py、temporal_clocks.py及主/补充/条件均值runner，见独立稿代码表。以下旧入口保留来源： [code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)；[code_reference/L2/snnbench/analysis.py](code_reference/L2/snnbench/analysis.py)

**文献怎样进入，而不是只列名字。**

- **MEM-MURRAY** — Stable population coding for working memory coexists with heterogeneous neural dynamics in prefrontal cortex。调用理由：并行测稳定群体读出与单元/跨时动态，避免把变化编码误作遗忘。
- **MEM-WOLFF** — Dynamic hidden states underlying working-memory-guided behavior。调用理由：比较自然读出与受控探测/恢复，检查潜在载体。
- **INT-ROTATION** — Rotational dynamics reduce interference between sensory and memory representations。调用理由：将感觉—记忆方向旋转与干扰、行为效用并测。
- **INT-STREAMROT** — Orthogonal-Rotational Dynamics Supports Efficient Encoding and Updating for Streaming Information in Working Memory。调用理由：把流式更新中的旋转/重排作为保持与更新兼顾的候选。
- **GEO-TEMPORALSUPER** — Temporal superposition and feature geometry of RNNs under memory demands。调用理由：把内容维数、年龄需求、稀疏性和网络维数共同纳入，先核线性回忆参照再测非线性系统。
- **REC-RHYTHM** — Rhythmic temporal structure organizes recurrent dynamics to support sequential working memory。调用理由：把序列节律与动态编码组织作为时间结构扩展。
- **MECH-OSCILLATION** — Efficient and robust temporal processing with neural oscillations modulated spiking neural networks。调用理由：作为多时间尺度和振荡机制专项阅读入口。
- **CAND-OPERATIONS** — Working memory operations emerge from dynamic changes in neural subspace geometry。调用理由：作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。
- **CAND-HIGHROT** — Efficient Working Memory Maintenance via High-Dimensional Rotational Dynamics。调用理由：作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。
- **MEM-REACTIVATION** — Reactivating and reorganizing activity-silent working memory: two distinct mechanisms underlying pinging the brain。调用理由：将再激活与重组织作为两种待判别路径。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：时间泛化矩阵与内容轨迹；同时连接论证A和C。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 交叉解码差本身不能证明旋转、隐藏状态或信息缺失；需同刻恢复和几何控制。

```bash
python retrieve.py module L3.6 --full
```


## L3.7 跨层传播与通信子空间

**属于什么、要回答什么。** 某层可恢复的内容是否真正传给下一层？

**原计划与当前状态。** 2026-10-27已完成独立v0.1深入复审与可执行代码，文档SNN_L3_7_Deep_Review_and_Execution_2026-10-27.md，配套SNN_L3_7_Content_Transmission_v0_1_2026-10-27.zip。42测试、12观察、真实边与条件/精度/FIT分析已验收；正式模型结论待训练。编号身份：L3原题名由对话摘要恢复；3.1另有完整协议。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[conversation_plan_recovery.md](sources/SRC084.md)（SRC084）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019）

**执行逻辑。** 比较源表示内容、层间映射、接收方内容与真实输出；先线性通信/噪声归一化，FIT等信息转移仅在样本与假设支持时使用。

**哪些变体改哪里、为何要控制。** 源端最强解码方向不等于有效传输方向；控制共同输入、间接路径和接收噪声；用通路操作及恢复增强证据。

**代码入口。** [code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)

**文献怎样进入，而不是只列名字。**

- **READ-SEMEDO** — Cortical Areas Interact through a Communication Subspace。调用理由：将层间低维通信与各层总方差/本地可解码性分开。
- **READ-TRANSFER** — Measuring Stimulus Information Transfer Between Neural Populations Through the Communication Subspace。调用理由：加入内容方向、接收噪声与有效传输的线性Fisher分析。
- **READ-FIT** — An information-theoretic quantification of the content of communication between brain regions。调用理由：把发送总活动与关于目标内容的新增传输分开。
- **READ-BINISH** — A communication subspace relays context-dependent actions from human prefrontal to motor cortex。调用理由：比较内容不变、规则改变时编码与下游通信各变多少。
- **READ-NULL** — Cortical activity in the null space: permitting preparation without movement。调用理由：区分状态存在与输出敏感方向，定位输出null/potent。
- **READ-FEEDBACK** — Motor Cortical Visuomotor Feedback Activity Is Initially Isolated from Downstream Targets in Output-Null Neural State Space Dimensions。调用理由：检查反馈初期编码是否对当前下游输出暂不可见。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：内容特异传播链与接收方瓶颈；论证C。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 总相关、CCA、Fisher或PID不是同一量；时间先后/统计转移不单独证明因果。

```bash
python retrieve.py module L3.7 --full
```


# L4 研究与实现调用卡

本文件的小节编号是检索路由。独立历史协议的编号身份另在每张卡标明。

## L4.1 观测权限与实际读出

**属于什么、要回答什么。** 系统真正允许下游看到什么？

**原计划与当前状态。** 2026-10-27已落地独立L4.1 v0.1深入稿、参考代码与开发验收；原检索路由保持不变。新增L4整体框架保留五部分，L4.2—L4.5仅形成详细实验规格，未计作已执行。31测试、1728未训练G/P/R/stream案例、288个同trajectory配置折叠验证；一个旧模型15观察/90联合拟合/120主风险单元与8常量基线。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）

**执行逻辑。** 把last/mean/sum/exponential/trajectory与output/pre/post登记为不同观察；原头、同权限重训头、额外内部诊断分组。

**哪些变体改哪里、为何要控制。** 最后状态与全轨迹成本不等；读出缓存属于系统状态；spike与膜访问权限明确。

**新增代码入口。** 独立SNN_L4_1_Access_and_Readout_v0_1_2026-10-27.zip中的access_operators.py、native_head_audit.py、folded_trajectory.py、run_access_review.py及对应控制/验收。原入口保留： [code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)

**文献怎样进入，而不是只列名字。**

- **REP-TEMPORALDESIGN** — Design and Analysis of Temporal Representations in Neural Networks。调用理由：核查编码—激活—解码和时间定义与本框架重叠，作为直接最近邻。
- **BEN-STP** — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects。调用理由：复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **READ-NULL** — Cortical activity in the null space: permitting preparation without movement。调用理由：区分状态存在与输出敏感方向，定位输出null/potent。
- **BEN-RETHINK** — Rethinking the performance comparison between SNNS and ANNS。调用理由：把输入、模型规模、训练和比较权限具体化。
- **REP-TRQR** — MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks。调用理由：保留冗余诊断→门控→读出/成本闭环，明确估计样本轴与新增目标证据。
- **READ-FEEDBACK** — Motor Cortical Visuomotor Feedback Activity Is Initially Isolated from Downstream Targets in Output-Null Neural State Space Dimensions。调用理由：检查反馈初期编码是否对当前下游输出暂不可见。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：权限—性能—资源比较，对应P1。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 额外访问带来的优势不能归为相同部署条件下的细胞优势。

```bash
python retrieve.py module L4.1 --full
```


## L4.2 可恢复性与真正利用的区分

**属于什么、要回答什么。** 新探针能读出，原模型是否用到了？

**原计划与当前状态。** 2026-10-27已完成L4.2独立深入稿、数学/潜空间主线、文献及参考代码开发验收。原七条路由完整保留。25测试、12解析控制；一个旧一次更新模型96主/打乱读者、384风险，另当前头24投影/48读者/192风险与8常量基线。正式多模型规律、内容操作与真实重建未计作完成。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019）

**执行逻辑。** 冻结表示比较原头、同权限重训、线性和受限非线性探针；结合任务标签而非只恢复全部z。

**哪些变体改哪里、为何要控制。** 独立probe划分与样本曲线；正则/预算/调参一致；非线性失败不证明不存在任何信息。

**新增代码入口。** SNN_L4_2_Latent_Usability_v0_1_2026-10-27.zip中的latent_theory42.py、readers42.py、run_reader_ladder.py、run_task_subspaces.py及验收代码；原入口保留： [code_reference/L2/snnbench/analysis.py](code_reference/L2/snnbench/analysis.py)；[code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)

**文献怎样进入，而不是只列名字。**

- **READ-PARK** — Confirmation bias through selective readout of information encoded in human parietal cortex。调用理由：测试目标仍编码但原头低估/偏置的可能，联系历史诱导判断偏差。
- **CAUSE-AMNESIC** — Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals。调用理由：删除可解码属性后测原模型行为，区别可读与利用。
- **GEO-WAKHLOO** — Neural population geometry and optimal coding of tasks with shared latent structure。调用理由：调用作者代码定义、校准Gauss/稀疏码，预测新潜变量任务Hebbian学习曲线。
- **REC-ACCESS** — Attentional prioritization enhances the accessibility of neural representations during working memory maintenance。调用理由：把注意优先级与可访问性分开于维护强度。
- **CAND-MONKEY** — Linking working memory maintenance and readout in monkey sensory and prefrontal cortex。调用理由：作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。
- **REC-FEATUREMATCH** — Behaviorally Irrelevant Feature Matching Increases Neural and Behavioral Working Memory Readout。调用理由：加入不相关特征也可能提高可读性的竞争结果。
- **REC-ALPHA** — Sustained alpha oscillations serve attentional prioritization in working memory, not maintenance。调用理由：检查活动指标究竟反映优先级还是维护。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：存储—可访问—实际输出三项账。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 重训头更好尚不能唯一定位为编码或读出机制，需匹配与操作证据。

```bash
python retrieve.py module L4.2 --full
```


## L4.3 内容、载体与通路干预

**属于什么、要回答什么。** 怎样把几何关联推进到输出依赖？

**原计划与当前状态。** 2026-10-27完成L4.3 v0.1逐节分析、数学规格、作者源码对照和开发执行。原238状态/24实际二值边条件用2048配对；追加112稳定性条件用新1024配对。40测试、10类已知答案控制、240未训练接口、1496场景损失数组验收；仍为旧一次更新模型，新SNN训练0、正式0/435。编号身份：本轮检索路由，不声称旧独立协议已经定稿。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 从自然内容配对到冻结状态操作，再到明确通路/子空间操作；同时测目标改变、非目标损伤、输出变化。

**哪些变体改哪里、为何要控制。** no-op、同维数/同能量随机方向、局部/全core、分布偏移；别把诊断future-oracle称在线机制。

**当前交付。** [SNN_L4_3_Deep_Review_and_Execution_2026-10-27.md](SNN_L4_3_Deep_Review_and_Execution_2026-10-27.md)、[数学规格](SNN_L4_3_Mathematical_Specification_2026-10-27.md)、[文献与代码表](SNN_L4_3_Literature_and_Code_Map_2026-10-27.md)及同名完整L4.3 ZIP。

**代码入口。** bridge43.py、intervention_math43.py、regularized_intervention43.py、runtime_interventions43.py、fit_intervention_bank.py、run_state_interventions.py、run_edge_interventions.py、run_stability_followup.py。保留原L2运行接口为历史基础。

**文献怎样进入，而不是只列名字。**

- **CAUSE-AMNESIC** — Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals。调用理由：删除可解码属性后测原模型行为，区别可读与利用。
- **CAUSE-MEANLEACE** — Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE。调用理由：采用匹配维数/扰动规模的随机方向对照，比较消除方法。
- **CAUSE-RELIABILITY** — How Reliable are Causal Probing Interventions?。调用理由：同时记录completeness、selectivity与原任务变化。
- **CAUSE-LEACE** — LEACE: Perfect linear concept erasure in closed form。调用理由：作为线性概念消除候选，独立验证目标抑制和非目标损伤。
- **CAUSE-ERASURE** — Fundamental Limits of Perfect Concept Erasure。调用理由：在概念消除时显式记录效用权衡和任务变量。
- **READ-DECODESHARE** — DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions。调用理由：借决策阶段定位、预算匹配、patchback与特异性完整证据链。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：完整性、选择性与任务效应证据卡。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 性能下降不自动由目标概念删除造成；有限探针校验不等于覆盖所有潜在概念。

```bash
python retrieve.py module L4.3 --full
```


**本轮关键适配。** 内容命中/非目标损伤/原生风险并列；自然内容、局部供体和拟合操作分开；保留操作—范围交互及事件跨界。无正则LEACE近奇异放大被保留，并追加作者默认与固定协方差收缩对照；追加分析是开发诊断。v/a/syn联合接口与实际二值边区分。原符号分类只为回归输出辅助评分，240接口不算已训练分类/重建内容证据。Canby官方代码未核到；其余选定源码固定commit、限定执行范围。下一节L4.4接收冻结对象，做对应损伤—恢复四组实验。

## L4.4 恢复、补丁与方向验证

**属于什么、要回答什么。** 恢复哪一部分能挽回哪一项损失？

**原计划与当前状态。** 2026-10-27已完成v0.1深入稿、数学说明、作者代码对照与开发验收；正式充分训练与跨模型实验待执行。编号身份：沿原计划逐项扩展，具体8小节见独立稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019）

**执行逻辑。** 在同一权重同一快照上做donor/recipient、清除/patchback及非目标对照；恢复成功与失败均保留。

**哪些变体改哪里、为何要控制。** 供体标签/时刻/统计匹配；同能量伪补丁；干预和恢复会否改变其他载体；不利用测试标签挑补丁。

**代码入口。** 新L4.4包中的recovery_runtime44.py、recovery_math44.py、run_recovery_main.py、snapshot_recovery44.py、edge_reblock44.py；原[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)仍为历史入口。

**文献怎样进入，而不是只列名字。**

- **READ-DECODESHARE** — DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions。调用理由：借决策阶段定位、预算匹配、patchback与特异性完整证据链。
- **REC-LINEARATTN** — How Linear Attention Remembers。调用理由：采用内容特异写入/保留/查询路径、负荷与经过长度分离和混合载体对照。
- **CAUSE-RELIABILITY** — How Reliable are Causal Probing Interventions?。调用理由：同时记录completeness、selectivity与原任务变化。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：损伤—恢复的对应性，支持P2具体路径。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**新增落实。** 4.4.1四组/信息权限；4.4.2逆操作与运行快照上界；4.4.3内容/供体/时间/方向；4.4.4联合状态所有者；4.4.5真实二值边与恢复后再封堵；4.4.6风险—输出交互、事件分层和剂量理论；4.4.7独立场景确认与恢复资格；4.4.8诊断到设计预测。原三条文献逐项覆盖，新增多中介、patching方法/指标和ROME路径近邻。

**实际证据。** 同一旧1更新检查点，两批1024/2048独立场景配对；38情境×10方法×2批=760主四组。补充使用确认池前256，不另算独立复制。72事件政策四组、30联合恢复情境/120子集条件、63二值四组、12实际封堵、72自然事件剂量条件；240未训练接口/1920scope只作实现验收。38测试、14已知控制。9300主损失+378二值损失数组、9232有界区间重算；760主单元各前16场景在batch128重放，载荷/预测/状态差0。

**结果及边界。** 内容probe可以恢复近正常水平，原任务损伤资格仍未成立；两批MSE恢复比例均未定义，不宣称原任务恢复成功。风险交互可来自评分非线性，另报输出Ω。固定事件政策max|Ω|约8.35e-7，自然事件下可达0.662；相同自然事件模式内的仿射输出/二次风险获得条件命题及诊断验收。恢复源膜但封堵未来消息可精确回到损伤输出，解释限当前前馈图。新SNN训练0、正式0/435；外部正常供体是诊断权限，不是免费在线恢复器。恢复也不证明自然系统只通过该一路径工作。

```bash
python retrieve.py module L4.4 --full
```


## L4.5 重建中的历史调用

**属于什么、要回答什么。** 恢复图像细节时到底用了哪帧哪块内容？

**原计划与当前状态。** 合成pilot有记录；真实数据和内容恢复待做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_spatiotemporal_capacity_plan_v1.md](sources/SRC022.md)（SRC022）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 沿C/H/U区域与已对齐来源，做内容配对、载体定位、读出恢复和残差分析；保留完整空间状态。

**哪些变体改哪里、为何要控制。** 观察质量与目标时点；可见mask与标签泄漏；双向恢复不当成因果流；图像类别可当前解决的对照。

**代码入口。** [code_reference/L2/snnbench/l1_contracts.py](code_reference/L2/snnbench/l1_contracts.py)；[code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)

**文献怎样进入，而不是只列名字。**

- **BEN-DNS** — The Intel Neuromorphic DNS Challenge。调用理由：作为连续输出/因果语音恢复与质量—延迟—资源参照。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **MEM-DMP** — Algorithm–hardware co-design of neuromorphic networks with dual memory pathways。调用理由：登记快慢路径、低维慢状态和状态/更新成本。
- **GEN-MUSPIKE** — MuSpike（完整题名以原始版本核验为准）。调用理由：生成/连续任务评价已有先例，重建需具体机制需求。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：区域任务效应与内容读取图，连到分类机制证据。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 清全层后H误差增加不能唯一定位某层某种细节。

```bash
python retrieve.py module L4.5 --full
```


# L5 研究与实现调用卡

本文件的小节编号是检索路由。独立历史协议的编号身份另在每张卡标明。

## L5.1 时长、负荷与能力边界

**属于什么、要回答什么。** 哪些需求超过当前机制和权限能处理的范围？

**原计划与当前状态。** 总体计划已有；该编号为本轮路由。编号身份：本轮检索路由，不声称旧独立协议已经定稿。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 基于L1已认证任务，分别扫描实际空白年龄、内容负荷、目标维数、噪声与观测预算；能力曲线而非单总分。

**哪些变体改哪里、为何要控制。** delay与load分开；增加T的计算成本；独立内容与重复测量区分；报告失败区。

**代码入口。** [code_reference/L2/snnbench/tasks.py](code_reference/L2/snnbench/tasks.py)；[code_reference/L2/snnbench/experiments.py](code_reference/L2/snnbench/experiments.py)

**文献怎样进入，而不是只列名字。**

- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。
- **GEO-TEMPORALSUPER** — Temporal superposition and feature geometry of RNNs under memory demands。调用理由：把内容维数、年龄需求、稀疏性和网络维数共同纳入，先核线性回忆参照再测非线性系统。
- **REC-ASSOCWALL** — Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It。调用理由：保留matched-state、干扰墙与课程改变的竞争解释。
- **MEM-ESS** — Quantifying Memory Utilization with Effective State-Size。调用理由：把分配状态和实际可利用记忆区分，作为线性算子参照。
- **TASK-MQAR** — Zoology / Multi-Query Associative Recall（项目与任务入口）。调用理由：检验原始符号流上的键值绑定、查询和冲突。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：条件化能力画像，对应P3第一步。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 主张是指定条件下的范围，不把某个宽度阈值称普遍容量常数。

```bash
python retrieve.py module L5.1 --full
```


## L5.2 更新、保护与历史干扰

**属于什么、要回答什么。** 能否更新A同时保护未更新B？

**原计划与当前状态。** stale基线已有；多项目协议待做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019）；[SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001）；[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）

**执行逻辑。** 从整体stale过渡到多项目最新值与受保护项目；分开过时目标、无关历史、冲突与纯噪声。

**哪些变体改哪里、为何要控制。** 同一输入不同规则需显式cue或不同训练；iid nuisance与匹配彩色高斯噪声可观测等价；期望风险才支持有害性。

**代码入口。** [code_reference/L2/snnbench/tasks.py](code_reference/L2/snnbench/tasks.py)；[code_reference/L2/snnbench/l1_contracts.py](code_reference/L2/snnbench/l1_contracts.py)；[code_reference/L2/snnbench/experiments.py](code_reference/L2/snnbench/experiments.py)

**文献怎样进入，而不是只列名字。**

- **INT-BARBOSA** — Interplay between persistent activity and activity-silent dynamics in the prefrontal cortex underlies serial biases in working memory。调用理由：把跨trial残留和适应偏差列为历史影响路径。
- **INT-UNABLE** — Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length。调用理由：区分上下文长度与前摄干扰、无关历史负荷。
- **INT-DUALPROCESS** — LLMs Remember First, Forget Last: Dual-Process Interference in Large Language Models。调用理由：比较不同年龄和干扰顺序，避免单一recency解释。
- **REC-SENSORYINT** — Overcoming sensory-memory interference in working memory circuits。调用理由：比较门控、吸引子和动态编码避免感觉—记忆干扰的不同解释。
- **REC-OUTDATED** — Initially encoding attended but outdated information into working memory: behavioral and neural evidence。调用理由：检验旧但被注意内容是否先进入状态，再被更新/压制。
- **REC-PLANUPDATE** — Working memory adaptively recomputes plans amid distraction。调用理由：从保持推进到干扰下计划重算与实际任务调用。
- **INT-PRIMACY** — Emergence of the Primacy Effect in Structured State-Space Models。调用理由：作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。
- **READ-PARK** — Confirmation bias through selective readout of information encoded in human parietal cortex。调用理由：测试目标仍编码但原头低估/偏置的可能，联系历史诱导判断偏差。
- **INT-RECENCY** — Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing。调用理由：纳入recency与over-smoothing两种连续状态瓶颈作强替代解释。
- **REC-ACTIVEPASSIVE** — Feature- and space-based interference with functionally active and passive items in working memory。调用理由：不预设active/passive项目遭受相同特征与空间干扰。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：更新收益、非目标损伤和前摄/后摄干扰图。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 清历史有利不等于模型已学会自主选择性遗忘。

```bash
python retrieve.py module L5.2 --full
```


## L5.3 真实任务、模态与分布变化

**属于什么、要回答什么。** 可控机制关系能否跨真实输入条件成立？

**原计划与当前状态。** 候选数据目录已有；正式外推未做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。完整独立协议优先；原文片段可按下方命令检索。

原计划：[SNN_Dataset_Plan_2026-10-21.md](sources/SRC012.md)（SRC012）；[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）

**执行逻辑。** 按分类/恢复各选有辨别力的数据，先审计时间语义，再留出噪声、更新频率、时长、规则或模态。

**哪些变体改哪里、为何要控制。** 来源分组独立；共享前处理；物理重采样与内部步分开；强连续/RNN/SSM参照。

**代码入口。** [code_reference/L2/snnbench/upstream.py](code_reference/L2/snnbench/upstream.py)；[code_reference/L2/snnbench/physical.py](code_reference/L2/snnbench/physical.py)；[code_reference/L2/snnbench/data.py](code_reference/L2/snnbench/data.py)

**文献怎样进入，而不是只列名字。**

- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **BEN-NSA** — Neuromorphic Sequential Arena: A Benchmark for Neuromorphic Temporal Processing。调用理由：借任务覆盖、时序处理和资源比较的组织逻辑。
- **BEN-HEIDELBERG** — The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks。调用理由：沿用SHD/SSC来源与说话者等划分，审计实际时序需求。
- **BEN-DNS** — The Intel Neuromorphic DNS Challenge。调用理由：作为连续输出/因果语音恢复与质量—延迟—资源参照。
- **BEN-LRA** — Long Range Arena: A Benchmark for Efficient Transformers。调用理由：借多任务长序列与强基线组织，逐项审计任务需求。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：机制关系外推与失败边界。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 多模态数量本身非新颖性；未经适配的数据不进入已验收结果表。

```bash
python retrieve.py module L5.3 --full
```


## L5.4 留出预测与设计效用

**属于什么、要回答什么。** 诊断是否比普通验证搜索多提供价值？

**原计划与当前状态。** 旧E05仅探索；留出确认尚未完成。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Benchmark_Deep_Review_20261014.md](sources/SRC005.md)（SRC005）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）

**执行逻辑。** 事先写条件→机制→修改的规则，用未见任务/架构/条件评估；比较随机、默认、等预算验证选择与强基线。

**哪些变体改哪里、为何要控制。** 计入诊断与试训成本；选择集/审计集隔离；E05若与验证选择相同必须如实报告。

**代码入口。** [code_reference/L2/snnbench/advisor.py](code_reference/L2/snnbench/advisor.py)；[code_reference/L2/snnbench/experiments.py](code_reference/L2/snnbench/experiments.py)

**文献怎样进入，而不是只列名字。**

- **REP-FREQUENCY** — Frequency Matching in Spiking Neural Networks for mmWave Sensing。调用理由：借频率匹配到超参规则的闭环；在留出条件检验规则并对照连续滤波。
- **REP-TRQR** — MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks。调用理由：保留冗余诊断→门控→读出/成本闭环，明确估计样本轴与新增目标证据。
- **REP-TRE** — Temporal Representation Enhancement (TRE): Learning to Forget Dominant Patterns for Enhanced Temporal Spiking Features。调用理由：将类别贡献、互补特征和训练门控与真正新增标签证据分别分析。
- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。
- **BEN-SCRATCH** — Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors。调用理由：把预训练/初始化先验列为公平比较轴，并做等搜索成本参照。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：独立选择收益、校准与成本，兑现P3。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 同一试验上事后解释或改名验证选择，不能称有额外预测价值。

```bash
python retrieve.py module L5.4 --full
```


## L5.5 鲁棒性、资源与系统价值

**属于什么、要回答什么。** 精度、历史能力和代价的取舍是否可信？

**原计划与当前状态。** 逻辑账本已有；硬件部署/正式鲁棒性未做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003）

**执行逻辑。** 分别报告参数、持久状态、活动、训练/推理内存和实测延迟；鲁棒性先校验攻击/测量器有效性。

**哪些变体改哪里、为何要控制。** 浮点spike不按1bit；AC/MAC代理不等实测能耗；系统边界含编码/访存；同精度轻量ANN与总调参预算。

**代码入口。** [code_reference/L2/snnbench/runner.py](code_reference/L2/snnbench/runner.py)；[code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)

**文献怎样进入，而不是只列名字。**

- **BEN-NEUROBENCH** — The neurobench framework for benchmarking neuromorphic computing algorithms and systems。调用理由：继承算法与系统指标边界，分别报逻辑工作量和实测部署。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **EVAL-ROBUST** — Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks。调用理由：把测量/攻击工具有效性放在能力结论之前。
- **EVAL-LIGHTWEIGHT** — From Lightweight CNNs to SpikeNets: Benchmarking Accuracy-Energy Tradeoffs with Pruned …（旧入口缩写）。调用理由：引入同精度轻量ANN和剪枝参照。
- **EVAL-FRAMEWORKS** — A comprehensive multimodal benchmark of neuromorphic training frameworks for spiking neural networks。调用理由：登记框架、内核与管线对速度的影响。
- **BEN-RETHINK** — Rethinking the performance comparison between SNNS and ANNS。调用理由：把输入、模型规模、训练和比较权限具体化。
- **REP-TMI** — Robust Spiking Neural Networks by Temporal Mutual Information。调用理由：核对MI随机变量、KDE轴与估计器、训练和攻击接口；理论用于其实际对象。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：能力—成本前沿和测量有效性卡。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 算子估能、CPU时间和特定硬件实测分别命名；摘要扩展文献不作为方法缺陷证据。

```bash
python retrieve.py module L5.5 --full
```


# L6 研究与实现调用卡

本文件的小节编号是检索路由。独立历史协议的编号身份另在每张卡标明。

## L6.1 时间信用分配与前向状态

**属于什么、要回答什么。** 训练找不到与系统表达不了怎样区分？

**原计划与当前状态。** 基础模式已有；正式学习因素确认待做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_L2_Mechanism_Plan_2026-10-21.md](sources/SRC018.md)（SRC018）

**执行逻辑。** 沿用full/detach/notd，明确full与detach固定前向而改梯度，notd改前向；冻结后统一probe与操作。

**哪些变体改哪里、为何要控制。** 同初始化/数据/预算；跨步梯度范围；推理态检查；不能用梯度范数替代保留内容。

**代码入口。** [code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)；[code_reference/L2/snnbench/runner.py](code_reference/L2/snnbench/runner.py)

**文献怎样进入，而不是只列名字。**

- **BEN-STP** — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects。调用理由：复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。
- **BEN-NSA** — Neuromorphic Sequential Arena: A Benchmark for Neuromorphic Temporal Processing。调用理由：借任务覆盖、时序处理和资源比较的组织逻辑。
- **LEARN-TSNEURON** — Resolving the Timestep Scaling Paradox in Spiking Neural Networks with a Timestep-Scalable Neuron Model。调用理由：把前向历史与指定STE下的梯度界、周期截断和实际训练效果分开。
- **LEARN-NEUROTRAIN** — NeuroTrain（完整题名以原始版本核验为准）。调用理由：参考学习规则接口与公平训练预算。
- **LEARN-BRAINTRACE** — Model-agnostic linear-memory online learning in spiking neural networks。调用理由：区分网络规模与序列长度的内存复杂度。
- **MECH-COMPLEXITY** — Impact of Neuron Models … A Complexity Based Classification Approach（原笔记缩写）。调用理由：参考可控统计过程上的神经元与学习规则比较。
- **LEARN-TIC** — Exploring Temporal Information Dynamics in Spiking Neural Networks。调用理由：先明确Fisher/参数敏感性、样本轴和批次计算，再讨论首步信息。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：前向、学习与最终利用的分离证据。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 训练失败只支持当前算法和预算未学到，不证明任何训练都不可能。

```bash
python retrieve.py module L6.1 --full
```


## L6.2 替代梯度、reset梯度与等价前向

**属于什么、要回答什么。** 相同计算是否因梯度规则而学出不同编码？

**原计划与当前状态。** 历史梯度校准有记录；训练网格待做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[section_2_4.md](sources/SRC066.md)（SRC066）；[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）

**执行逻辑。** 先验证相同权重下的前向等价，再改SG形状/尺度、detach_reset或MPR类等价表达，记录梯度与后训练内容。

**哪些变体改哪里、为何要控制。** 真硬阈值导数、SG和自然有限扰动分开；TSNeurontheorem按规定STE；MPR保留总量不等于顺序。

**代码入口。** [code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)

**文献怎样进入，而不是只列名字。**

- **LEARN-TSNEURON** — Resolving the Timestep Scaling Paradox in Spiking Neural Networks with a Timestep-Scalable Neuron Model。调用理由：把前向历史与指定STE下的梯度界、周期截断和实际训练效果分开。
- **THEORY-INFLOR** — InfLoR-SNN（原笔记简称；完整题名待核）。调用理由：沿MPR前向等价性建立固定计算、改变训练的对照。
- **THEORY-RMP** — RMP-SNN（原笔记简称；Residual Membrane Potential类方法）。调用理由：区分硬/软重置、ANN转换与直接训练。
- **MECH-RESET** — Revisiting Reset Mechanisms（旧计划简称；完整题名待核）。调用理由：沿记忆/发放模块和反馈复位的已有区分，设计具体分支控制。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：梯度形成假设、等价前向控制和形成后的表征。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 更大梯度上界不保证实际梯度更大或训练更稳。

```bash
python retrieve.py module L6.2 --full
```


## L6.3 归一化、编码与初始工作点

**属于什么、要回答什么。** 所谓记忆优势是否由首步、尺度或编码先验造成？

**原计划与当前状态。** 已有审计约束；专门确认待做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[PROTOCOL.md](sources/SRC074.md)（SRC074）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）

**执行逻辑。** 检查bias、norm、首步锚点、编码非线性、beta初始化与状态初值；共同编码器保持输入绑定可表示性。

**哪些变体改哪里、为何要控制。** 只用训练统计；时间维/批维归一化；逐样本Fisher与批均值平方不同；无时间状态不等于没有跨批泄漏。

**代码入口。** [code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)；[code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)；[code_reference/L2/snnbench/runner.py](code_reference/L2/snnbench/runner.py)

**文献怎样进入，而不是只列名字。**

- **LEARN-TIC** — Exploring Temporal Information Dynamics in Spiking Neural Networks。调用理由：先明确Fisher/参数敏感性、样本轴和批次计算，再讨论首步信息。
- **REP-TRQR** — MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks。调用理由：保留冗余诊断→门控→读出/成本闭环，明确估计样本轴与新增目标证据。
- **REP-TMI** — Robust Spiking Neural Networks by Temporal Mutual Information。调用理由：核对MI随机变量、KDE轴与估计器、训练和攻击接口；理论用于其实际对象。
- **BEN-SCRATCH** — Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors。调用理由：把预训练/初始化先验列为公平比较轴，并做等搜索成本参照。
- **REP-HIGHFREQ** — Spiking Neural Networks Need High-Frequency Information。调用理由：分别审时间/空间/事件频率轴和残差、DWC、pool的实际作用。
- **REP-TRE** — Temporal Representation Enhancement (TRE): Learning to Forget Dominant Patterns for Enhanced Temporal Spiking Features。调用理由：将类别贡献、互补特征和训练门控与真正新增标签证据分别分析。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：编码/工作点形成审计与对照。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 输入投影或ReLU是控制选择，需要解释它改变的函数族和预算。

```bash
python retrieve.py module L6.3 --full
```


## L6.4 优化、课程、初始化与公平预算

**属于什么、要回答什么。** 负荷墙或模型差距是否只是训练条件限制？

**原计划与当前状态。** 总体规则已有；专项矩阵待做。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_AI_execution_handoff_v2.md](sources/SRC002.md)（SRC002）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）

**执行逻辑。** 固定每模型可比较的调参/选择预算，单独做课程与初始化扩展；记录成功率、更新数和失败run。

**哪些变体改哪里、为何要控制。** 固定epoch未必同更新/计算；预训练先验；独立种子；早停/测试集身份核验。

**代码入口。** [code_reference/L2/snnbench/runner.py](code_reference/L2/snnbench/runner.py)；[code_reference/L2/snnbench/cli.py](code_reference/L2/snnbench/cli.py)

**文献怎样进入，而不是只列名字。**

- **REC-ASSOCWALL** — Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It。调用理由：保留matched-state、干扰墙与课程改变的竞争解释。
- **BEN-SCRATCH** — Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors。调用理由：把预训练/初始化先验列为公平比较轴，并做等搜索成本参照。
- **LEARN-NEUROTRAIN** — NeuroTrain（完整题名以原始版本核验为准）。调用理由：参考学习规则接口与公平训练预算。
- **LEARN-BULLET** — Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks。调用理由：连续时间事件原生训练是比较轴，不默认所有SNN都time-bin+普通SG。
- **LEARN-BRAINTRACE** — Model-agnostic linear-memory online learning in spiking neural networks。调用理由：区分网络规模与序列长度的内存复杂度。
- **EVAL-FRAMEWORKS** — A comprehensive multimodal benchmark of neuromorphic training frameworks for spiking neural networks。调用理由：登记框架、内核与管线对速度的影响。
- **REC-TITANS** — Titans: Learning to Memorize at Test Time。调用理由：把测试时学习记忆与固定参数递归作为不同状态系统登记。
- **REC-VARIATIONAL** — Variational Linear Attention: Stable Associative Memory for Long-Context Transformers。调用理由：追踪关联存储稳定性与写入规则对负荷的影响。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：学习成功边界与训练成本。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 课程改善不自动解释原系统的存储上界；任何新大矩阵先有具体合同。

```bash
python retrieve.py module L6.4 --full
```


## L6.5 表征形成与可学习参数

**属于什么、要回答什么。** 几何和载体组织怎样随训练形成？

**原计划与当前状态。** 总体计划已有；专项实验未完成。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_AI_execution_handoff_v2.md](sources/SRC002.md)（SRC002）

**执行逻辑。** 保存训练阶段checkpoint，复用L3内容/几何/载体测量，区分plif单参数学习、通道异质性初值与适应状态作用。

**哪些变体改哪里、为何要控制。** 固定统计/审计集；不能把阶段改变和任意不同网络混作同模型轨迹；噪声和probe预算固定。

**代码入口。** [code_reference/L2/snnbench/neurons.py](code_reference/L2/snnbench/neurons.py)、[code_reference/L2/source_original/neurons.py](code_reference/L2/source_original/neurons.py)；[code_reference/L2/snnbench/models.py](code_reference/L2/snnbench/models.py)、[code_reference/L2/source_original/models.py](code_reference/L2/source_original/models.py)；[code_reference/L2/snnbench/analysis.py](code_reference/L2/snnbench/analysis.py)

**文献怎样进入，而不是只列名字。**

- **GEO-WAKHLOO** — Neural population geometry and optimal coding of tasks with shared latent structure。调用理由：调用作者代码定义、校准Gauss/稀疏码，预测新潜变量任务Hebbian学习曲线。
- **MEM-DMP** — Algorithm–hardware co-design of neuromorphic networks with dual memory pathways。调用理由：登记快慢路径、低维慢状态和状态/更新成本。
- **MEM-HIPPO** — HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models。调用理由：用显式历史表示机制作强连续参照，具体比较内容和时间尺度。
- **REC-PALIMPSA** — Learning to Remember, Learn, and Forget in Attention-Based Models。调用理由：参考可学习写入/遗忘与存储规则。
- **REC-ATTRACTOR** — Internal state dynamically gates task-specific attractor dynamics in prefrontal cortex。调用理由：把内部状态对任务特异动力学的门控列为替代机制。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：形成轨迹与学习敏感性；强结果才单独成为贡献。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 训练相关性仍需前向等价/冻结操作来增强解释；当前不预设形成规律。

```bash
python retrieve.py module L6.5 --full
```


# X 研究与实现调用卡

本文件的小节编号是检索路由。独立历史协议的编号身份另在每张卡标明。

## X.PAPER 论文主张、图表与论证顺序

**属于什么、要回答什么。** 六层怎样形成一篇论文而非六个清单？

**原计划与当前状态。** 已有人论证稿；结果依赖主张尚待实验。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_Introduction_Contributions_and_Pilot_v1.md](sources/SRC014.md)（SRC014）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）；[conversation_plan_recovery.md](sources/SRC084.md)（SRC084）

**执行逻辑。** 沿用P1测量条件、P2内容×载体×时间的实际作用、P3留出设计效用；第三层A/B/C嵌入主张，不按指标排列正文。

**哪些变体改哪里、为何要控制。** 每张主图有竞争解释、判别证据和失败分支；方法协议与经验发现分开。

**代码入口。** [code_reference/L2/snnbench/advisor.py](code_reference/L2/snnbench/advisor.py)

**文献怎样进入，而不是只列名字。**

- **BEN-STP** — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects。调用理由：复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **BEN-STEP** — STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking。调用理由：借统一组件替换和可复现平台设置；我们的增量必须落到内容/通路证据。
- **GEO-TEMPORALSUPER** — Temporal superposition and feature geometry of RNNs under memory demands。调用理由：把内容维数、年龄需求、稀疏性和网络维数共同纳入，先核线性回忆参照再测非线性系统。
- **REC-LINEARATTN** — How Linear Attention Remembers。调用理由：采用内容特异写入/保留/查询路径、负荷与经过长度分离和混合载体对照。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：引言、方法、三条证据链、讨论和附录定位。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 六层数量、任务数量和更多指标不单独构成新颖性。

```bash
python retrieve.py module X.PAPER --full
```


## X.STATS 统计、复现与证据等级

**属于什么、要回答什么。** 哪些数字可在论文中作为何种证据？

**原计划与当前状态。** 有旧规则；本轮仅验证索引和来源一致性。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Benchmark_Question_Taxonomy_v1.md](sources/SRC006.md)（SRC006）；[SNN_Benchmark_Research_Argument_Draft_v0_2026-10-20.md](sources/SRC008.md)（SRC008）；[SNN_Cross_Layer_Claims_and_Evidence_v1.md](sources/SRC010.md)（SRC010）；[SNN_L2_Complete_2026-10-21.md](sources/SRC062.md)（SRC062）；[PROTOCOL.md](sources/SRC074.md)（SRC074）

**执行逻辑。** 按场景/序列/种子等正确单位划分和估计，区分代数、工具、pilot、确认和部署；记录配置、哈希、环境与完整失败。

**哪些变体改哪里、为何要控制。** 时间点/像素不自动独立样本；协方差/探针/模型选择集隔离；多比较与预设主测量。

**代码入口。** [code_reference/L2/snnbench/runner.py](code_reference/L2/snnbench/runner.py)；[code_reference/L2/snnbench/cli.py](code_reference/L2/snnbench/cli.py)

**文献怎样进入，而不是只列名字。**

- **BEN-NEUROBENCH** — The neurobench framework for benchmarking neuromorphic computing algorithms and systems。调用理由：继承算法与系统指标边界，分别报逻辑工作量和实测部署。
- **TASK-MAD** — Mechanistic Design and Scaling of Hybrid Architectures。调用理由：优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。
- **EVAL-ROBUST** — Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks。调用理由：把测量/攻击工具有效性放在能力结论之前。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：证据账、版本链和审计记录。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 本轮索引完整性检查不等于重新运行旧神经模型或重新精读全部论文。

```bash
python retrieve.py module X.STATS --full
```


## X.COMP 最近邻、原文版本与贡献边界

**属于什么、要回答什么。** 现有论文已经做到什么，我们具体新增哪条证据？

**原计划与当前状态。** 最近邻审计已有；终稿差异持续待核。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007）；[SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003）；[SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020）

**执行逻辑。** 每节先列已有覆盖、借用方法、尚待验证增量和不能声称的首次；上传稿、arXiv、会议/期刊终版分别建版本。

**哪些变体改哪里、为何要控制。** STP≠NSA；Soul上传题名≠公开题名；预印本后发表不重复算工作；访问失败不等于没有对照。

**代码入口。** [code_reference/L2/snnbench/upstream.py](code_reference/L2/snnbench/upstream.py)

**文献怎样进入，而不是只列名字。**

- **BEN-STP** — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects。调用理由：复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。
- **BEN-NSA** — Neuromorphic Sequential Arena: A Benchmark for Neuromorphic Temporal Processing。调用理由：借任务覆盖、时序处理和资源比较的组织逻辑。
- **BEN-SOUL** — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices。调用理由：保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。
- **BEN-STEP** — STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking。调用理由：借统一组件替换和可复现平台设置；我们的增量必须落到内容/通路证据。
- **REP-TEMPORALDESIGN** — Design and Analysis of Temporal Representations in Neural Networks。调用理由：核查编码—激活—解码和时间定义与本框架重叠，作为直接最近邻。
- **BEN-RETHINK** — Rethinking the performance comparison between SNNS and ANNS。调用理由：把输入、模型规模、训练和比较权限具体化。
- **GEN-MUSPIKE** — MuSpike（完整题名以原始版本核验为准）。调用理由：生成/连续任务评价已有先例，重建需具体机制需求。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：逐节相近/差异卡及版本冲突待核队列。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 旧阅读等级是导入记录，不自动升级成本轮全文或代码独立核验。

```bash
python retrieve.py module X.COMP --full
```


## X.CODE 代码入口与实现交接

**属于什么、要回答什么。** 后续实现如何直接找到对应计划和证据？

**原计划与当前状态。** 本轮交付索引；不新增SNN训练。编号身份：本轮检索路由，不声称旧独立协议已经定稿。本节继承下列总计划和专项记录；关键词路由只是入口，不宣称所有片段已人工确认。

原计划：[SNN_Current_Implementation_and_Gaps.md](sources/SRC011.md)（SRC011）；[SNN_Benchmark_Deep_Review_20261014.md](sources/SRC005.md)（SRC005）；[SNN_AI_execution_handoff_v2.md](sources/SRC002.md)（SRC002）；[historySNN_expert_audit_20261012.md](sources/SRC024.md)（SRC024）；[SNN_round1_independent_review_20260919.md](sources/SRC021.md)（SRC021）

**执行逻辑。** 按模块先检索计划片段、再文献卡、再当前代码与校准；保留原始快照和只读代码参考，不静默覆盖。

**哪些变体改哪里、为何要控制。** WIP与可执行参考分开；historySNN关联分支与本internal_bench不合并结果；索引只能检索不能批准新实验。

**代码入口。** [code_reference/L2/snnbench/l2_runtime.py](code_reference/L2/snnbench/l2_runtime.py)；[code_reference/L3_1/factor_data.py](code_reference/L3_1/factor_data.py)；[code_reference/L3_1/paired_geometry.py](code_reference/L3_1/paired_geometry.py)

**文献怎样进入，而不是只列名字。**

- **BEN-NEUROBENCH** — The neurobench framework for benchmarking neuromorphic computing algorithms and systems。调用理由：继承算法与系统指标边界，分别报逻辑工作量和实测部署。

**与现有工作的相近和候选增量。** 上述文献的方法、任务或问题为本节起点；本节希望交付：离线retrieve.py、来源哈希、覆盖表和当前代码树。 只有其在当前系统、合法观察和完整对照下获得验证，才成为我们的实质增量。

**停止过度解释的边界。** 可找到源码不等于已运行、测试通过或复现原论文。

```bash
python retrieve.py module X.CODE --full
```



# 按论文逻辑组织的重点文献卡


# 文献按论文论证位置组织

每个条目只在一个主组展开，跨层调用保留在module_ids；主题不同不代表文献只能用于一层。下列顺序用于建立论文论证，原阅读卡在retrieve.py中完整保留。

## 一、直接benchmark与最近邻：贡献边界先从这里建立


### BEN-STP — Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；上传/已审arXiv v1；旧笔记记录2026期刊线索，终稿差异未核 |
| 优先级/对应模块 | P0；L1.1、L1.2、L2.1、L4.1、L6.1、X.COMP、L1.C、X.PAPER |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://arxiv.org/abs/2502.09449 |
| 做到了什么（沿原笔记） | 比较前向时间状态、跨时间梯度与NoTD等设置，检查SNN时间处理和学习条件；已有任务套件与配置。 |
| 怎样用于研究 | 复用任务和full/detach/notd问题；先审观察—目标合同，再解释机制。 |
| 不可越过的边界 | 不能称首次切时间梯度；不能把2025上传稿当2026终稿。 |
| 本轮核查范围 | 本轮回收两份上传17页v1原稿并视觉核对一份首页；字节不同而本轮文字抽取SHA相同；不推定期刊终稿已取得 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-STP --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-NSA — Neuromorphic Sequential Arena: A Benchmark for Neuromorphic Temporal Processing

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；IJCAI 2025（沿用旧核验） |
| 优先级/对应模块 | P0；L1.1、L1.G、L5.3、L6.1、X.COMP |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://www.ijcai.org/proceedings/2025/544 |
| 做到了什么（沿原笔记） | 围绕真实序列任务建立神经形态时序处理基准，并组织性能与资源比较。 |
| 怎样用于研究 | 借任务覆盖、时序处理和资源比较的组织逻辑。 |
| 不可越过的边界 | 与STP是相关独立工作；任务成绩不唯一识别内部机制。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-NSA --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-SOUL — A modality-aware benchmark reveals when spiking neural networks benefit edge sensing / Benchmarking spiking neural networks across sensing modalities on edge devices

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；用户24页上传稿与公开arXiv分开登记 |
| 优先级/对应模块 | P0；L1.C、L1.E、L1.F、L2.2、L2.6、L4.1、L5.3、L5.5、X.COMP、L4.5、X.PAPER |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://arxiv.org/abs/2609.00026 |
| 做到了什么（沿原笔记） | 用户上传稿覆盖不同传感模态、编码、神经元和骨架，并有频谱/特征、重建及边缘部署分析。具体版本以上传原稿为准。 |
| 怎样用于研究 | 保留多模态、编码/单元/骨架、频谱/特征与重建覆盖；检查实际物理与模拟两时间轴。 |
| 不可越过的边界 | 不能说原文没有机制解释、重建或部署；上传和公开题名版本不假定完全一致。 |
| 本轮核查范围 | 本轮回收上传24页PDF、全文文字抽取并视觉核对首页题名/作者；具体实验分析沿旧审计，未假定公开版本相同 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-SOUL --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-NEUROBENCH — The neurobench framework for benchmarking neuromorphic computing algorithms and systems

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；Nature Communications（沿用旧核验） |
| 优先级/对应模块 | P0；L2.1、L2.7、L5.5、X.STATS、X.CODE |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://www.nature.com/articles/s41467-025-56739-4 |
| 做到了什么（沿原笔记） | 给出神经形态算法和系统两个层面的基准框架与指标，区分模型工作量和系统实现。 |
| 怎样用于研究 | 继承算法与系统指标边界，分别报逻辑工作量和实测部署。 |
| 不可越过的边界 | 不能把代理估能或浮点二值数值直接当硬件能耗/1bit存储。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-NEUROBENCH --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-STEP — STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；NeurIPS 2025 Datasets and Benchmarks |
| 优先级/对应模块 | P1；L2.6、X.COMP、X.PAPER |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://papers.nips.cc/paper_files/paper/2025/file/e9fece10c0f4ffdcf015ab676f370fbd-Paper-Datasets_and_Benchmarks_Track.pdf |
| 做到了什么（沿原笔记） | 用统一平台和组件替换组织spiking Transformer评价与可复现比较。 |
| 怎样用于研究 | 借统一组件替换和可复现平台设置；我们的增量必须落到内容/通路证据。 |
| 不可越过的边界 | 统一平台与组件诊断已有，不能作为独占新颖性。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-STEP --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REP-TEMPORALDESIGN — Design and Analysis of Temporal Representations in Neural Networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；arXiv 2610.19730；2026-10-14预印本 |
| 优先级/对应模块 | P0；L1.E、L4.1、X.COMP |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://arxiv.org/abs/2610.19730 |
| 做到了什么（沿原笔记） | 直接讨论神经网络时间表示的编码、激活和解码，与我们的时间与观察合同形成近邻。 |
| 怎样用于研究 | 核查编码—激活—解码和时间定义与本框架重叠，作为直接最近邻。 |
| 不可越过的边界 | 仅有六层/写入保留读出框架不构成未被占据的新颖性。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REP-TEMPORALDESIGN --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MECH-STSEP — Unleashing Temporal Capacity of Spiking Neural Networks through Spatiotemporal Separation

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；上传15页稿/arXiv 2512.05472v1 |
| 优先级/对应模块 | P0；L2.4、L2.6、L3.1、L3.2、L3.3 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://arxiv.org/abs/2512.05472 |
| 做到了什么（沿原笔记） | 提出空间/时间计算分离，展示去状态及分支操作可能改善表现，并用资源竞争解释相关现象。 |
| 怎样用于研究 | 沿去状态与时空分支现象，检验工作点、读出、路径替代和内容竞争解释。 |
| 不可越过的边界 | 去状态改善不是固定容量竞争的唯一证明；实现参数与公式须单独核对。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MECH-STSEP --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-LINEARATTN — How Linear Attention Remembers

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026-09-27 arXiv v1；作者主页under review，不赋予oral/spotlight身份 |
| 优先级/对应模块 | P0；L1.D、L2.1、L2.6、L3.2、L3.5、L4.4、X.PAPER |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://arxiv.org/abs/2609.33093 |
| 做到了什么（沿原笔记） | **核心方法与结果。** 在预训练GLA/Gated DeltaNet中，分别干预事实写入、查询前递归状态、查询时读取通路；以配对donor/recipient改变目标内容，使用开发集选择和独立确认、随机范数对照，并检测非目标事实受损。多个事实可以共存且被选择性访问，但不等于独立存储槽；负荷与后续写入重叠会影响回忆和选择性编辑。 |
| 怎样用于研究 | 采用内容特异写入/保留/查询路径、负荷与经过长度分离和混合载体对照。 |
| 不可越过的边界 | 高度直接最近邻；框架不新，SNN事件与完整系统证据须具体兑现。 |
| 本轮核查范围 | L4.4本轮进一步阅读原HTML方法及附录中的写入/保留/读取、独立确认、非目标和联合载体；既有用户上传21页PDF记录保留。官方复现包具体公开URL未解决，未跑GLA/GDN，不宣称全文每式或作者整篇实证复现 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-LINEARATTN --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-RETHINK — Rethinking the performance comparison between SNNS and ANNS

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2020；Neural Networks 121（沿用旧核验） |
| 优先级/对应模块 | P1；L2.2、L4.1、L5.5、X.COMP |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://doi.org/10.1016/j.neunet.2019.09.005 |
| 做到了什么（沿原笔记） | 讨论SNN与ANN性能比较的条件与公平性，要求把计算、训练和资源放在具体设置中比较。 |
| 怎样用于研究 | 把输入、模型规模、训练和比较权限具体化。 |
| 不可越过的边界 | 不是一般证明SNN或ANN永远更强。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-RETHINK --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-DNS — The Intel Neuromorphic DNS Challenge

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2023；原始挑战论文 |
| 优先级/对应模块 | P1；L1.F、L1.G、L4.5、L5.3 |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://arxiv.org/abs/2303.09503 |
| 做到了什么（沿原笔记） | 把实时语音降噪作为连续输出的神经形态挑战，联合考虑质量、运行与资源。 |
| 怎样用于研究 | 作为连续输出/因果语音恢复与质量—延迟—资源参照。 |
| 不可越过的边界 | 挑战提出时的计划不等于后来全部部署已经完成。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-DNS --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-HEIDELBERG — The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2022；TNNLS 2022卷期；online 2020 |
| 优先级/对应模块 | P1；L1.G、L5.3 |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/ |
| 做到了什么（沿原笔记） | 建立面向SNN系统评价的SHD/SSC等脉冲语音数据和基线，并提供真实时序测量。 |
| 怎样用于研究 | 沿用SHD/SSC来源与说话者等划分，审计实际时序需求。 |
| 不可越过的边界 | 不能由一个SNN/RNN排名推一般时序能力优劣。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-HEIDELBERG --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-LRA — Long Range Arena: A Benchmark for Efficient Transformers

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2021；ICLR 2021（原始版本） |
| 优先级/对应模块 | P1；L1.1、L1.G、L5.3 |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://openreview.net/forum?id=qVyeW-grC2k |
| 做到了什么（沿原笔记） | 以多种长序列任务比较高效Transformer等模型，形成任务套件、基线与资源评价。 |
| 怎样用于研究 | 借多任务长序列与强基线组织，逐项审计任务需求。 |
| 不可越过的边界 | 序列长不自动等于必须长期记忆或顺序；平均分口径需锁版本。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-LRA --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### GEN-MUSPIKE — MuSpike（完整题名以原始版本核验为准）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；2025-08 arXiv预印本 |
| 优先级/对应模块 | P1；L1.F、L4.5、X.COMP |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://arxiv.org/abs/2508.19251 |
| 做到了什么（沿原笔记） | 在SNN生成任务上提供benchmark与多维评价；旧记录为方法和评价部分的局部阅读。 |
| 怎样用于研究 | 生成/连续任务评价已有先例，重建需具体机制需求。 |
| 不可越过的边界 | 生成覆盖不是本项目未被占据的新贡献；旧记录仅方法/评价部分。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper GEN-MUSPIKE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MECH-COMPLEXITY — Impact of Neuron Models … A Complexity Based Classification Approach（原笔记缩写）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；2025 arXiv预印本；完整题名待核 |
| 优先级/对应模块 | P1；L1.C、L2.4、L6.1 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://arxiv.org/abs/2509.06970 |
| 做到了什么（沿原笔记） | 在可控统计过程分类中比较神经元和学习规则，作为内部单元benchmark近邻。 |
| 怎样用于研究 | 参考可控统计过程上的神经元与学习规则比较。 |
| 不可越过的边界 | 不要把内部单元比较和合成任务本身说成首次。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MECH-COMPLEXITY --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

## 二、任务与时间信息：先认证我们到底测什么


### TASK-MAD — Mechanistic Design and Scaling of Hybrid Architectures

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2024；MAD原项目；身份沿旧计划，正文审计待补 |
| 优先级/对应模块 | P0；L1.1、L1.2、L1.G、L3.2、L5.4、L1.C、L1.F、L5.1、X.STATS |
| 原笔记来源 | [SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016） |
| 原始入口 | https://github.com/athms/mad-lab |
| 做到了什么（沿原笔记） | 建立复制、回忆、压缩等机制任务和混合架构设计/扩展比较流程。这里只把任务/规范作为待适配来源。 |
| 怎样用于研究 | 优先复用复制/回忆/选择性任务和等状态/参数、搜索规范。 |
| 不可越过的边界 | 任务代码存在不等于本项目已适配或严格复现。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper TASK-MAD --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### TASK-DEEPSETS — Deep Sets

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2017；原始论文；用于无序学习参照 |
| 优先级/对应模块 | P1；L1.2 |
| 原笔记来源 | [SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016） |
| 原始入口 | https://arxiv.org/abs/1703.06114 |
| 做到了什么（沿原笔记） | 给出适用于集合的置换不变建模思路：逐项变换后汇总，能成为强于原始均值的无序参照。 |
| 怎样用于研究 | 采用逐项非线性编码后置换不变汇总，而不是只比较原均值。 |
| 不可越过的边界 | 强学习参照仍不是最优风险的形式证明；项的定义必须合法。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper TASK-DEEPSETS --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### TASK-MQAR — Zoology / Multi-Query Associative Recall（项目与任务入口）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 任务条目/待核；任务/代码条目；不作为已核验完整论文题名 |
| 优先级/对应模块 | P1；L1.2、L1.G、L5.1 |
| 原笔记来源 | [SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016） |
| 原始入口 | https://github.com/HazyResearch/zoology |
| 做到了什么（沿原笔记） | 提供多查询关联回忆任务与生成/实验入口，用符号序列测键值绑定和查询能力；本卡是任务入口而非完整论文身份。 |
| 怎样用于研究 | 检验原始符号流上的键值绑定、查询和冲突。 |
| 不可越过的边界 | 先绑定的记录输入可能预先解决任务，需与原符号条件分开。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper TASK-MQAR --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### TASK-NEUROGYM — NeuroGym IntervalDiscrimination（任务/源码入口）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 任务条目/待核；任务条目 |
| 优先级/对应模块 | P1；L1.2、L1.G |
| 原笔记来源 | [SNN_L1_Completion_2026-10-21.md](sources/SRC016.md)（SRC016） |
| 原始入口 | https://neurogym.github.io/_modules/neurogym/envs/intervaldiscrimination.html |
| 做到了什么（沿原笔记） | 原任务用刺激持续时长比较等认知范式产生序列和监督；不是默认的两离散事件间隔任务。 |
| 怎样用于研究 | 复用刺激持续时长比较，登记通道、时点与离散化。 |
| 不可越过的边界 | 刺激持续时长不改名成两事件间隔。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper TASK-NEUROGYM --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REP-RATE — Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；arXiv 2507.16043v3；旧稿未认定正式发表 |
| 优先级/对应模块 | P1；L1.2、L2.4 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://arxiv.org/abs/2507.16043 |
| 做到了什么（沿原笔记） | 通过计数/时机等控制检查代理梯度训练的SNN是否学习和使用精确时序信息。 |
| 怎样用于研究 | 采用计数保持、时机改变等对照区分rate与timing。 |
| 不可越过的边界 | 时序任务成绩不自动证明使用了精确spike timing。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REP-RATE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

## 三、前向算子与可解析参照：写入、滤波、事件、复位


### REP-FREQUENCY — Frequency Matching in Spiking Neural Networks for mmWave Sensing

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；ICML 2026；已读版本另记 |
| 优先级/对应模块 | P0；L2.2、L2.3、L5.4 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://proceedings.mlr.press/v306/liao26e.html |
| 做到了什么（沿原笔记） | 将毫米波数据的频率特点与SNN时间常数/滤波联系，并提出匹配和设计规则。 |
| 怎样用于研究 | 借频率匹配到超参规则的闭环；在留出条件检验规则并对照连续滤波。 |
| 不可越过的边界 | 6.22为平均相对百分比而非百分点；有限窗、噪声、读出约束要明确。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REP-FREQUENCY --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REP-HIGHFREQ — Spiking Neural Networks Need High-Frequency Information

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；NeurIPS 2025；Max-Former源码审计见旧路线图 |
| 优先级/对应模块 | P1；L2.3、L2.6、L6.3、L2.2 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://papers.nips.cc/paper_files/paper/2025/file/956834836f36dd07df7064ff42ca69f2-Paper-Conference.pdf |
| 做到了什么（沿原笔记） | 从频率角度分析SNN并设计Max-Former；原记录已核其空间/时间轴、残差和相关模块。 |
| 怎样用于研究 | 分别审时间/空间/事件频率轴和残差、DWC、pool的实际作用。 |
| 不可越过的边界 | 不能由某低通因子推出全网络必低通；原文已有残差对照。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REP-HIGHFREQ --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### THEORY-RMP — RMP-SNN（原笔记简称；Residual Membrane Potential类方法）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2020；arXiv 2003.01811v2；完整题名待核 |
| 优先级/对应模块 | P1；L2.4、L6.2 |
| 原笔记来源 | [SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020） |
| 原始入口 | https://arxiv.org/abs/2003.01811 |
| 做到了什么（沿原笔记） | 以保留残余膜电位的转换/重置机制减少转换损失；具体完整题名和实际实现按原论文版本核对。 |
| 怎样用于研究 | 区分硬/软重置、ANN转换与直接训练。 |
| 不可越过的边界 | 不同soft reset减法量不同；转换精度解释不自动等于时间顺序保留。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper THEORY-RMP --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### THEORY-INFLOR — InfLoR-SNN（原笔记简称；完整题名待核）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2023；arXiv 2307.04356v2 |
| 优先级/对应模块 | P1；L2.4、L6.2 |
| 原笔记来源 | [SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020） |
| 原始入口 | https://arxiv.org/abs/2307.04356 |
| 做到了什么（沿原笔记） | 围绕脉冲网络的信息损失和膜电位等机制提出方法；旧记录独立核对MPR前向等价与训练作用，完整书目信息待补。 |
| 怎样用于研究 | 沿MPR前向等价性建立固定计算、改变训练的对照。 |
| 不可越过的边界 | 前向保留总量不保证保持每步输入次序；梯度规则须逐式核验。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper THEORY-INFLOR --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MECH-RESET — Revisiting Reset Mechanisms（旧计划简称；完整题名待核）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；旧L2计划方法入口，完整身份待核 |
| 优先级/对应模块 | P2；L2.4、L6.2 |
| 原笔记来源 | [section_2_4.md](sources/SRC066.md)（SRC066） |
| 原始入口 | 准确入口尚待核验；不猜链接 |
| 做到了什么（沿原笔记） | 旧L2笔记记载已有工作区分记忆/发放模块并讨论复位；本轮不凭简称补造完整方法或身份。 |
| 怎样用于研究 | 沿记忆/发放模块和反馈复位的已有区分，设计具体分支控制。 |
| 不可越过的边界 | 不能将已经存在的概念分离说成首次；未核版本前不写正式引用。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MECH-RESET --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MEM-DMP — Algorithm–hardware co-design of neuromorphic networks with dual memory pathways

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.5、L3.5、L4.5、L6.5 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://www.nature.com/articles/s42256-026-01255-3 |
| 做到了什么（沿原笔记） | **问题与方法。** DMP-SNN在快速脉冲通路之外，为每层引入低维慢状态，压缩近期活动并反馈为输入电流；联合考察时序任务、记忆大小、更新间隔、梯度传播和硬件数据流。 |
| 怎样用于研究 | 登记快慢路径、低维慢状态和状态/更新成本。 |
| 不可越过的边界 | 双记忆已有；本ref单元和作者原模型不可混同，终稿/预印本差异另核。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MEM-DMP --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### THEORY-SSMFREQ — Tuning Frequency Bias of State Space Models（旧计划题名）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；ICLR 2025（沿L2旧记录，需正式入口再核） |
| 优先级/对应模块 | P2；L2.3 |
| 原笔记来源 | [section_2_3.md](sources/SRC065.md)（SRC065） |
| 原始入口 | 准确入口尚待核验；不猜链接 |
| 做到了什么（沿原笔记） | 旧L2笔记将其列为SSM频率偏置与调节参照；准确出版身份和原文细节仍需专项核验。 |
| 怎样用于研究 | 用连续状态模型频率偏置作为泄漏机制的竞争参照。 |
| 不可越过的边界 | 待核正文与准确入口；不由题名推定所有SSM结论。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper THEORY-SSMFREQ --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MECH-OSCILLATION — Efficient and robust temporal processing with neural oscillations modulated spiking neural networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；Nature Communications（旧扩展条目） |
| 优先级/对应模块 | P2；L2.5、L3.6 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://www.nature.com/articles/s41467-025-63771-x |
| 做到了什么（沿原笔记） | 旧扩展记录介绍振荡调制SNN的时序处理和鲁棒性；尚不作完整方法批评。 |
| 怎样用于研究 | 作为多时间尺度和振荡机制专项阅读入口。 |
| 不可越过的边界 | 摘要级条目不用于宣布其缺少某项实验。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MECH-OSCILLATION --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

## 四、内容容量、叠加与几何泛化：第三层论证B


### THEORY-IPC — Information Processing Capacity of Dynamical Systems

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2012；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P0；L3.2、L3.4 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://www.nature.com/articles/srep00514 |
| 做到了什么（沿原笔记） | **问题与方法。** 用输入历史的正交目标函数及线性读出衡量动力系统能完成哪些历史变换；不仅问能否复制过去，也问能否计算过去输入的非线性组合。 |
| 怎样用于研究 | 按正交目标族、线性读出和总体投影条件使用容量总和。 |
| 不可越过的边界 | 相关标签、重复lag或有限拟合R²不能无条件相加套上界。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper THEORY-IPC --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### THEORY-REFINED — A refined information processing capacity metric allows an in-depth analysis of memory and nonlinearity trade-offs in neurocomputational systems

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2023；Scientific Reports；旧正文/补充阅读记录 |
| 优先级/对应模块 | P1；L3.2、L3.4 |
| 原笔记来源 | [SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020） |
| 原始入口 | https://www.nature.com/articles/s41598-023-37604-0 |
| 做到了什么（沿原笔记） | 修订信息处理容量测量，以更细致分析编码、噪声、记忆与非线性折衷。 |
| 怎样用于研究 | 在SNN编码、噪声和观测条件下校准容量计算。 |
| 不可越过的边界 | 噪声扣除和编码效应依赖条件，不把修正指标当所有数据上的绝对容量。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper THEORY-REFINED --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### THEORY-INUBUSHI — Reservoir Computing Beyond Memory-Nonlinearity Trade-off

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2017；Scientific Reports；旧正文/补充阅读记录 |
| 优先级/对应模块 | P1；L2.3、L3.4 |
| 原笔记来源 | [SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020） |
| 原始入口 | https://www.nature.com/articles/s41598-017-10257-6 |
| 做到了什么（沿原笔记） | 研究reservoir的记忆与非线性关系及混合动力学，结合理论与数值分析；旧稿另核局部扰动的解释边界。 |
| 怎样用于研究 | 用混合动力学和局部扰动对照解释记忆/非线性权衡。 |
| 不可越过的边界 | 局部扰动增长与完整历史信息不是同一对象。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper THEORY-INUBUSHI --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MEM-ESS — Quantifying Memory Utilization with Effective State-Size

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.3、L3.2、L5.1 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://proceedings.mlr.press/v267/parnichkun25a.html |
| 做到了什么（沿原笔记） | **问题与方法。** 从已实现的输入—输出算子出发，考察跨某个序列切点的“过去输入→未来输出”块，利用秩及有效秩形式衡量Effective State-Size，而非直接把缓存长度或名义状态维度当作利用量。 |
| 怎样用于研究 | 把分配状态和实际可利用记忆区分，作为线性算子参照。 |
| 不可越过的边界 | ESS不是所有非线性硬重置网络的即插即用精确容量。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MEM-ESS --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MEM-HIPPO — HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.3、L2.5、L6.5、L1.E |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://proceedings.mlr.press/v306/goffinet26a.html |
| 做到了什么（沿原笔记） | **问题与方法。** 将压缩历史表示做成可解释机制。Volterra HiPPO用多项式读出区分历史的一阶、二阶等相互作用；Salience HiPPO通过输入与状态决定的速率调整历史权重，并以时间重参数化解释选择性记忆。 |
| 怎样用于研究 | 用显式历史表示机制作强连续参照，具体比较内容和时间尺度。 |
| 不可越过的边界 | 不能只与无状态MLP比较就声称独有记忆优势。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MEM-HIPPO --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### GEO-TEMPORALSUPER — Temporal superposition and feature geometry of RNNs under memory demands

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；ICLR 2026；本轮核对正式摘要与作者代码入口 |
| 优先级/对应模块 | P0；L3.2、L3.3、L3.6、L3.1、L5.1、X.PAPER |
| 原笔记来源 | [conversation_plan_recovery.md](sources/SRC084.md)（SRC084） |
| 原始入口 | https://proceedings.iclr.cc/paper_files/paper/2026/hash/bfcb583d225b1db8d3ca2331f6785774-Abstract-Conference.html |
| 做到了什么（沿原笔记） | 研究线性递归延迟序列回忆中数据稀疏性、记忆需求和网络维数如何改变编码几何，并将洞见扩展到非线性RNN。此概述来自本轮正式摘要核对。 |
| 怎样用于研究 | 把内容维数、年龄需求、稀疏性和网络维数共同纳入，先核线性回忆参照再测非线性系统。 |
| 不可越过的边界 | 当前只新核身份/摘要/代码目录；尚未逐式审计或运行其notebook，不直接照搬损失分解。 |
| 本轮核查范围 | 本轮核对ICLR正式摘要、作者、作者仓库README和notebook目录；未逐式读PDF或运行代码 |

**读取原阅读卡与原计划：** `python retrieve.py paper GEO-TEMPORALSUPER --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### GEO-WAKHLOO — Neural population geometry and optimal coding of tasks with shared latent structure

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；Nature Neuroscience 29:682–692；2026-02-04 |
| 优先级/对应模块 | P0；L3.3、L3.4、L6.5、L1.C、L3.1、L4.2 |
| 原笔记来源 | [STSep_LIF_Wakhloo_Research_Protocol.md](sources/SRC023.md)（SRC023） |
| 原始入口 | https://www.nature.com/articles/s41593-025-02183-y |
| 做到了什么（沿原笔记） | 在共享潜变量随机分类任务、监督Hebbian读出和Gaussian近似下，将泛化误差联系到四个群体几何统计量；另有人工/生物数据检验。 |
| 怎样用于研究 | 调用作者代码定义、校准Gauss/稀疏码，预测新潜变量任务Hebbian学习曲线。 |
| 不可越过的边界 | Gaussian近似和监督Hebbian读出是公式条件，不是任意probe或原任务头。 |
| 本轮核查范围 | 本轮核对出版社身份、日期、摘要和指定Hebbian/Gaussian方法段；另核上传PDF首页视觉身份、抽取全文文字 |

**读取原阅读卡与原计划：** `python retrieve.py paper GEO-WAKHLOO --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REP-TRQR — MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；NeurIPS 2025；官方代码路径见旧路线图 |
| 优先级/对应模块 | P1；L3.1、L4.1、L5.4、L6.3 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://papers.nips.cc/paper_files/paper/2025/file/dfb89b72be941eeeaeb5ee53bd3e7fc1-Paper-Conference.pdf |
| 做到了什么（沿原笔记） | 用互信息相关统计量刻画跨步冗余，结合概率掩蔽/读出等方法追求性能和活动成本改善。 |
| 怎样用于研究 | 保留冗余诊断→门控→读出/成本闭环，明确估计样本轴与新增目标证据。 |
| 不可越过的边界 | 跨时MI高不等于无新任务信息；原文已有随机mask对照，不要虚构缺口。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REP-TRQR --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REP-TRE — Temporal Representation Enhancement (TRE): Learning to Forget Dominant Patterns for Enhanced Temporal Spiking Features

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；CVPR 2026；上传11页PDF |
| 优先级/对应模块 | P1；L3.1、L3.3、L5.4、L6.3 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Temporal_Representation_Enhancement_TRE_Learning_to_Forget_Dominant_Patterns_for_CVPR_2026_paper.html |
| 做到了什么（沿原笔记） | 通过训练期抑制主导特征鼓励跨步互补，构造类别贡献分数、门控及监督；旧稿已逐式审查其证据范围。 |
| 怎样用于研究 | 将类别贡献、互补特征和训练门控与真正新增标签证据分别分析。 |
| 不可越过的边界 | 贡献分数不等于删除后的置信度变化；须与TET等强训练基线比较。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REP-TRE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REP-TMI — Robust Spiking Neural Networks by Temporal Mutual Information

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；CVPR 2026；上传10页PDF；独立supp缺口保留 |
| 优先级/对应模块 | P1；L3.1、L5.5、L6.3 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Robust_Spiking_Neural_Networks_by_Temporal_Mutual_Information_CVPR_2026_paper.html |
| 做到了什么（沿原笔记） | 把时间互信息/信息瓶颈思想用于SNN鲁棒性正则，包含理论、估计、训练与攻击结果；旧审计记录若干推导和代码缺口。 |
| 怎样用于研究 | 核对MI随机变量、KDE轴与估计器、训练和攻击接口；理论用于其实际对象。 |
| 不可越过的边界 | 确定性不等于可逆；IB泛化信息差不自动等于对抗风险；当前源码缺口见原审计。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REP-TMI --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-ASSOCWALL — Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.2、L5.1、L6.4 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://arxiv.org/abs/2609.16183 |
| 做到了什么（沿原笔记） | **已核对的摘要内容。** 固定状态预算，拆分短因果卷积、递归转移结构和衰减，指出整架构结果可能混入卷积模块差异；同时研究课程训练与关联回忆。 |
| 怎样用于研究 | 保留matched-state、干扰墙与课程改变的竞争解释。 |
| 不可越过的边界 | 负荷失败可能来自学习；不能先验归为固定状态维数绝对上界。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-ASSOCWALL --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-VARIATIONAL — Variational Linear Attention: Stable Associative Memory for Long-Context Transformers

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.2、L6.4 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://arxiv.org/abs/2605.11196 |
| 做到了什么（沿原笔记） | **方法与研究对象。** 将关联记忆写入表述为在线正则化最小二乘，维护自适应惩罚矩阵，研究状态增长、写入稳定性与回忆负荷。 |
| 怎样用于研究 | 追踪关联存储稳定性与写入规则对负荷的影响。 |
| 不可越过的边界 | 预印本，不据摘要定其完整强基线缺口。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-VARIATIONAL --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

## 五、载体与重新编码：第三层论证A


### MEM-SYNAPTIC — Synaptic Theory of Working Memory

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2008；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.5、L3.5 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://doi.org/10.1126/science.1150769 |
| 做到了什么（沿原笔记） | **问题与方法。** 构建具有短时突触可塑性的递归integrate-and-fire网络，以突触易化变量及资源耗竭变量表达刺激留下的动态痕迹；非特异输入可以触发项目选择性的再激活。 |
| 怎样用于研究 | 把低活动/突触载体列入状态边界，做载体分组操作。 |
| 不可越过的边界 | 生物机制是问题来源，不是SNN辅助变量与真实突触一一对应的证明。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MEM-SYNAPTIC --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MEM-MURRAY — Stable population coding for working memory coexists with heterogeneous neural dynamics in prefrontal cortex

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2017；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.1、L3.6 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://johndmurray.org/papers/murray_2017_pnas.pdf |
| 做到了什么（沿原笔记） | **问题与方法。** 分析前额叶群体活动，比较单神经元时间变化与群体层面低维、可稳定读取的记忆编码。 |
| 怎样用于研究 | 并行测稳定群体读出与单元/跨时动态，避免把变化编码误作遗忘。 |
| 不可越过的边界 | 稳定子空间存在不等于所有内容不变或真实下游自动使用。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MEM-MURRAY --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MEM-WOLFF — Dynamic hidden states underlying working-memory-guided behavior

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2017；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.5、L3.6 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5446784/ |
| 做到了什么（沿原笔记） | **问题与方法。** 在工作记忆延迟期加入不携带项目身份的视觉扰动，利用EEG观察扰动响应能否反映被保持的方向信息；区分暂未被注意但仍需保持的项目与被指示遗忘的项目。 |
| 怎样用于研究 | 比较自然读出与受控探测/恢复，检查潜在载体。 |
| 不可越过的边界 | ping可测性不自动等价于原系统自然利用；诊断激励改变了条件。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MEM-WOLFF --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MEM-PALUCH — Unattended working memory items are coded by persistent activity in human medial temporal lobe neurons

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.5 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://www.nature.com/articles/s41562-025-02235-0 |
| 做到了什么（沿原笔记） | **问题与方法。** 人类内侧颞叶单神经元记录，研究注意优先级变化后，未被注意的工作记忆项目是否仍有持续活动编码。 |
| 怎样用于研究 | 加入未被注意项目的持续活动竞争解释。 |
| 不可越过的边界 | 旧记录为摘要/局部，不能由标题排除所有静默机制。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MEM-PALUCH --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### MEM-REACTIVATION — Reactivating and reorganizing activity-silent working memory: two distinct mechanisms underlying pinging the brain

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.5、L3.6 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://academic.oup.com/cercor/article/35/2/bhae494/7942771 |
| 做到了什么（沿原笔记） | **直接相关问题。** 扰动后重新可解码，究竟是旧表征被重新激活，还是编码方式被重组？该论文明确研究这一区分。 |
| 怎样用于研究 | 将再激活与重组织作为两种待判别路径。 |
| 不可越过的边界 | 旧记录B级；需原文方法后才确定SNN对应操作。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper MEM-REACTIVATION --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-ACTIVEPASSIVE — Feature- and space-based interference with functionally active and passive items in working memory

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.5、L5.2 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://research.rug.nl/en/publications/feature-and-space-based-interference-with-functionally-active-and/ |
| 做到了什么（沿原笔记） | **方法与结果。** 通过查询次序或奖励改变项目优先级，以特征或空间匹配的干扰任务影响维持过程。四组实验未发现主动与被动项目对干扰的不同易感性；混合模型参数分析也未支持该差异。 |
| 怎样用于研究 | 不预设active/passive项目遭受相同特征与空间干扰。 |
| 不可越过的边界 | 行为实验的无差异/差异结论不直接等于模型机制。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-ACTIVEPASSIVE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-FEATUREMATCH — Behaviorally Irrelevant Feature Matching Increases Neural and Behavioral Working Memory Readout

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L4.2 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.70020 |
| 做到了什么（沿原笔记） | **方法与结果。** 方向记忆任务中，让脉冲扰动或测试刺激在任务无关的空间频率上匹配/不匹配原项目。匹配扰动提高EEG记忆特异性读取，匹配测试刺激缩短行为反应时间；比较了全局响应和试次变异等替代解释。 |
| 怎样用于研究 | 加入不相关特征也可能提高可读性的竞争结果。 |
| 不可越过的边界 | 不能预设一切非目标内容都有害。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-FEATUREMATCH --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-ACCESS — Attentional prioritization enhances the accessibility of neural representations during working memory maintenance

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L4.2、L3.5 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://doi.org/10.64898/2026.05.04.722761 |
| 做到了什么（沿原笔记） | **为什么直接相关。** 题目明确把维持期的神经表征与可访问性联系起来；应优先核对注意优先级究竟作用于编码、维持还是读取。 |
| 怎样用于研究 | 把注意优先级与可访问性分开于维护强度。 |
| 不可越过的边界 | 预印本候选，旧记录C级，不将标题当证据结论。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-ACCESS --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-ALPHA — Sustained alpha oscillations serve attentional prioritization in working memory, not maintenance

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.5、L4.2 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://doi.org/10.1162/IMAG.a.1199 |
| 做到了什么（沿原笔记） | **问题与可核查结果。** 区分持续alpha活动反映所有存储内容的维持，还是被优先项目的注意选择；报告结果更支持后者。 |
| 怎样用于研究 | 检查活动指标究竟反映优先级还是维护。 |
| 不可越过的边界 | alpha生物机制不直接对应SNN某个状态数值。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-ALPHA --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-ATTRACTOR — Internal state dynamically gates task-specific attractor dynamics in prefrontal cortex

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.5、L6.5 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://doi.org/10.64898/2026.05.20.726585 |
| 做到了什么（沿原笔记） | **摘要范围内的研究。** 小鼠mPFC群体记录与数据约束RNN，研究任务投入状态如何调节记忆维持子空间的吸引子动力学，同时感觉诱发响应相对保留。 |
| 怎样用于研究 | 把内部状态对任务特异动力学的门控列为替代机制。 |
| 不可越过的边界 | 旧记录局部/摘要，尚不据此定量设计SNN参数。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-ATTRACTOR --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-RHYTHM — Rhythmic temporal structure organizes recurrent dynamics to support sequential working memory

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.6 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://doi.org/10.64898/2026.05.20.726720 |
| 做到了什么（沿原笔记） | **问题与方法。** 具有兴奋/抑制约束和短时突触可塑性的RNN，在规则或抖动时间结构下完成序列延迟匹配，分析输入节律与群体动力学的关系。 |
| 怎样用于研究 | 把序列节律与动态编码组织作为时间结构扩展。 |
| 不可越过的边界 | eLife reviewed preprint附评审争议，不能当常规终审定论。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-RHYTHM --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-COMPLEMENTARY — A complementary learning system for continual episodic memory in large language models

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.1、L3.5 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://www.biorxiv.org/content/10.64898/2026.08.24.746712v1.full |
| 做到了什么（沿原笔记） | **机制。** 以稀疏、分离的事件专用低秩适配器快速存储，竞争门控选择检索的痕迹，并通过回放将内容逐步整合到共享基础权重。 |
| 怎样用于研究 | 对照事件记忆的互补系统与检索载体。 |
| 不可越过的边界 | event-specific LoRA存储随事件增长，不可称固定状态等预算基线。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-COMPLEMENTARY --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### CAND-MONKEY — Linking working memory maintenance and readout in monkey sensory and prefrontal cortex

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；bioRxiv；旧C级 |
| 优先级/对应模块 | P2；L3.5、L4.2 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://doi.org/10.64898/2026.03.13.711560 |
| 做到了什么（沿原笔记） | 目前保留连接工作记忆维护与读取的原始记录入口，全文方法/因果证据未齐，不补写发现。 |
| 怎样用于研究 | 作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。 |
| 不可越过的边界 | 候选身份不提供已验证实验发现；不由题名反推证据。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper CAND-MONKEY --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### CAND-OPERATIONS — Working memory operations emerge from dynamic changes in neural subspace geometry

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；bioRxiv；旧C级 |
| 优先级/对应模块 | P2；L3.3、L3.6 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://doi.org/10.64898/2026.08.31.748237 |
| 做到了什么（沿原笔记） | 目前保留动态神经子空间与工作记忆操作的原始记录入口，全文未取得，不补写发现。 |
| 怎样用于研究 | 作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。 |
| 不可越过的边界 | 候选身份不提供已验证实验发现；不由题名反推证据。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper CAND-OPERATIONS --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### CAND-HIGHROT — Efficient Working Memory Maintenance via High-Dimensional Rotational Dynamics

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；eLife reviewed preprint 111445；旧C级 |
| 优先级/对应模块 | P2；L3.3、L3.6 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://elifesciences.org/reviewed-preprints/111445 |
| 做到了什么（沿原笔记） | 目前保留高维旋转维护的eLife reviewed preprint入口，尚未完成方法审计。 |
| 怎样用于研究 | 作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。 |
| 不可越过的边界 | 候选身份不提供已验证实验发现；不由题名反推证据。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper CAND-HIGHROT --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

## 六、传递与真实读出：第三层论证C连接L4


### READ-NULL — Cortical activity in the null space: permitting preparation without movement

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2014；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.7、L4.1 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3955357/ |
| 做到了什么（沿原笔记） | **问题与方法。** 研究运动准备期的皮层活动为何不立刻产生动作，依据输出映射区分output-potent与output-null活动方向。 |
| 怎样用于研究 | 区分状态存在与输出敏感方向，定位输出null/potent。 |
| 不可越过的边界 | 特定读出null不等于对任意未来动态都无影响。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper READ-NULL --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### READ-FEEDBACK — Motor Cortical Visuomotor Feedback Activity Is Initially Isolated from Downstream Targets in Output-Null Neural State Space Dimensions

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2017；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.7、L4.1 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5547570/ |
| 做到了什么（沿原笔记） | **问题与方法。** 借助脑机接口的已知下游映射，研究视觉反馈活动首先落在哪些状态方向，随后如何影响输出。 |
| 怎样用于研究 | 检查反馈初期编码是否对当前下游输出暂不可见。 |
| 不可越过的边界 | 自然记录关联不等于SNN通路干预已完成。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper READ-FEEDBACK --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### READ-SEMEDO — Cortical Areas Interact through a Communication Subspace

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2019；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.7 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://users.ece.cmu.edu/~byronyu/papers/SemedoNeuron2019.pdf |
| 做到了什么（沿原笔记） | **问题与方法。** 对同时记录的V1/V2群体活动做降秩回归，寻找一群体中能够预测另一群体活动的低维方向，并与本区主要变异方向比较。 |
| 怎样用于研究 | 将层间低维通信与各层总方差/本地可解码性分开。 |
| 不可越过的边界 | 通信子空间不自动提供内容特异因果结论。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper READ-SEMEDO --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### READ-TRANSFER — Measuring Stimulus Information Transfer Between Neural Populations Through the Communication Subspace

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.3、L3.7 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12392746/ |
| 做到了什么（沿原笔记） | **问题与方法。** 在线性群体映射与噪声模型下，把关于刺激的线性Fisher信息区分为通信子空间、私有子空间，以及考虑接收方噪声后的impactful information等量。 |
| 怎样用于研究 | 加入内容方向、接收噪声与有效传输的线性Fisher分析。 |
| 不可越过的边界 | Fisher不是MI；欧氏夹角和解释方差不足以决定有效信息传输。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper READ-TRANSFER --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### READ-FIT — An information-theoretic quantification of the content of communication between brain regions

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2023；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.7 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://tobiasdonner.net/wp-content/uploads/2025/01/celottoNeurIPS2023.pdf |
| 做到了什么（沿原笔记） | **问题与方法。** 提出feature-specific information transfer（FIT），使用部分信息分解，寻找接收区当前关于某特征的信息中，与发送区过去共享、又相对接收区过去为新的部分；验证不同记录类型并处理若干混杂。 |
| 怎样用于研究 | 把发送总活动与关于目标内容的新增传输分开。 |
| 不可越过的边界 | 依赖特定PID定义与估计；共同输入/间接路径仍须控制，非通用因果度量。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper READ-FIT --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### READ-PARK — Confirmation bias through selective readout of information encoded in human parietal cortex

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L4.2、L5.2 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://www.nature.com/articles/s41467-025-61010-x |
| 做到了什么（沿原笔记） | **问题与方法。** 人类序列证据判断与MEG；将刺激、神经响应和最终报告联系起来，区分刺激编码信息与和行为共同相关的信息，并比较中途选择及控制条件。 |
| 怎样用于研究 | 测试目标仍编码但原头低估/偏置的可能，联系历史诱导判断偏差。 |
| 不可越过的边界 | 统计内容—行为联系不单独识别具体因果通路。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper READ-PARK --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### READ-BINISH — A communication subspace relays context-dependent actions from human prefrontal to motor cortex

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.7 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://www.nature.com/articles/s41593-026-02290-4 |
| 做到了什么（沿原笔记） | **相关性。** 以人类前额叶—运动皮层记录研究情境依赖动作与通信子空间，提供“同一信息是否随任务规则被下游选择”的近期参照。 |
| 怎样用于研究 | 比较内容不变、规则改变时编码与下游通信各变多少。 |
| 不可越过的边界 | 旧记录B级，不从relays题名推定完整因果干预。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper READ-BINISH --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

## 七、干预和恢复的可信度：P2证据强度


### CAUSE-AMNESIC — Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2021；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L4.2、L4.3 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://aclanthology.org/2021.tacl-1.10/ |
| 做到了什么（沿原笔记） | **问题与方法。** 从表征中消除某属性的线性可解码信息，再观察原模型行为怎样改变，而不是只问新训练的探针能否读出属性。 |
| 怎样用于研究 | 删除可解码属性后测原模型行为，区别可读与利用。 |
| 不可越过的边界 | 必须检查目标改变是否充分及非目标误伤。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper CAUSE-AMNESIC --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### CAUSE-MEANLEACE — Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L4.3 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://aclanthology.org/2025.findings-acl.674/ |
| 做到了什么（沿原笔记） | **问题与方法。** 比较INLP与Mean Projection、LEACE等消除方法，并检查随机方向干预造成的任务损伤。 |
| 怎样用于研究 | 采用匹配维数/扰动规模的随机方向对照，比较消除方法。 |
| 不可越过的边界 | 不是简单否定所有旧amnesic结论；方法有效性按对象检查。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper CAUSE-MEANLEACE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### CAUSE-RELIABILITY — How Reliable are Causal Probing Interventions?

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L1.D、L2.7、L4.3、L4.4 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://aclanthology.org/2025.ijcnlp-long.47/ |
| 做到了什么（沿原笔记） | **问题与方法。** 区分干预是否充分改变目标概念（completeness），以及是否保留非目标概念（selectivity），评估干预可靠性。 |
| 怎样用于研究 | 同时记录completeness、selectivity与原任务变化。 |
| 不可越过的边界 | 有限验证概念/探针可靠不等于对所有内容可靠。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper CAUSE-RELIABILITY --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### CAUSE-LEACE — LEACE: Perfect linear concept erasure in closed form

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2023；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L4.3 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://papers.nips.cc/paper/2023/file/d066d21c619d0a78c5b557fa3291a8f4-Paper-Conference.pdf |
| 做到了什么（沿原笔记） | **问题与方法。** 在明确的线性可访问性和失真准则下构造闭式概念消除变换。 |
| 怎样用于研究 | 作为线性概念消除候选，独立验证目标抑制和非目标损伤。 |
| 不可越过的边界 | perfect linear不等于任何非线性读出均失效或未来动力学不受扰。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper CAUSE-LEACE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### CAUSE-ERASURE — Fundamental Limits of Perfect Concept Erasure

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L4.3 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://proceedings.mlr.press/v258/chowdhury25a.html |
| 做到了什么（沿原笔记） | **问题与方法。** 以互信息刻画概念完全消除与保留原表征信息的权衡；定理在有限支持、Markov关系等条件下成立，部分构造使用概念条件分布的支持分离等假设。 |
| 怎样用于研究 | 在概念消除时显式记录效用权衡和任务变量。 |
| 不可越过的边界 | 原utility以与原表示的信息联系定义，不自动等于任意下游任务效用。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper CAUSE-ERASURE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### READ-DECODESHARE — DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L4.3、L4.4 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://proceedings.mlr.press/v306/shao26j.html |
| 做到了什么（沿原笔记） | **问题与方法。** 在实际KV缓存解码期估计跨任务共享子空间，实施解码期投影删除；设置维数与激活能量匹配的对照、跨任务评估，并做patchback恢复及特异性检查。 |
| 怎样用于研究 | 借决策阶段定位、预算匹配、patchback与特异性完整证据链。 |
| 不可越过的边界 | 这些控制原文已有；我们的增量要落在历史内容、状态种类和时刻。 |
| 本轮核查范围 | 本轮核对PMLR正式身份、作者、摘要和作者代码入口；patchback等细节沿旧文献卡，需正式全文后续核对 |

**读取原阅读卡与原计划：** `python retrieve.py paper READ-DECODESHARE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

## 八、干扰、更新与条件化能力：连接L5


### INT-ROTATION — Rotational dynamics reduce interference between sensory and memory representations

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2021；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.3、L3.6 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://www.nature.com/articles/s41593-021-00821-9 |
| 做到了什么（沿原笔记） | **问题与方法。** 小鼠听觉序列学习的神经记录，比较感觉和近期记忆的群体编码方向；结合稳定选择性与翻转选择性神经元的结构及理论模型。 |
| 怎样用于研究 | 将感觉—记忆方向旋转与干扰、行为效用并测。 |
| 不可越过的边界 | 动物记录不自动提供显式任务的行为干预；正交不等于资源释放。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper INT-ROTATION --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### INT-BARBOSA — Interplay between persistent activity and activity-silent dynamics in the prefrontal cortex underlies serial biases in working memory

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2020；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L5.2 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://doi.org/10.1038/s41593-020-0644-4 |
| 做到了什么（沿原笔记） | **问题与方法。** 猴神经活动、人类行为/EEG、计算模型及试次间PFC单脉冲TMS。研究上一次空间记忆的潜在痕迹，在下一刺激前重新激活后如何偏移本次报告；包括vertex位置和不同强度/无刺激对照。 |
| 怎样用于研究 | 把跨trial残留和适应偏差列为历史影响路径。 |
| 不可越过的边界 | trial间carry与单trial内reset不是同一操作。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper INT-BARBOSA --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### INT-FLESCH — Orthogonal representations for robust context-dependent task performance in brains and neural networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2022；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.3 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://www.saxelab.org/assets/papers/flesch_orthogonal_2022.pdf |
| 做到了什么（沿原笔记） | **问题与方法。** 神经网络情境依赖决策、人类行为/影像与猴神经数据；比较初始化/学习方式对应的表征几何以及抗干扰和噪声性质。 |
| 怎样用于研究 | 以情境相关表示分离解释泛化与干扰，检查噪声/任务条件。 |
| 不可越过的边界 | 按原稿纠正后的Neuron身份记录；正交与最优性的条件要保留。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper INT-FLESCH --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### INT-STREAMROT — Orthogonal-Rotational Dynamics Supports Efficient Encoding and Updating for Streaming Information in Working Memory

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.3、L3.6 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://doi.org/10.1523/JNEUROSCI.2220-24.2025 |
| 做到了什么（沿原笔记） | **问题与方法。** 结合RNN工作记忆任务与人类EEG，研究流式输入中的编码、更新及正交/旋转动力学。 |
| 怎样用于研究 | 把流式更新中的旋转/重排作为保持与更新兼顾的候选。 |
| 不可越过的边界 | 在线2025与卷期2026不重复；原机制不能直接移植为SNN证明。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper INT-STREAMROT --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### INT-UNABLE — Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L5.2 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://arxiv.org/html/2506.08184v3 |
| 做到了什么（沿原笔记） | **问题与方法。** 同一key反复被赋予新值，最终查询当前值，观察模型是否返回已被覆盖的旧值；包含固定输入长度的控制，不能把所有结果都解释为上下文变长。 |
| 怎样用于研究 | 区分上下文长度与前摄干扰、无关历史负荷。 |
| 不可越过的边界 | ICML workshop不是ICML主会；结果范围限定LLM测试条件。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper INT-UNABLE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### INT-DUALPROCESS — LLMs Remember First, Forget Last: Dual-Process Interference in Large Language Models

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L5.2 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://arxiv.org/html/2603.00270v3 |
| 做到了什么（沿原笔记） | **问题与方法。** 用关联学习中的AB–AC等范式比较旧关联干扰新关联（前摄干扰）与新关联干扰旧关联（倒摄干扰），覆盖多个LLM。 |
| 怎样用于研究 | 比较不同年龄和干扰顺序，避免单一recency解释。 |
| 不可越过的边界 | 题名更改的不同版本属同工作，不重复计数。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper INT-DUALPROCESS --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### INT-RECENCY — Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.3、L5.2 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://arxiv.org/html/2501.00658v1 |
| 做到了什么（沿原笔记） | **问题与方法。** 从近因偏置与层间过平滑分析状态空间模型的时序信息瓶颈，并提出对应缓解策略。 |
| 怎样用于研究 | 纳入recency与over-smoothing两种连续状态瓶颈作强替代解释。 |
| 不可越过的边界 | 不是所有SSM都有同一个普遍失效曲线。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper INT-RECENCY --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### INT-PRIMACY — Emergence of the Primacy Effect in Structured State-Space Models

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；ACML 2025；URL含26a不改会议年份 |
| 优先级/对应模块 | P2；L2.3、L5.2 |
| 原笔记来源 | [SNN_Memory_Readout_Interference_Literature_2026-10-17.md](sources/SRC019.md)（SRC019） |
| 原始入口 | https://proceedings.mlr.press/v304/morita26a.html |
| 做到了什么（沿原笔记） | 旧记录提示结构化SSM中可能出现首因偏置；需要在其指定初始化和结构条件下阅读全文。 |
| 怎样用于研究 | 作为对应小节后续精读入口，先补正文、方法和代码再决定是否加入实验。 |
| 不可越过的边界 | 候选身份不提供已验证实验发现；不由题名反推证据。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper INT-PRIMACY --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-SENSORYINT — Overcoming sensory-memory interference in working memory circuits

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L3.3、L5.2 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://doi.org/10.1101/2025.03.17.643652 |
| 做到了什么（沿原笔记） | **原文问题。** 保持旧内容时，持续进入的新感觉输入为何没有破坏记忆？作者比较手工构造/训练的RNN，并与猕猴数据联系。 |
| 怎样用于研究 | 比较门控、吸引子和动态编码避免感觉—记忆干扰的不同解释。 |
| 不可越过的边界 | 旧记录B级，需方法全文才能决定与SNN模块逐项对应。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-SENSORYINT --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-OUTDATED — Initially encoding attended but outdated information into working memory: behavioral and neural evidence

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L5.2 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://doi.org/10.1016/j.jml.2025.104668 |
| 做到了什么（沿原笔记） | **方法与结果。** 三组行为/EEG实验改变刺激起始异步SOA，研究曾被注意但随后无须报告的内容。短SOA下仍能检测其记忆痕迹，并观察到再次遇到相匹配内容时的注意抑制证据。 |
| 怎样用于研究 | 检验旧但被注意内容是否先进入状态，再被更新/压制。 |
| 不可越过的边界 | 先编码与最终使用分开，不能把输入旧内容都叫无效写入。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-OUTDATED --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-PLANUPDATE — Working memory adaptively recomputes plans amid distraction

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L5.2 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://doi.org/10.64898/2026.08.03.742569 |
| 做到了什么（沿原笔记） | **任务层面的价值。** 在含干扰的动态导航/目标任务中，以EEG与眼动研究工作记忆如何根据当前行动需求组织未来计划。 |
| 怎样用于研究 | 从保持推进到干扰下计划重算与实际任务调用。 |
| 不可越过的边界 | 预印本局部核查；重计算与单纯存储保持须分开。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-PLANUPDATE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

## 九、学习形成、公平选择和系统效用：L6与P3


### LEARN-TSNEURON — Resolving the Timestep Scaling Paradox in Spiking Neural Networks with a Timestep-Scalable Neuron Model

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；ICML 2026；上传20页PDF |
| 优先级/对应模块 | P1；L1.E、L2.4、L6.1、L6.2 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://proceedings.mlr.press/v306/ye26k.html |
| 做到了什么（沿原笔记） | 分析扩大模拟时间步带来的梯度和表现问题，提出timestep-scalable神经元并给出规定条件下的推导与实验。 |
| 怎样用于研究 | 把前向历史与指定STE下的梯度界、周期截断和实际训练效果分开。 |
| 不可越过的边界 | 更大上界不推出实际梯度更大；Proposition 4证明空白等旧核查边界保留。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper LEARN-TSNEURON --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### LEARN-TIC — Exploring Temporal Information Dynamics in Spiking Neural Networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2023；AAAI 2023；旧公开代码审计记录 |
| 优先级/对应模块 | P1；L6.1、L6.3 |
| 原笔记来源 | [SNN_paper_roadmap_and_reading_log_v01.md](sources/SRC020.md)（SRC020） |
| 原始入口 | https://arxiv.org/abs/2211.14406 |
| 做到了什么（沿原笔记） | 讨论SNN时间信息分布并提出相关训练方法；旧记录核对其Fisher相关对象、首步论证和全部可见功能代码。 |
| 怎样用于研究 | 先明确Fisher/参数敏感性、样本轴和批次计算，再讨论首步信息。 |
| 不可越过的边界 | 批均值梯度平方不等于逐样本平方平均；Fisher不是内容保留测量。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper LEARN-TIC --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### LEARN-NEUROTRAIN — NeuroTrain（完整题名以原始版本核验为准）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；2026-05 arXiv；正式身份待核 |
| 优先级/对应模块 | P1；L6.1、L6.4 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://arxiv.org/abs/2605.15058 |
| 做到了什么（沿原笔记） | 提供神经形态学习规则/训练比较框架；完整题名与当前正式身份仍按旧条目待核。 |
| 怎样用于研究 | 参考学习规则接口与公平训练预算。 |
| 不可越过的边界 | 统一接口不等于算法已在同条件下比较；不据摘要批评全文缺少实验。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper LEARN-NEUROTRAIN --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### LEARN-BRAINTRACE — Model-agnostic linear-memory online learning in spiking neural networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；Nature Communications（旧扩展条目） |
| 优先级/对应模块 | P2；L6.1、L6.4 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://www.nature.com/articles/s41467-026-68453-w |
| 做到了什么（沿原笔记） | 旧扩展记录介绍模型无关的在线学习及相对网络规模的线性内存特点；仅作后续全文入口。 |
| 怎样用于研究 | 区分网络规模与序列长度的内存复杂度。 |
| 不可越过的边界 | 旧记录C级；linear-memory不自动等于任意条件O(1)。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper LEARN-BRAINTRACE --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### LEARN-BULLET — Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；2026 arXiv；作者标ICML，旧记录未独立核正式目录 |
| 优先级/对应模块 | P2；L1.E、L6.4 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://arxiv.org/abs/2603.13283 |
| 做到了什么（沿原笔记） | 提出连续时间事件原生并行训练方向；旧记录的会议信息尚未独立对照正式目录。 |
| 怎样用于研究 | 连续时间事件原生训练是比较轴，不默认所有SNN都time-bin+普通SG。 |
| 不可越过的边界 | 正式身份与代码执行仍待核。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper LEARN-BULLET --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-PALIMPSA — Learning to Remember, Learn, and Forget in Attention-Based Models

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.5、L6.5 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://proceedings.mlr.press/v306/bonnet26a.html |
| 做到了什么（沿原笔记） | **问题与方法。** Palimpsa把在线关联记忆中的稳定性—可塑性权衡显式化，以Bayesian metaplasticity及重要性状态调节注意状态的可塑性；在MQAR与推理任务评估。 |
| 怎样用于研究 | 参考可学习写入/遗忘与存储规则。 |
| 不可越过的边界 | 不能把引入记忆模块本身当空白；训练和测试更新权限需区分。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-PALIMPSA --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### REC-TITANS — Titans: Learning to Memorize at Test Time

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；沿原文献卡的出版/预印本身份；本轮不自动升级 |
| 优先级/对应模块 | P1；L2.1、L6.4 |
| 原笔记来源 | [SNN_2025_2026_Preprint_Supplement_2026-10-17.md](sources/SRC001.md)（SRC001） |
| 原始入口 | https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html |
| 做到了什么（沿原笔记） | **边界。** 有效写入或surprise指标不是当前任务相关性的充分证明；“惊讶的内容”和“未来有用的内容”需要分别验证。该文已有正式发表版本，不应仍仅标“2025预印本”。 |
| 怎样用于研究 | 把测试时学习记忆与固定参数递归作为不同状态系统登记。 |
| 不可越过的边界 | 旧笔记记为NeurIPS2025主会，不只写预印本；梯度/参数更新成本单列。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper REC-TITANS --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### BEN-SCRATCH — Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2024；ICLR 2024；扩展摘要不重复计数 |
| 优先级/对应模块 | P1；L5.4、L6.3、L6.4 |
| 原笔记来源 | [SNN_Benchmark_Reference_Audit_2026-10-17.md](sources/SRC007.md)（SRC007） |
| 原始入口 | https://arxiv.org/abs/2310.02980 |
| 做到了什么（沿原笔记） | 说明长序列模型比较会受到数据驱动先验/预训练条件影响，公平性不能只固定表面架构与训练轮数。 |
| 怎样用于研究 | 把预训练/初始化先验列为公平比较轴，并做等搜索成本参照。 |
| 不可越过的边界 | 原结论不能直接推广为所有SNN训练的相同结论。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper BEN-SCRATCH --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### EVAL-ROBUST — Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；CVPR 2026；2025预印本同一工作 |
| 优先级/对应模块 | P1；L5.5、X.STATS |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://arxiv.org/abs/2512.22522 |
| 做到了什么（沿原笔记） | 研究攻击/评估方式对SNN对抗鲁棒性结论的影响，说明评价工具自身需要验证。 |
| 怎样用于研究 | 把测量/攻击工具有效性放在能力结论之前。 |
| 不可越过的边界 | 预印本与会议版不重复计数；旧正文分析主要基于arXiv。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper EVAL-ROBUST --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### EVAL-FRAMEWORKS — A comprehensive multimodal benchmark of neuromorphic training frameworks for spiking neural networks

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2025；Engineering Applications of AI（旧官方摘要记录） |
| 优先级/对应模块 | P2；L5.5、L6.4 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://www.sciencedirect.com/science/article/abs/pii/S0952197625015453 |
| 做到了什么（沿原笔记） | 对多个SNN训练软件框架作多模态任务评价；旧记录为官方摘要/元信息。 |
| 怎样用于研究 | 登记框架、内核与管线对速度的影响。 |
| 不可越过的边界 | 框架速度不是神经元机制能力排名。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper EVAL-FRAMEWORKS --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。

### EVAL-LIGHTWEIGHT — From Lightweight CNNs to SpikeNets: Benchmarking Accuracy-Energy Tradeoffs with Pruned …（旧入口缩写）

| 项目 | 本研究的调用记录 |
|---|---|
| 年份/出版身份 | 2026；WACV 2026；完整题名待从正式页补齐 |
| 优先级/对应模块 | P2；L5.5 |
| 原笔记来源 | [SNN_Benchmark_2025_2026_Expanded_Audit.md](sources/SRC003.md)（SRC003） |
| 原始入口 | https://openaccess.thecvf.com/content/WACV2026/html/Bin_Kabir_From_Lightweight_CNNs_to_SpikeNets_Benchmarking_Accuracy-Energy_Tradeoffs_with_Pruned_WACV_2026_paper.html |
| 做到了什么（沿原笔记） | 比较轻量CNN与SpikeNets、剪枝及精度—能耗取舍；旧记录为官方摘要。 |
| 怎样用于研究 | 引入同精度轻量ANN和剪枝参照。 |
| 不可越过的边界 | 旧记录只有官方摘要，不推定训练/转换细节。 |
| 本轮核查范围 | 未在本轮逐篇重新独立核验全文；保留原稿记录和入口 |

**读取原阅读卡与原计划：** `python retrieve.py paper EVAL-LIGHTWEIGHT --full`。旧A/B/C阅读等级在原卡保留；本轮收录不升级阅读等级。



# 修订与后续调用

`DECISIONS_AND_CORRECTIONS.md`保留D1–D13、代码指标修正、L3.2约束和版本差异；`IMPLEMENTATION_READING_GUIDE.md`规定每次实现先读原plan、文献定义、真实代码，再做校准和证据归档；`COVERAGE.md`区分原文字节/行完整性与语义审定，`literature/UNRESOLVED_REFERENCES.md`保存缺口。上述文件与来源、代码、离线检索脚本一并在完整包中。

本总册服务后续研究执行，不把索引整理、工具校准、探索训练和论文确认结论混为一种完成状态。


## 3.3.3复审与3.4—3.7执行记录（2026-10-25）

本次更新保持已有研究目标：系统性时序能力判断与归因，并将诊断用于预测与设计。不是新增指标目录，也不是因方法有来源就放弃新的研究连接。

| 执行项 | 实际状态 | 主要新用途 |
|---|---|---|
| 3.3.3 v0.2 | 63项测试；旧8视图差分重算；同视图fresh重验1次；0新SNN训练 | 检出有限样本伪交互，修补情境斜率对照和标签预算 |
| 3.4 | 22项测试；Gaussian原函数等价核对；Gaussian/同矩H3压力测试；9视图×3组 | 成分→任务方向m→权重波动M→学习曲线→读取改进 |
| 3.5 | 固定LIF载体偏序、计数匹配和计数保持控制；旧模型轨迹清单 | 区分瞬时载体、窗口时序和完整运行状态 |
| 3.6 | 独立序列跨时矩阵、仅训练集正交对齐、旋转控制、6时刻接入 | 区分读取失配和恢复下降 |
| 3.7 | 独立选rank RRR（含0）、限定Gaussian Fisher、纯nuisance反例、实际current等式 | 区分共享变异、结构接口、目标到达和输出作用 |

仍是原单次更新checkpoint272ae6230942ebeb；其SHA256为f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e。任务/场景/划分重复不作为独立SNN seed；formal=0/435；audit_test未访问。

读取顺序：3.3.3复审稿→3.4—3.7协议第4—5节理论连接→第7节设计用途→3.5/3.6/3.7→代码及证据。新的阅读记录区分正式论文、预印本版本、选读范围、完整选定源码、数值执行及未解决入口。旧文献卡不自动升级阅读等级。

完整文件为SNN_L3_3_3_Nonlinear_v0_2_2026-10-25.md与SNN_L3_4_to_3_7_Research_Protocol_2026-10-25.md；配套ZIP包含可执行代码、数据用途合同、已知答案、接入结果和散列核验。


## 3.4独立深入复审与执行更新（2026-10-25，v0.2）

本次保留原W0–W3与六层责任。完整稿为SNN_L3_4_Deep_Review_and_Execution_v0_2_2026-10-25.md，按“问题合同→参照校准→内容/任务/谱/连续恢复连接→冻结后收益验证”组织，每部分均列原文、作者代码、适配理由与本项目研究增量。

新增理论/分析包括：含响应均值的有限p Gaussian条件律；五风险逐步替换及Berry–Esseen总体界/plug-in分离；Gaussian标签一阶正交化η；η的方差最优仿射控制及超额方差关系；H3任务支持翻转；谱能量与任务容量/harmonic关系；固定观测噪声与绝对PCR支撑控制；任意向量目标的有限p Hebbian MSE与独立矩增益；按独立scene构造的冻结后配对收益界。

执行：55测试通过（含Torch，无skip）；13分类视图×3角色；N8192独立分类评价场景/视图；4个训练draw；MLP限C末层post/sphere的F/G/FG，7680 vectorized optimizer steps、1536单任务头拟合；13连续目标视图、1248闭式映射；9高阶估计控制诊断。当前9项SNN诊断没有选中比普通控制更好的加权控制，该无收益结果保留。

单checkpoint流程结果：C两层post拼接/F/p128的Hebbian分类错误.36972→所选可逆谱读取.26901，收益.10071，冻结条件下同时区间[.05170,.14972]；同p固定λ ridge更好，为.23199。C末层/H1末层的收益小且区间跨0。39组合共享模型/场景，不当独立seed。连续值增益拟合额外用了2048个标签，不与纯p标签方法混算。

仍为checkpoint272ae6230942ebeb、原1次训练更新，SHA不变；新SNN训练0；formal=0/435；audit_test未访问。当前工作把设计推进到可直接接正式模型的程度，没有冒称正式架构发现。

[最新代码包](https://chatgpt.com/library?file_id=libfile_390c2c6120308191b4bd93dd1cf6f35f&account_id=personal)；[3.4–3.7总协议与本次入口](https://chatgpt.com/library?file_id=libfile_af585a9693ec8191828a8016de1d805e&account_id=personal)。旧执行表保留为历史；本节的v0.2状态覆盖旧3.4九视图/22测试进度。

## 3.5深入复审与执行记录（2026-10-26）

当前入口为 **SNN_L3_5_Deep_Review_and_Execution_2026-10-26.md**；完整代码包为 **SNN_L3_5_Carriers_State_Timing_v0_1_2026-10-26.zip**。[3.4–3.7总协议](https://chatgpt.com/library?file_id=libfile_af585a9693ec8191828a8016de1d805e&account_id=personal)已同步路由与历史标记。原3.5主题、论证A及11条计划文献保留，逐项用途/阅读深度存于新包references/。

| 子节 | 已执行的研究增量 | 对照与代码 |
|---|---|---|
| 3.5.1 | pre/post/output/event合同；soft/partial/hard复位的可逆性、越阈歧义与精度放大 | snnTorch选定函数、DMP驱动阶段；carrier_theory.py、capture35.py |
| 3.5.2 | 全零观测原子的任意分类器风险下界及同时有效置信下界；条件偏差/变异 | 2025少活跃神经元偏差；carrier_theory.py、条件重复panel |
| 3.5.3 | 增量风险与条件期望关系；冗余/XOR/可逆重参数化；联合相对两个单项 | conditional probing及选定官方代码；probes35.py |
| 3.5.4 | 四种精确计数匹配任务；时序不变量；冻结/重新拟合读者；尾批、碰撞、quota合同 | Beyond Rate Coding 2025/26选定原文/源码；timing_controls.py、event_adapter35.py |
| 3.5.5 | aux/cache/readout完整状态；可达共同探测；局部可观测性；独立写/读门控 | Wolff、Paluch、DMP、Delay-SSM、Linear attention；run_state_contracts.py、slow_path_reference.py |
| 3.5.6 | 自然接口、同维、同payload位数；访问、变换及状态成本分列 | 22观察主分析与U-only补充；run_carrier_review.py、run_unrequested_extension.py |

实际验收：58测试全部通过，无skip；12注册cell（11主变体+analog对照）×8参考骨架×5读出=480个随机未训练实例的快照续跑逐值一致。完整forward与stream的最大相对差约2.38e-7。此覆盖不构成训练后架构性能图谱。

同一个原训练只有一次更新的checkpoint保持不变。主分析22观察×6角色任务系统、p8/32/128、2个共享嵌套训练draw；22个U-only系统另用新8192场景。闭式映射共1848（分类924、连续924）；主MLP2560 vectorized steps、512个单任务头；人工时序MLP另800步、8个头，并有48个闭式时序拟合。没有新增SNN训练；正式benchmark仍0/435；audit_test未访问；未下载真实数据集。两个评价池各8192个独立场景，不能跨池声称配对优势。

原两个评价池各前128场景重新生成后，应用已保存冻结读者，932个场景损失数组前缀全部逐值一致，最大差0。此为一致性重放，不算独立统计确认。作者12个固定源码blob的字节匹配，核读/执行范围按文件登记；未执行作者完整训练、Octave或Chisel构建。

重要发现的解释：联合观察在当前旧模型中可明显优于spike，却未在p128所查角色中以正置信下界同时胜过两个单项；因此没有将访问膜状态的收益写成已证实的普遍互补。时序反转可使冻结读者0→100%错误，而同类读者在反转数据上重新拟合后回到0，说明要区分可逆变化后的读出失配与信息损失。它们形成正式实验的辨别控制和设计候选，研究目标保持系统性能力判断与归因。

同步勘误：3.4曾将G口头概括为全部其他来源，现已改正为cotarget；U另列。现模型F=C/G=H1或交换，各4维，FG8维，U=H2–H4共12维。原3.4代码和数值均不变。下一深审为3.6，3.7与L4再接目标到达、选择性操作和原生任务收益；本轮不以工程验收代替这些正式结论。

## 3.6深入复审与执行记录（2026-10-26）

当前完整文档 **SNN_L3_6_Deep_Review_and_Execution_2026-10-26.md**，代码包 **SNN_L3_6_Temporal_Geometry_v0_1_2026-10-26.zip**。原3.6题名和10条计划文献保持，逐篇原文/源码/实际执行/缺口见新包references/literature_use_and_plan_coverage.json。总协议已同步入口，原3.6六时刻段落标为历史。

7部分：3.6.1来源/年龄与可用性；3.6.2同刻、固定跨时与适配读取；3.6.3平移/增益/O(D)/SO(D)/affine；3.6.4内容特异及离散事件几何；3.6.5事件、输入和辅助状态时钟；3.6.6噪声/时序相关性/参与任务的动态模式；3.6.7独立选择及设计验收。

理论与控制新增：平方风险的Bayes项/失配项；旋转—增益—噪声的精确Gaussian参照；任意时间均值/协方差的延迟损失；可达/可观察模式限定；单次阈值映射零导数与全局可读的共存；有限差分平方与事件跨界率的尺度律。条件均值和概率几何另列，避免将逐路径性质误套到PSTH/平均率。8内容×128噪声重复检查了条件均值度量与平均逐路径度量的半正定差。

工程验收：50测试无skip（含12注册cell导出子案例）；9观察×5来源×16×16×4映射，分类和连续恢复共92160相关风险单元。378次矩阵求解对应3060逻辑内容/目标/读者拟合。每独立读者128场景，两draw共有222独特训练场景；2048独立选择，8192新场景确认。45同时比较中22下界为正。6种映射的长deadline补充1024场景，900比较中15下界为正，两族95%不合称全研究95%。

第一层post/H1同刻错误由写入时约.108升至末时刻约.492；逐时间读者提高早期读取不等于记忆延长。另在相同晚时刻t23，原t15头的第一层post/C错误由.46942经affine适配降至.25909，全900族区间的降低约[.06718,.35347]。该增益使用额外成对校准和映射，属于指定访问改善，不是原生任务或架构排名。6配对schedule、360时钟行、12有限尺度组和8条件均值组均留存。

所有证据仍来自原一次更新checkpoint；文件SHA256不变。新SNN训练0，正式0/435，audit_test未访问，未下载真实数据集，未使用A40。8个固定作者源码文件字节匹配，执行仅限选定函数及明确设备适配；部分2026全文/官方代码仍未取得，单独记录。

工程恢复备注：本轮取回的旧3.5压缩包缺尾，完整恢复未完成；3.5正文可读取。当前3.6包自包含代码依赖与checkpoint，依赖取自完整CRC校验的3.4包，41个继承Python文件逐字节相同，当前包不需要旧3.5压缩包。

下一深审为3.7：在这些来源/观察/时间合同上接真实层间到达和目标传递；再由L4的选择性干预、恢复与原生任务收益完善设计证据。50项工程测试和格点数量均不替代正式多模型发现。


## 3.7深入复审与执行记录（2026-10-27）

文档 **SNN_L3_7_Deep_Review_and_Execution_2026-10-27.md**，完整代码包 **SNN_L3_7_Content_Transmission_v0_1_2026-10-27.zip**。7部分以真实返回消息和接收状态为主线，逐项覆盖原3.7六条计划文献；新增近期群体通信与有限字母表参照。相关全文/选读/源码/执行/缺口分开登记。

理论与控制：区分惩罚RRR最优解和先ridge后预测PCA；共享响应与目标信息分开；有限合法消息的碰撞条件不等同于线性rank；加入噪声/精度/译码权限与已知逆充分条件；保留相关噪声Fisher交互、Schur条件增量和接收残余调谐抵消/增强；I_min FIT/cFIT与context XOR、共同原因及有限样本偏差对照；瞬时null与有限视野可观察性分别测量。

实际完成：42测试无skip，含12单元的两层MLP实际边等式及96个未训练单元×骨架被动模块观察子案例。主模型12观察×5来源×6时刻×p128/512，2共享draw，288次联合矩阵求解、1440相关分类/连续风险单元。512无标签校准，1024训练池（各读者仅128或512标签），2048独立选择、8192新场景评价，47条独立选rank RRR。主10个同时区间均含0；3960个既有固定读者成对比较另构造补充族，属于同数据重放，两族各自95%，非全研究95%。

t15/p128的C：sender post错误.20263，actual output .49454，receiver current .49438，receiver post .42398；source count4 .40193、raster4 .40874。该结果支持合法历史窗口的访问作用，没有证明精细时序优于计数。层间算子rank32、条件数约62.69；4096新场景中全精度已知逆恢复全部原二值消息，量化实验另列指定读者、范围/clipping和32×bit载荷，非Bayes最优误差或硬件节能结论。

条件噪声128内容×16重复，按内容64/32/32划分，48条均值/残差/打乱RRR记录。H1/C各128配对场景、4种真实边操作：完整donor后缀精确重现donor接收轨迹与原生预测，baseline patchback精确恢复。源局部状态和操作前缀不变；此为已知前馈图中的指定路径作用，内容选择性和原生任务提升仍交L4验证。

原checkpoint仍只有一次原训练更新，SHA256不变；新SNN训练0，正式0/435，audit_test未访问，未下载真实实验数据集，未使用A40。16作者固定源码文件blob一致；选定原Python函数实际执行，FIT仅明确公式转写，MATLAB与完整作者训练未执行。42工程测试和所有格点、时刻、读者均不作训练模型数。

下一阶段沿既有L4计划：先固定内容选择性作用、一般损伤/工作点对照、恢复及原生任务收益，再接P3未见条件的诊断选择与成本。复杂骨架完整cut、正式训练seed与真实分类/重建仍需实测。


## 2026-10-27：3.6 v0.2收尾交付

只收尾3.6，原3.7记录保留。交付入口SNN_L3_6_Delivery_Guide_2026-10-27.md、正式规格SNN_L3_6_Paper_and_Experiment_Specification_2026-10-27.md及完整v0.2代码包。正文原文件身份保留并更新内容。

67项测试、96分类+24图像重建捕获合同；540含偏置循环、54同lag迁移、25相位比较、576有限尺度/阈值/噪声/有限级观察量化记录；旧检查点8阶段和未训练continuous/q=1/20步9阶段全部通过，后者700个延迟比较匹配其自有检查点哈希。作者固定源码8份重新核查；2026流式旋转文献新增PMC方法/结果阅读。

保留进取的主线：系统判断同一内容的可恢复、可读取与事件尺度失配，并形成能迁移的设计规则。工具出处、实现适配、已知控制、旧模型发现和正式确认分别记录。没有新增SNN训练或下载真实数据；formal=0/435，audit_test未访问。新配置纯工程控制不计入论文模型证据。


## 2026-10-27：进入L4，完成整体框架与L4.1开发阶段

沿原五部分：观察权限与实际读出；可恢复性与真正利用；内容/载体/通路干预；损伤与恢复；重建中的历史调用。整体稿SNN_L4_Framework_and_Research_Plan_2026-10-27.md，第一部分深入稿SNN_L4_1_Deep_Review_and_Execution_2026-10-27.md，代码/证据包SNN_L4_1_Access_and_Readout_v0_1_2026-10-27.zip。L4.2—4.5有具体对照与验收规格，未冒称完成实验。

本轮31测试无skip；1440分类+288图像重建G/P/R与stream接口案例均为未训练控制。另在同一288个trajectory配置上验证已知线性头的输出累计实现，无需保存完整轨迹；固定且预先已知的头、T和查询范围是条件，不将其称为任意晚到查询的通用轨迹压缩。原头12个同W配置核查，只有post/last为原训练配置。

旧一次更新模型上15观察、p128/512、3个ridge候选共90联合拟合，120主风险单元与8常量基线；512校准、1024训练池、2048选择、8192新评价。12个预定分类同时比较中2个正下界：H1的spike轨迹访问、C的内部post访问。它们增加了不同权限/参数，不是神经元记忆延长；连续MSE变差及弱于常量的情况保留。保存场景重放最大差0。正式SNN训练仍0/435，audit_test未访问，未下载真实数据，未使用A40。

作者近邻MI-TRQR2025已选读方法，5个固定Git blob及选定函数/语句检查。区分训练mask、训练loss汇总、部署输出及自适应权重所读变量；恒定输入的MI归一化退化只作明确人工边界案例，不据此否定作者实验。2026-10-14 Temporal Representations预印本为直接框架近邻，目前只核实主记录/摘要，全文和官方代码缺口保留。


## 2026-10-27：L4.2及贯穿L4的数学/表征/潜空间补充

交付SNN_L4_Mathematical_and_Latent_Framework_2026-10-27.md、SNN_L4_2_Deep_Review_and_Execution_2026-10-27.md、SNN_L4_2_Literature_and_Code_Map_2026-10-27.md及完整L4.2包。整体框架原文件更新v0.2，保留原五部分。数学负责解释任务充分、嵌套风险、函数族/学习限制、任务度量、查询族与历史条件效用，并接入4.3—4.5的操作、恢复和重建。

实际：25测试、12解析/穷举控制；主6视图×2预算×2draw×4主/控制读者=96被选头、384风险。150次ridge联合求解含打乱和独立验证，96个MLP优化拟合共38400次离线头更新。额外2个独立验证头不参与主选择。拟合标签128/512、选择32/128，总160/640；512无标签校准与独立诊断640监督单列。原核心未更新。

当前头的请求8输出与全部20输出分别取row/null，各5个同秩随机方向；24投影、48冻结读者、144次矩阵求解、192风险另8常量基线。主8192与子空间4096独立评价池，不跨池构成配对优势；同一场景内16方向标签也不算独立实验单位。保存的384+192损失数组各前128场景重放；此为一致性验收。

旧模型发现：同权限重新拟合可改善未充分训练原头；17/24残差头选回零步，96个非线性辅助分类同时区间无正下界；当前头null中仍可恢复部分C，但需同时看同秩随机与常量，不推为独有存储或未来无作用。仿射经验目标从原头/零权重均达到闭式解，原完整网络训练充分性仍未成立。全部负结果保留。

文献：原七条逐项覆盖；新增Layer by Layer（ICML2025 Oral）、概念几何（ICLR2025 Oral）、REPA（ICLR2025 Oral）、REG（NeurIPS2025 Oral）、RAE（ICLR2026主会）、Latent Concept Disentanglement（ICLR2026主会）、Usable Information（2020基础）。后两篇2026报告形式未核，不沿二手列表升级。选读版本、8份固定作者源码、实际函数执行与代码缺口逐项登记。两个2026bioRxiv原计划候选全文403，候选身份保留。

下一节L4.3：在内容/观察/读者候选上加入自然配对、独立目标/非目标验证、随机损伤匹配、真实状态/通路及未来续跑；再接4.4恢复与4.5原生重建。新SNN训练0，formal0/435，audit_test未访问，真实数据和原预训练模型未下载，未用A40。


## 2026-10-27：L4.3内容、载体与通路干预开发交付

保留原6条文献调用与L4五部分。逐节稿按4.3.1自然内容配对、4.3.2消除/替换与稳定性、4.3.3独立验证、4.3.4状态所有者/阶段、4.3.5真实二值通路、4.3.6理论归因、4.3.7任务效用、4.3.8设计预测展开。新增三运行误差恒等式、有限输出风险界、SNN事件包络、相关内容MSE下限和固定协方差算子界，均注明基础来源和适用条件。

原模型仍只训练1次更新。开发执行238状态条件+24边条件（同2048配对）；看到无正则放大后追加112稳定性条件（新1024配对）。原/追加拟合bank均在各自评估前冻结；追加不称原计划确认。离线probe有186次ridge候选求解、14次ΔF映射求解、68次MLP拟合/27200更新，SNN权重不变。

40测试、10类已知答案控制、240未训练接口通过；1496场景损失数组均值与预定比较族重算；原238状态及24边条件各前32场景在batch128重放，输出/状态最大差0。6个完整快照及3个完整二值后缀供体控制精确重现。无操作注意力布局与实际float32 beta/包络余量问题均有修复记录，原952场景损失未改变。

6条原计划文献逐项覆盖；5个仓库9个固定文本源文件，原类/选定AST定义执行，未跑作者全模型/数据。DecodeShare核正式2026 Spotlight、36页作者PDF选定方法及维护源码；保留其错时/跨供体补丁可成功的事实。Canby正式2025版为准，官方仓库仍未确认；另登记2026-09候选预印本为仅摘要阅读，不据此引入数值或实现。

本轮新SNN训练0、正式0/435、audit_test未访问、未下载真实数据和预训练模型、未用A40。原生分类、原生重建、充分训练、多seed、复杂反馈与相关因素的内容结论待正式确认；L4.4、L4.5仍待逐节执行。


## 2026-10-27：L4.4恢复、补丁与方向验证开发交付

原L4五部分与Master三条调用保留。SNN_L4_4_Deep_Review_and_Execution_2026-10-27.md按8小节，SNN_L4_4_Mathematical_Specification_2026-10-27.md给出风险交互恒等式、完整快照/mean累计、信息权限、联合有限差分与事件区域内仿射/二次律，SNN_L4_4_Literature_and_Code_Map_2026-10-27.md覆盖原三条及四近邻；作者2仓库6固定文件、选定原定义/模块有限执行。DecodeShare已读作者36页PDF及维护入口；How Linear Attention Remembers进一步读方法/附录，官方包URL缺口保留；多中介2026论文所列repo访问404，不捏造执行。

实际主实验14损伤、38恢复情境、10方法，两批1024/2048场景配对，共760四组、9300损失数组。14新增补丁时刻算子、24独立验证probe；36ridge候选求解、24MLP拟合/9600离线更新。两批主策略在生成前冻结；晚补的自然事件几何明确为后主分析，未按结果选择剂量。补录R10延迟语义基线及MSE-only恢复比例元数据均有receipt，不改原预测/损失与策略。

确认池前256的补充：72事件政策四组、30联合情境/120子集、63真实二值四组、12真实通路封堵、72剂量条件；不是额外独立确认。240未训练分类/重建接口、1920scope验收，全部mean接口core-only仍有历史累计缺口，system恢复精确。38测试、14控制；9300+378损失数组、9232区间重算；760主单元各前16场景batch128重放载荷/状态/输出差0，171继承文件不变。

开发发现：早期H1独立ridge内容MSE正常0.08148、损伤1.01098、目标补丁0.08008；原任务没有达到MSE损伤0.01资格，两批不报告恢复比例。正风险交互可伴随补丁有害；另报输出Ω。固定事件max|Ω|约8.35e-7，自然可达0.662；同事件模式内仿射输出与二次MSE有条件证明/验收。发送膜恢复保留但未来脉冲改回损伤消息时，12案例均回到损伤输出且接收入口核验一致。结论限当前旧前馈模型及诊断权限。

L4框架更新v0.4，全L4数学主线更新v0.3；候选论文贡献是状态—事件—通信—读出之间的可辨别归因和未见条件设计预测，非首次发明patchback。充分训练、多seed、自然任务、非线性补丁族及原生重建待执行。新SNN训练0、正式0/435、audit_test未访问、未下载真实数据/作者模型、未用A40；下一部分L4.5。
