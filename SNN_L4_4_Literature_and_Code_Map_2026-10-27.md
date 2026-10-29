> **执行状态更新（2026-10-29）**：本文保留原研究记录；当前代码入口、训练/验收状态、数据清单以[当前状态](docs/CURRENT_STATUS.md)和[执行手册](docs/SERVER_RUNBOOK.md)为准。旧“完成”不自动表示全量训练完成。

# L4.4文献与作者代码：继承、适配、缺口和研究增量

2026-10-27，v0.1。先覆盖Master原三条调用，再添加直接影响4.4设置和解释的近邻。读原文、核源码、执行选定定义、复现作者整篇实证是不同级别；本轮没有运行任何作者预训练大模型或数据集。

## 1. 原计划逐项覆盖

| 原调用ID | 本轮阅读与代码状态 | 进入4.4哪里 |
|---|---|---|
| READ-DECODESHARE | 作者36页PDF的patchback/控制/附录及维护源码；正式ICML2026 Spotlight身份由作者项目确认 | 供体、时间、剂量、补丁回写、跨情境控制 |
| REC-LINEARATTN | arXiv v1原HTML方法与附录、作者主页；公开复现包具体URL未解决 | 写入/保留/读取路径、多个记忆载体、能力资格、独立确认 |
| CAUSE-RELIABILITY | 2025正式会议PDF方法/实验/结论；官方仓库仍未核实 | 独立自然probe、目标和非目标变化、任务变化分别记录 |

Master中的三条均保留。下列新增近邻补足多中介、补丁度量、AND/OR和路径再封堵；不能把已有activation patching本身包装成原创。

## 2. R1：DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions

**身份与来源。** ICML2026 Spotlight，作者项目[DecodeShare](https://zishan-shao.github.io/decodeshare/)，[作者PDF](https://zishan-shao.github.io/decodeshare/assets/DecodeShareICML2026.pdf)，[官方代码](https://github.com/Zishan-Shao/decodeshare)。本轮PDF已成功读取，不能写成不可访问。重点读patchback、迁移/控制和相关附录，并非声称重跑全部论文。

**他们做到了什么。** 对解码阶段共享子空间做定位与回写，检查供体/上下文/时间及维度或能量控制。其错时/跨情境补丁也可能成功，这属于需要解释的共享信号现象，不是失败证据。

**代码核对。** 固定commit `89ed6a36603d81d024b0bd5438048a93d3fff2ac`。维护入口为`experiments/03_patchback/subspace_patching_transfer.py`；另保留downstream/patchback_provenance的两脚本及README作为版本来源。`SubspacePatchHook`做x′=x−QQᵀx+p，先去掉原投影再写入p；不是无条件只加p。

**实际执行。** 原AST定义在人工Torch输入执行；patch hook与预期最大差0，tuple/prefill策略通过；每行随机范数最大误差2.39×10⁻⁷。人为零维span且要求非零范数时输出0；空flip集合的作者百分比为NaN。这些是退化输入的合同核验，不据此否定论文实验。

**我们怎样适配。** 保留同供体/异供体、错F、正确F但其他上下文、错时、三随机方向；使用同一已冻结外部delta的四组，独立probe和实际原任务分开。SNN额外声明post阶段、当步已发事件、未来续跑、aux/cache/readout所有者，以及不可匹配的零方向。

**增量位置。** 不声称首次patchback或首次匹配能量。希望新增的是事件系统中的恢复解释：何时只是当前读出覆写，何时内容被后续脉冲传递，何时需要多个所有者；最终要预测等预算的模型改进。

## 3. R2：How Linear Attention Remembers

**身份与来源。** Lee、Park、Kim、Ko；[arXiv 2609.33093v1](https://arxiv.org/html/2609.33093v1)，2026-09-27预印本。作者主页列为under review，不赋予oral/spotlight身份。

**他们做到了什么。** 将内容特异的写入、保留和读路径分开，控制非目标、负荷和经过距离，并考虑混合记忆载体及多个head的非加性；开发选择与留出确认也有区分。它是直接方法近邻，不能概述成“只测准确率”。

**本轮细读范围与限制。** 原HTML方法及附录选定段落，特别是write blocking、donor specificity、固定语义方向、read path、nonadditive heads和能力不足时不能做预定因果分析的限制。文中提到复现包，但本轮未从一级来源解决其公开官方URL；没有猜仓库，也没有运行GLA/GDN。

**我们怎样适配。** 保留写入—保持—读取和联合载体逻辑；把运行时所有者扩展到膜、适应、突触、缓存和部署读出累计。补充受损背景的独立内容验证，并把任务资格、风险交互、预测交互和真实二值通路封堵分开。

**增量位置。** 线性注意力的研究问题已经覆盖很多记忆机制环节。我们的空间是SNN离散事件/重置导致的载体—通信分离、事件分层理论和部署读出记忆。只有跨模型规律与设计收益成立后，才能形成相对于该近邻的实证贡献。

## 4. R3：How Reliable are Causal Probing Interventions?

**身份与来源。** Canby、Davies、Rastogi、Hockenmaier；IJCNLP-AACL2025正式论文，[正式PDF](https://aclanthology.org/2025.ijcnlp-long.47.pdf)，[作者项目](https://ahdavies6.github.io/causal_probing_reliability/)。正式版为准，不沿用更早版本相反的概括。

**他们做到了什么。** 分别检查intervention completeness、selectivity及任务变化，并用独立数据训练验证probe。正式版观察到非线性方法通常更可靠；它不是“一切线性干预都错”或“所有非线性操作都可靠”的定理。

**代码状态。** 官方仓库仍未核实。只登记原文方法阅读，不称执行了作者实现，也不把无关同名代码补为官方仓库。

**我们怎样适配。** 自然状态上独立训练/选择的ridge和预算MLP；逐内容而非只报总体平均；特别补齐延迟patch时R10受损状态的probe风险。内容变好、非目标不伤、原头变好分别评分。本轮补录缺失R10基线没有重拟合probe或更改主策略，留下completion receipt。

**增量位置。** 在具有真实阈值、reset和未来递归的系统中，同时检验语义命中与后续作用；更强非线性补丁族尚未完成。有限probe读不到不等于信息已被证明消失。

## 5. R4：The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching

**身份与来源。** Vaidyanathan、Arbour、Mueller、Niekum、Jensen；[arXiv 2606.27510v1](https://arxiv.org/html/2606.27510v1)，2026-06-25预印本。原文[摘要页](https://arxiv.org/abs/2606.27510)；不升级为已接收顶会报告。

**他们做到了什么。** 从多中介角度讨论直接/间接与交互项，展示旁路、负向补偿和组件共同作用对patch解释的影响。单个补丁不变不能自动推出该组件不参与计算。

**代码状态。** 原文给出`https://github.com/sankaranv/mech-interp-mediation-analysis`。本轮访问该repo、contents和commits为404，未取得可固定源码；只记录“该URL此时未取到”，不宣称作者永远没有开源。

**我们怎样适配。** 引入联合损伤下的∅/A/B/AB恢复、二阶Möbius输出差分，保留AND/OR/XOR控制；不把单点效应强行归一成因果份额。同时区分I_R和Ω：评分非线性可使预测仿射系统产生正风险交互。

**关系与原创边界。** 本文不宣称首次发现多组件交互。风险分解是精确代数适配，不是对该文的反例；其结果变量也可以直接选任务损失。我们的目标是用事件层、状态所有者和真实通信实验，解释SNN多载体恢复的具体来源并指导改动。

## 6. R5：How to use and interpret activation patching

**身份与来源。** Heimersheim、Nanda；[arXiv 2404.15255v1](https://arxiv.org/html/2404.15255v1)，2024方法预印本。

**他们做到了什么。** 系统解释noising/denoising、路径patching及AND/OR式冗余/协同，提醒补丁结果依赖被问的问题，不能自动给出最小回路。

**代码状态。** 本轮未确定专属官方实证仓库；不把方法说明称完整作者模型复现。

**适配与增量。** AND/OR控制明确归于该方法近邻，XOR是经典布尔扩展。加入SNN精确逆操作、末步覆写、mean累计未恢复等已知答案；对“更多恢复一定更好”作可执行否证。新意不在布尔例子本身，而在这些控制排除SNN恢复分析的具体误判。

## 7. R6：Towards Best Practices of Activation Patching in Language Models: Metrics and Methods

**身份与来源。** Fred Zhang、Neel Nanda；[arXiv 2309.16042v2](https://arxiv.org/html/2309.16042v2)，[PDF](https://arxiv.org/pdf/2309.16042)。本轮以2024更新版本阅读；正式报告形式不依赖未通过的OpenReview页面访问。

**他们做到了什么。** 比较破坏方式、评分指标和窗口patch设置；选择指标或较大窗口可以改变观察到的解释。概率类指标和logit类指标并非总有同样的解释性质。

**代码状态。** 原文提到[Easy-Transformer](https://github.com/redwoodresearch/Easy-Transformer)代码基础。本轮没有额外执行该代码库，也未把它称为已核专属全实验包。

**我们怎样适配。** 同时报MSE、预测保真、辅助有界损失及sign错误；不跨任务量纲共用恢复分母；原任务恢复比例只在预定损伤资格后定义。全后缀、单步、选定单元和匹配计数/翻转分别报告，不把窗口大小当无关实现细节。

**增量位置。** 面向SNN的真实二值消息、已发事件和所有者阶段；硬件能耗未测，不把消息范数或spike数直接命名能效收益。

## 8. R7：Locating and Editing Factual Associations in GPT

**身份与来源。** Meng、Bau、Andonian、Belinkov，NeurIPS2022；[作者项目](https://rome.baulab.info/)，[官方仓库](https://github.com/kmeng01/rome)。

**他们做到了什么。** causal tracing通过污染输入与恢复中间激活定位影响，代码亦包含恢复后重新破坏另一位置的repatch逻辑。权重编辑与因果追踪是不同模块。

**代码核对。** 固定commit `0874014cd9837e4365f3e6f3c71400ef11509e04`，`experiments/causal_trace.py`及`util/nethook.py`。原trace_with_patch、trace_with_repatch、layername选定AST与原nethook模块在人工Tiny命名Torch网络上执行；恢复正常概率差0，再破坏回到损伤概率差0。

**我们怎样适配。** 恢复发送层post状态后，只封堵后续实际消息，保持发送层恢复状态；另核查接收affine层的真实输入和错窗口。12案例全通过。

**增量位置。** 不声称首次repatch或路径patching。贡献目标是SNN真实脉冲边、post时序与状态/通信分离的可审计实施；本轮没有实现ROME权重编辑，也未复现其知识编辑结果。

## 9. 固定源码账与复用边界

`references/selected_author_sources.json`记录2仓库、6个文件的完整commit、Git blob、SHA256、字节数和URL；`fetch_pinned_sources_l44.py`按固定commit下载并验证；`audit_recovery_authors.py`按原AST和原模块作限定执行。原作者整段源码不在新包再次分发，审查机上的私有副本不作为交付依赖。

| 源文件组 | 用途 | 本轮执行范围 |
|---|---|---|
| DecodeShare维护patchback文件 | 规范维护实现 | 选定hook、随机范数和汇总定义 |
| DecodeShare历史两脚本/README | 版本与控制来源 | 阅读与固定，不算额外模型实验 |
| ROME causal_trace.py | 恢复与repatch操作 | 选定原函数＋人工网络 |
| ROME nethook.py | 实际hook上下文 | 原模块原样导入 |

其余文献代码缺口留空并说明，不拿自写参考实现冒充作者代码。人工边界检查只验证函数合同；未宣称对作者全部工程、算法最优性或实证结果的审计结论。

## 10. 从文献到4.4每一小节

| 小节 | 主要借鉴 | 我们的适配 | 当前证据/待补 |
|---|---|---|---|
| 4.4.1四组和权限 | R1/R3/R4/R6 | 固定外部delta、原标签与供体标签分开 | 两pool已执行；在线恢复器未做 |
| 4.4.2实现上界 | R5/R7及递推基础 | 逆操作、末步覆写、完整运行snapshot | 240接口，不冒充训练实证 |
| 4.4.3内容/时间/供体 | R1/R2/R3 | 38情境、10方法、独立R10/修复probe | 760主四组；非线性patch族待做 |
| 4.4.4联合所有者 | R2/R4/R5 | v/aux/cache/readout、联合∅/A/B/AB | 120条件及mean/readout差异 |
| 4.4.5作用方向 | R2/R7 | post膜恢复、真实二值消息、实际接收再封堵 | 12封堵＋63二值四组；复杂反馈待做 |
| 4.4.6理论分析 | R4/R6＋线性代数 | 风险/预测交互拆分、事件凸区域、剂量二次律 | 条件证明与后主分析验收 |
| 4.4.7确认资格 | R2/R3/R6 | 冻结族、场景级有界区间、损伤分母资格 | 新场景确认，无模型seed复制 |
| 4.4.8设计指向 | 上述全部 | 根据失败归因选择读出/所有者/事件/通路改动 | 预测先冻结，未来等预算检验 |

## 11. 对论文贡献的务实要求

需要争取的突出结果是：不同SNN失败表象可由内容缺失、受限读取、历史累计、事件切换和通信窗口加以区分，并能据此在未见条件选择有效改动。文献已提供大量工具，本研究要交付的是系统性的辨别力和可迁移的设计预测，不靠重新命名patching获得新意。

本轮已把该目标变为代码、理论条件、对照和开发结果。充分训练、跨seed/模型/任务、原生重建与实际设计收益尚未完成；这些是未来论文最关键的证据，而非在文献表中补几篇即可替代。
