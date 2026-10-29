> **执行状态更新（2026-10-29）**：本文保留原研究记录；当前代码入口、训练/验收状态、数据清单以[当前状态](docs/CURRENT_STATUS.md)和[执行手册](docs/SERVER_RUNBOOK.md)为准。旧“完成”不自动表示全量训练完成。

# L4.3文献与作者代码调用表

2026-10-27，v0.1。本表覆盖Master原定L4.3的6条核心调用，保留其身份；另列事件理论参照与2026待深入预印本。论文结论、作者代码函数行为、我们的适配和实际执行分开登记。阅读范围以本表为准，不把获取PDF解释成每页逐字重读，也不把选定函数运行解释成原论文全实验复现。

## 1. 先按研究需要组织

| 论文内需要 | 核心文献 | 进入本节的具体位置 | 实际代码 |
|---|---|---|---|
| 区分可读与原利用 | R1 Amnesic Probing | 4.3.1自然内容参考、4.3.7原生风险 | bridge43、run_state_interventions |
| 消除方法及随机损伤对照 | R2 MP/LEACE2025；R4 LEACE2023 | 4.3.2、4.3.3 | fit_eraser、random_eraser、norm_matched |
| 目标/非目标验证可靠性 | R3 Canby2025 | 4.3.3 | 独立probe与重新拟合、能力提示 |
| 消除与保留效用不可兼得 | R5 AISTATS2025 | 4.3.6 | gaussian_erasure_utility_floor与解析控制 |
| 估计—执行阶段对齐、真实决策作用、恢复 | R6 DecodeShare2026 Spotlight | 4.3.4、4.3.5及L4.4输入 | run_post_edit、真实边；恢复实验另属L4.4 |
| 阈值/重置引起的离散跳变 | R7 EventProp2021 | 4.3.6 | 自行推导的离散有限扰动包络，未运行EventProp训练 |
| 输出成功与自然机制可能分离 | R8 2026预印本候选 | 抵消反例和进一步阅读问题 | 未复用其算法或源码 |

## 2. R1：Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals

**身份。** Elazar、Ravfogel、Jacovi、Goldberg，TACL2021。Master键CAUSE-AMNESIC。[正式文章](https://aclanthology.org/2021.tacl-1.10/)；[PDF](https://aclanthology.org/2021.tacl-1.10.pdf)；[作者代码](https://github.com/yanaiela/amnesic_probing)。

**做到了什么。** 以属性消除后的原模型行为研究信息利用，说明常规probe表现不应直接当任务重要性。我们的“可恢复”与“原利用”分开继承自这类问题，不是新提出的区分。

**本轮阅读/运行范围。** 核对正式页面和既有调用卡，重点复审`amnesic_probing/debias/debias.py`的三个投影函数。选定定义经AST原样提取运行；不导入wandb或训练分类器，不运行完整INLP流程。

**代码发现和适配。** `debias_by_specific_directions`通过多个行空间投影的和求联合零空间；两条独立方向的3维控制得到diag(0,0,1)，误差0。我们的主实验采用闭式LEACE/MP等候选，保留整载体置零等一般损伤控制，不把INLP方向迭代次数作为内容量真值。

**用于我们的哪一步。** 同权重原输出比较；自然F替换、F=0中性参考与局部供体状态分开。新研究目标是加入SNN实际状态阶段和通路，确定删除的是哪种内容、影响的是哪段后续计算。

## 3. R2：Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE

**身份。** Dobrzeniecka、Fokkens、Sommerauer，Findings of ACL2025。CAUSE-MEANLEACE。[正式文章](https://aclanthology.org/2025.findings-acl.674/)；[PDF](https://aclanthology.org/2025.findings-acl.674.pdf)；[作者代码](https://github.com/efemeryds/amnesic-probing-with-single-projection)。代码入口由PDF脚注确认，未用名称相似仓库代替。

**做到了什么。** 对比INLP、MP、LEACE，检查消除后属性probe、原任务、同数量随机方向、表征变化和添加真实标签后的恢复。其工作已经把随机基线和选择性检查加入行为解释；不能把这些列成我们首次补上的内容。

**阅读范围。** PDF第2—3节的步骤、随机/Dropout基线及恢复控制；代码`MPMethod.py`和打包`LEACE.py`。后者为源码审读，未作为第二套独立作者实验执行。

**实际函数审计。** MP的`get_directions`直接归一化类别均值差；在人工构造的等均值类上产生非有限方向。`mean_projection_method`便利入口固定input_dim=768。我们改为从实际h推断宽度、等均值返回秩0恒等矩阵；人工3维显式投影控制误差0。这些是适配边界，不表明该论文实际数据触发过错误。

**我们的扩展。** 连续4维因素与单个符号属性分开；逐样本位移范数匹配另加同秩控制；真实二值通路使用计数/翻转双匹配。金标签恢复在原论文已有，我们将其作为L4.4直接近邻，不把本轮带ΔF标签的诊断映射称无需监督的机制。

## 4. R3：How Reliable are Causal Probing Interventions?

**身份。** Canby、Davies、Rastogi、Hockenmaier，IJCNLP-AACL2025正式版。CAUSE-RELIABILITY。[正式文章](https://aclanthology.org/2025.ijcnlp-long.47/)；[PDF](https://aclanthology.org/2025.ijcnlp-long.47.pdf)；[项目页](https://ahdavies6.github.io/causal_probing_reliability/)。

**做到了什么。** 用completeness、selectivity及其harmonic reliability衡量操作；独立验证probe与干预probe使用不同训练样本，验证分布控制目标和非目标相关性。正式版报告其设置中非线性干预通常更可靠。本轮以正式版为准，不沿用早期摘要推断线性方法更好。

**阅读范围和代码状态。** 重点读第3—4节定义、40/40/20划分、验证probe能力、线性/非线性干预设置和附录范围。本轮未核到可确定归属的官方代码仓库；项目网页模板链接不当代码证据。我们自行实现对应诊断原则，不声称运行作者实现。

**用于我们的哪一步。** 方向拟合、独立自然probe、操作后probe重新拟合与评估分开；多来源分别报告，未到达内容不进入保留结论，弱probe增加能力提示。当前不复现其全部攻击和harmonic评分，而是保留任务风险、命中和损伤向量。

**研究增量。** 将可靠性接入动态拥有者/阶段、供体范围和事件传播，观察可靠操作何时仍无法重现完整内容变化。相关因素下的SNN实验还未执行，不能把独立生成器上的验证当成已经解决相关内容问题。

## 5. R4：LEACE: Perfect linear concept erasure in closed form

**身份。** Belrose等，NeurIPS2023。CAUSE-LEACE。[官方论文PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/d066d21c619d0a78c5b557fa3291a8f4-Paper-Conference.pdf)；[作者代码](https://github.com/EleutherAI/concept-erasure)。

**做到了什么。** 给出线性guardedness的等价条件及闭式仿射消除，以受约束最小失真构造解释性工具。其分类命题针对对仿射预测分数凸的损失，不将0—1阈值准确率直接替换进去。

**阅读范围。** 定义2.3、定理3.1、协方差等价、闭式构造及`leace.py/caching.py/shrinkage.py`完整相关函数。实际导入原始LeaceEraser/LeaceFitter类，不改写作者实现。关闭收缩/方差约束、固定浮点64后，人工满秩数据上与我们的参照最大差6.65×10⁻¹³。

**对照后发现。** 作者默认有自适应协方差收缩、方差trace约束和绝对svd_tol=.01。本轮初始无正则参照省略前两项，出现近奇异放大；追加作者默认版本和固定收缩版本。连续概念单位缩小的人工测试中，默认绝对阈值秩0、相对阈值秩2，提示连续扩展应记录单位；不据此反驳one-hot理论。

**我们的适配。** 32维实际状态、明确连续/二值目标、独立验证、相对数值秩、固定度量的范数界及真实SNN注入。固定收缩有前人基础，论文新意应体现在它如何避免错误机制判断并指导后续实验。

## 6. R5：Fundamental Limits of Perfect Concept Erasure

**身份。** Chowdhury等，AISTATS2025。CAUSE-ERASURE。[正式文章](https://proceedings.mlr.press/v258/chowdhury25a.html)；[可读原文](https://arxiv.org/html/2503.20098v1)；[作者代码](https://github.com/brcsomnath/PEF)。

**做到了什么。** 讨论完美概念消除的条件、表征信息保留效用和极限，并构造PEF。其信息论命题依赖各自分布/支持/随机性假设，不可直接用作任意连续状态的MSE理论。

**阅读范围与执行。** 重点核概念消除定义、outer bounds、PEF构造及`src/pef.py`。原脚本有顶层数据/优化依赖，本轮只通过AST执行`_min_entropy/min_entropy`，核对返回耦合的排序后边缘；未运行Bayesian优化或数据实验，也不声称贪心耦合普遍达到全局最小熵。

**不直接复用的部分。** `get_mi`在同一批样本拟合分类器并预测概率，其数值依赖有限模型和拟合误差；本研究不将该量当严格总体互信息真值。我们报告独立任务风险和已知解析控制。

**我们的理论适配。** 对G=ρF+√(1−ρ²)U推导T⊥F时的任务MSE下限ρ²，并说明完全删除相关F与零误差保留G不可兼得。它是任务化适配，有完整推导；不是把原论文表征信息utility更名成我们的任务效用。主SNN相关因素扫描仍待正式实验。

## 7. R6：DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions

**身份。** Shao等，ICML2026 **Spotlight**；正式PMLR及作者项目页均可核。READ-DECODESHARE。[正式记录](https://proceedings.mlr.press/v306/shao26j.html)；[36页作者PDF](https://zishan-shao.github.io/decodeshare/assets/DecodeShareICML2026.pdf)；[项目页](https://zishan-shao.github.io/decodeshare/)；[作者代码](https://github.com/Zishan-Shao/decodeshare)。

**做到了什么。** 在真实KV-cached decode阶段估计跨任务公共子空间，采用维数/能量匹配的decode-only干预，并研究patchback和prefill→decode不对齐。它已经包含阶段对齐与恢复证据，不能将其写成只做相关。

**原文阅读范围。** 第2节对齐原则、H1/H2/H3及证伪条件；第3节patchback；附录能量对照。原文还报告跨供体及错时刻补丁可有较好恢复：公共信号可能跨样本/时刻稳定。我们据此保留对照的实际结果，不假定错时供体必须失败。

**实际源码。** 使用维护中的`decodeshare/subspace.py`，审读共享基、验证和JointSubspaceRemovalHook；原样提取hook运行人工二维/三维隐藏状态和tuple输出控制。有效单方向删除与期望逐值一致；disabled为恒等。

**适配边界。** 单独给hook全零基时，SVD保留了无奇异值筛选的列，人工输入变化最大4。源码另有基验证流程，因此这是独立hook的边界案例，不是证明原论文用了无效基。preserve_statistics默认False；若开启，会用当前批次统计改变坐标。我们的操作使用拟合池冻结均值、明确秩0行为，SNN消息则保持二值。

**研究扩展。** 将“估计在哪里、作用在哪里”落实到pre/event/post/aux/读出，区分存储和发送分支、单步和完整后缀，加入自然内容与局部范围参考。L4.4再比较内容恢复与公共通道恢复；SNN中是否存在可迁移的时间公共通道仍是待验证问题。

## 8. 另外两条参照

**R7：Event-based backpropagation can compute exact gradients for spiking neural networks。** Wunderlich、Pehle，Scientific Reports2021。[原文](https://www.nature.com/articles/s41598-021-91786-z)；[作者代码入口](https://github.com/eventprop)。用于提醒阈值事件和导数跳变要专门处理。当前未审计/运行其训练代码；本节的有限扰动递推由当前离散SNN方程另行推导，不声称复现EventProp。

**R8：Right Answer, Wrong Mechanism: Detecting Pernicious Divergence in Causal Interventions。** 2026-09预印本，[arXiv入口](https://arxiv.org/abs/2609.39243)。本轮核到检索摘要，全文与官方代码未确认；登记为后续候选，未采用其报告数字或HPC实现。其主题与输出正确不能排除机制失配相关；本文抵消反例和三运行比较有独立定义和验证。

## 9. 固定源码及可重复审计

| 仓库 | 固定commit | 文件数 | 实际执行边界 |
|---|---|---:|---|
| EleutherAI/concept-erasure | 9f51753821316a1edacf78b52b464ab26d40e60a | 3 | 原类、协方差/收缩依赖；人工数据及14个本项目拟合位置 |
| efemeryds/amnesic-probing-with-single-projection | 972d6b9a88b5134a29a35e7a66979c118df54d57 | 2 | MP原定义；LEACE打包文件仅源码审读 |
| brcsomnath/PEF | a0f708b126920da4240ceca8771e50286faf2b7d | 1 | 两个耦合函数，不运行全脚本 |
| Zishan-Shao/decodeshare | 89ed6a36603d81d024b0bd5438048a93d3fff2ac | 2 | 维护中的subspace.py选定hook；README核身份 |
| yanaiela/amnesic_probing | a96e3067a9f9918099015a7173c94830584bbe61 | 1 | 三个投影函数，不训练INLP |

共9个文本文件，Git blob和SHA256写入`references/selected_author_sources.json`。作者全文源码不打包成我们的原创代码；包内提供固定下载器和审计脚本：

```bash
python fetch_pinned_sources_l43.py --directory /absolute/path/author_sources
python audit_author_sources.py --source-root /absolute/path/author_sources \
  --output /absolute/path/author_audit.json
```

仓库快照和依赖函数审计可复现不等于原论文全部训练可复现。本轮没有下载LLM权重、原论文数据或申请GPU。

## 10. 原计划覆盖、复用与独创的归属

| 原计划键 | 覆盖状态 | 我们借用的内容 | 不能算独创的内容 | 我们要形成的增量 |
|---|---|---|---|---|
| CAUSE-AMNESIC | 已覆盖 | 删除后看原行为 | 可读与利用的区分 | 内容×载体×阶段的系统性功能判断 |
| CAUSE-MEANLEACE | 已覆盖 | MP/LEACE比较及随机控制 | 随机损伤/标签恢复本身 | 连续/二值合法性、逐样本剂量及动态后果 |
| CAUSE-RELIABILITY | 论文已覆盖；官方代码未核 | 独立验证与完整性/选择性 | reliability思想及其已有定义 | 多来源、到达条件、原生风险和范围差异 |
| CAUSE-LEACE | 已覆盖且原类执行 | 闭式消除与源码稳定化 | LEACE公式和协方差收缩 | 内容因果解释的数值稳定性审计及预测价值 |
| CAUSE-ERASURE | 已覆盖，选定函数执行 | 消除—效用极限 | 信息论基本限制 | 任务MSE下限、相关因素的可比较前沿 |
| READ-DECODESHARE | 原文/维护源码已覆盖 | 阶段对齐、预算、恢复 | decode-only/能量对照/patchback | SNN事件/状态所有者/实际二值路由与设计归因 |

真正有价值的新结论应是：这些分析在充分训练模型上揭示什么可迁移规律，以及据此提出的改动是否在未见设置兑现。现有邻近方法不妨碍我们形成强系统贡献；同时须让每条结论的来源、适配和验证范围可被审阅。
