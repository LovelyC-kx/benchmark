# 3.2文献使用与独创性账本

核对日期：2026-10-25。下列区分原文已经做到的内容、我们实际使用的部分、独立新增的实验合同及尚待结果支持的发现。没有把相近工作视为否定本研究价值，也没有把相近工作已有的方法据为独创。

## R1 — Temporal superposition and feature geometry of RNNs under memory demands

Sharma, Proca, Prieto, Mediano；ICLR 2026。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bfcb583d225b1db8d3ca2331f6785774-Abstract-Conference.html)；[PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/bfcb583d225b1db8d3ca2331f6785774-Paper-Conference.pdf)；[作者代码](https://github.com/kashparty/iclr-rnn-superposition)。

本轮读取位置：正文§3.1–3.4、§4相关线性/稀疏非线性讨论及§4.5，Appendix B.2；PDF第4/5页作视觉核对。代码读取Fig.1/5 notebook的数据、BatchedRNN核心、训练与候选选择cell；没有声称42页附录及全部notebook已逐行读完或运行。

使用：年龄编码方向与时空需求区分，作为本节表示组织的理论起点。限制：原数据和读出条件决定其损失分解，不直接作为SNN定理。本节的独立设置是同输入需求配对、等目标量控制、SNN发放/反馈交叉与完整状态权限定位。

代码事实与边界：生成器为稀疏Bernoulli×Uniform，默认可剔除全零序列；核心可分别设线性/ReLU状态与读出，示例可绑定读出；Fig.1/5 cell选择并行候选的末端训练EMA最优者。我们采用独立模型/探针验证划分和自由任务头，不复制该选择流程为benchmark协议。

## R2 — Information Processing Capacity of Dynamical Systems

Dambre, Verstraeten, Schrauwen, Massar；Scientific Reports 2:514，2012。[原文PDF](https://www.nature.com/articles/srep00514.pdf)。

本轮核对其容量定义、线性读出及独立状态方向的维数约束；原专题记录另保留相关定理审计。使用：独立标准化目标和受限线性读出下的容量参照。新生成器、有限样本ridge和恒等式校准为本项目独立实现。没有给非线性读出、固定精度bit容量或任意非平稳SNN建立普遍上界。

## R3 — A refined information processing capacity metric allows an in-depth analysis of memory and nonlinearity trade-offs in neurocomputational systems

Schulte to Brinke, Dick, Duarte, Morrison；Scientific Reports 13:10517，2023。[出版社原文](https://www.nature.com/articles/s41598-023-37604-0)。

使用层次为测量约束：检查编码、噪声与观测对容量评估的影响。本轮没有移植其完整refined-IPC估计器，也不把它等同本节Gaussian线性oracle。实际方法细读记录继续见旧roadmap14.6。

## R4 — Quantifying Memory Utilization with Effective State-Size

Parnichkun等；ICML 2025，PMLR 267。[正式记录](https://proceedings.mlr.press/v267/parnichkun25a.html)。本轮核对正式身份、摘要与线性算子适用范围。

使用：分清状态分配与输入—输出功能性利用，分别记Ncore与Dobs。我们将这个要求落实到当前SNN实际Cell、缓存、读出和实值接口；本节不输出未经适用性验证的硬重置ESS数值。

## R5 — How Linear Attention Remembers

Lee, Park, Kim, Ko；arXiv:2609.33093v1，2026-09-27。[原文](https://arxiv.org/html/2609.33093v1)。本轮读取§2及§3–4相关分析，尤其§4.2。

使用：将负荷、经过距离及后续写入重叠分开，关注内容是否可读取而非只看状态存在。我们为SNN补充发放/反馈路径的交叉和完整状态权限；donor/recipient及通路恢复是后续L4借鉴对象，当前代码不冒充复现全部干预。

## R6 — Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It

Boesch, Wee；arXiv:2609.16183v1，2026-09-14。[原文](https://arxiv.org/html/2609.16183v1)。本轮读取§3–6相关方法、结果与限制。

使用：固定状态预算比较具体因素，将某些失败继续交给训练/课程检验。我们按完整运行边界登记SNN状态；在线卷积/延迟buffer即使不属于某论文定义的recurrent state，也另计缓存。其小规模合成结果不是所有模型的普遍定理，本节没有复现其数值表。

## R7 — Unleashing Temporal Capacity of Spiking Neural Networks through Spatiotemporal Separation

Dong等；arXiv:2512.05472v1，2025-12-05。[原文](https://arxiv.org/html/2512.05472v1)。原项目已有上传15页稿与审计；本轮补核NS/rNS与资源竞争解释位置。

使用：以去状态非单调和时空竞争为直接研究动机。我们把当前、各份有效历史与非目标内容分开计量，检验竞争、工作点、读出及训练等解释。这不是声称STSep没有机制分析，也不是单凭我们的一个反例否定其方法收益。

## R8 — Mechanistic Design and Scaling of Hybrid Architectures

Poli等；ICML 2024，PMLR 235。[正式记录](https://proceedings.mlr.press/v235/poli24a.html)；[MAD代码](https://github.com/athms/mad-lab)。本轮核对正式身份与方法概要，原任务入口沿L1目录。

使用：以小型机制任务和资源约束形成可检验流程，并要求后续用途验证。本节数据与训练代码独立编写，不声称是MAD原始任务复现；“合成benchmark/等状态/设计用途”本身不作为独占新颖性。

## R9 — Working Memory in Recurrent Spiking Neural Networks With Heterogeneous Synaptic Delays

Laurent U Perrinet；arXiv:2604.14096v2，2026-07-11。[v2原文](https://arxiv.org/html/2604.14096v2)。v1与v2题名/参数存在差异，本轮以前者的搜索线索定位后再读取v2方法§2.1–2.3及讨论入口，不混用其数值。

使用：异质延迟说明单数神经元数不足以描述历史存储路径。其固定模式库的线索回忆与我们每条序列新采样内容的在线保持，是不同任务合同。我们借鉴状态/延迟账本问题，不复现该模型，也不将其边缘部署表述当本节实测能耗证据。

## 独创性如何在论文中表述

| 层次 | 现在可以准确说什么 | 结果出来后才能说什么 |
|---|---|---|
| 设置 | 本研究设计了SNN具体的同输入需求配对、等总目标量、完整状态/观察权限及发放/反馈交叉协议 | 是否比已有设置发现更多不可被简单参照解释的现象 |
| 实验 | 我们提供可执行任务、独立内容恢复、已知等价/反例、冻结反馈与清状态接口 | 哪个机制在何种需求/资源区间有稳定效应 |
| 分析 | 按内容、状态位置、原头/探针、年龄和工作点区分候选瓶颈 | 是否发现可复现的条件规律、是否预测留出任务或指导架构 |
| 结果/理论 | 目前有解析/数据/实现校准；没有SNN容量新定理 | 有数据与假设支持后再给发现和适用边界 |

本轮定向检索覆盖了时序叠加、固定状态回忆、SNN记忆/复位的近期入口，不是穷尽性系统综述。以上三项具体增量可作为候选协议贡献；“全球首次”仍需针对最终主张逐项查重，不能从未检出一篇论文直接推出。
