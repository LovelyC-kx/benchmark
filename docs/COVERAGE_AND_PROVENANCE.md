# 原研究计划、文献复用和本轮实现对应

研究论证仍以旧Master、各节Deep Review、Literature and Code Map及Mathematical Specification为来源。本轮没有把旧文献综述改成新的已验证论文结果；新增的是共同执行链和明确的数据/模型合同。

## 1. 来源与迁移

`source_migration_provenance.json`逐文件列旧来源与原SHA。`legacy/`恢复完整旧L2的14个模块，保留其任务与接口。`reference/`来自最新L3.2父代码的12单元/8骨架。19份 `kernels/`来自既有L3.3—L4.5数学模块，仅把跨目录隐式import改为相对import；明确的辅助函数以原公式移入。

新模型工厂、持续状态/读出快照、连续参照、计数器数据包装、训练与恢复、真实数据转换、工作队列、数据下载器、公共分析和测试是本轮工程实现。它们服务研究，不能单独作为“首次提出时序保持/概念擦除/ANOVA”的方法新颖性主张。

## 2. 按旧层级逐项定位

| 原层级/子节 | 统一实现 | 复用思想/代码 | 当前边界 |
|---|---|---|---|
| L1任务与数据 | specs/data/protocols/prepare/oracles/readouts | 原16任务、G01—G09、adding/TCN/Zoology参照 | 新生成版本与官方语料分开；额外真实格式不自动伪装 |
| L2状态与骨架 | reference/models/runtime | 原12单元、8骨架、显式状态作用域 | 新增连续controls；官方模型需独立复现合同 |
| L3.1合同 | source_schema、RunSpec、provenance | 目标/来源/角色/权限与单位 | 训练身份和分析身份分开 |
| L3.2读取 | fitted_probes、readouts | 训练/选择/评价分离与负R²保留 | 不是探针失败即可判无信息 |
| L3.3.1统计分解 | kernels/decomposition、geometry | 联合编码、协方差与全部交叉项 | 残差不改名为内禀噪声 |
| L3.3.2几何 | kernels/subspace_geometry | 角色子空间、支撑与Haar参照 | 当前公共runner用数值支撑；旧独立副本/秩选择专用控制仍在原目录 |
| L3.3.3非线性 | ANOVA、非线性探针、functional_anova | grouped Sobol/Hoeffding完整lattice | 公共renderer当前只对tagged独立产品世界做ANOVA |
| L3.4泛化理论 | geometry_generalization/task_spectrum/continuous_risk | Wakhloo式命名几何、跨拟合任务谱、有限p混合矩 | Gaussian/random-teacher与Hebbian参照不是原分类头定理 |
| L3.5载体 | traces、state/message/aux操作 | v/a/syn与完整系统状态的区别 | cache/全部高维状态不被压缩观测偷换 |
| L3.6时间 | temporal_alignment、TGM | 无标签transport、族/秩/方向与预算 | 固定T trajectory权限显式；不规则时钟与模拟步分开 |
| L3.7通信 | message_edit/communication/event_edit | 实际消息、边、合法二值与门控制 | module消费者输入已支持；任意非module分支另需adapter |
| L4.1权限与读出 | access_controls、native risk | 可观测/摘要/读出族/拟合间隙分开 | 经验新读出不组成必然单调理论链 |
| L4.2潜空间 | kernels/latent_theory42＋数学控制 | 线性Gaussian嵌套风险、query可读性、任务度量 | 理论内核可调用；尚未将所有控制自动适配每个深网络 |
| L4.3选择操作 | selective | LEACE/orthogonal、随机秩/剂量、独立验证 | 连续局部状态操作；不是全状态非线性擦除证明 |
| L4.4恢复 | recovery＋recovery_math44 | 四风险、patch-only、Ω、平方风险恒等式 | 当前精确来源世界仅tagged；oracle patch明确记录 |
| L4.5空间恢复 | spatial generators、reconstruction | 区域、频率交叉项、PSNR次序、SAM | 真正空间头、干净目标与配准缺口单列 |

## 3. 与核心同类benchmark的关系

对《Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects》及其NeuroSeqBench，我们保留原生任务和训练/比较的重要性，复用adding等任务规则；新增工作围绕相同检查点中的内容角色、观测/读出权限、状态/消息分离、干预/恢复及可迁移诊断。当前adding的条件平衡采样不等于作者有限过采样语料；MQAR也独立命名为reference。

对《Benchmarking spiking neural networks across sensing modalities on edge devices》及其数据路线，我们保留跨模态分类和实际资源观察；新增受控历史需求、连续重建、表示与机制分析。当前PyTorch状态计数不能替代边缘设备延迟/能耗实验，数据格式统一也不能替代各传感模态的原生任务审核。

对表示/因果领域，线性探针、LEACE、functional ANOVA、Procrustes/affine alignment等思想均有既有来源。我们应争取的贡献是：针对SNN时序任务形成角色明确、阶段明确、权限明确的系统诊断，发现可复现规律，并预测和验证改动怎样影响后续模型；不是给既有公式换名。

## 4. 需要服务器证据才能更新的状态

官方完整语料适配成功、各族公平调参/收敛、真实数据内容重复/空间邻近审核、GPU续跑与吞吐、跨seed一致性、独立确认、oracle诊断转为有限权限可部署改动，都必须在产生实际receipt后更新。当前没有因CPU smoke通过而把这些标成完成。

未来若把旧专用数学控制接入公共runner，应补：适用任务/状态、拟合角色、预测目标、参考假设、已执行检查点SHA、独立数据与未通过情况。旧目录文件存在或旧PDF中出现过某公式，都不能代替这份迁移证据。
