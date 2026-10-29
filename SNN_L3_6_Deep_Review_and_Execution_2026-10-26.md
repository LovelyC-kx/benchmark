> **执行状态更新（2026-10-29）**：本文保留原研究记录；当前代码入口、训练/验收状态、数据清单以[当前状态](docs/CURRENT_STATUS.md)和[执行手册](docs/SERVER_RUNBOOK.md)为准。旧“完成”不自动表示全量训练完成。

# 第三层3.6：跨时间几何、重新编码与保持——深入复审和执行稿

2026-10-27，v0.2收尾版（初稿2026-10-26）。承接3.5，沿用原六层和第三层编号。本文覆盖原3.6计划中的10条文献；新增时间泛化方法、动态流形形变和数值对齐的对照。全文、源码和实际执行深度分别登记。

完整交付入口与逐项清单见SNN_L3_6_Delivery_Guide_2026-10-27.md；正式论文假设、对照、统计、运行与数据规格见SNN_L3_6_Paper_and_Experiment_Specification_2026-10-27.md。本版补齐循环/同lag、相位条件读者、阈值—噪声—观察精度、120项骨架/任务捕获验收，并修复自定义检查点及尺寸合同。

**核心问题：同一份内容在以后还能不能被恢复？原读者失败，是内容减弱、读取规则变化、来源身份错位，还是观察到了不完整的状态？**

English abstract: This section studies temporal access to the same content in a frozen SNN. It separates within-time decoding, fixed-reader transfer, restricted coordinate transport, source identity and event age. A central SNN-specific issue is that a spike-mediated downstream representation can be locally constant while globally informative. We therefore combine finite-scale content perturbations, event-boundary crossing and global decoding. The deliverable includes known-law references, selected author-code checks, a 16-by-16 frozen-model analysis and independently selected readout comparisons. These are development evidence on one old, one-update checkpoint, not a trained architecture benchmark.

## 0. 本节在论文中的位置

3.5展开“看哪个载体”；3.6固定载体后展开“在什么时候、用哪套读取规则看同一内容”；3.7再检查内容是否通过真实接口到达下游。这里同时服务第三层论证A（内容在哪里、是否仍可恢复）和C（自然读取与访问条件）。

主文应组织成三段论证：

1. **时间泛化失败需要分型。** 用同刻读取、固定跨时读取和受限适配分开判断。
2. **几何分析必须适合模型。** 坐标方向、噪声和离散事件边界会改变几何指标的含义。
3. **诊断产生可验证的读取或核心改进。** 读取失配、内容恢复下降和更新时钟问题应对应不同改动，并在留出数据上验收。

我们争取的突出性质是：benchmark能够识别具体失败原因，并预测什么改动有用。时间泛化、Procrustes或局部度量本身都有前人；新的研究价值体现在这些工具如何形成可辨别的SNN问题设置、理论连接、实测规律和设计收益。

七小节不是七条并列的创新。3.6.1–3.6.3建立比较合同和辨别工具；3.6.4是本节最值得深入的SNN特异问题；3.6.5–3.6.6补足时间、噪声和干扰解释；3.6.7检验诊断能否产生设计收益。论文中应围绕“为什么跨时读不出—怎样判断—判断能否指导改进”连成一条论证。

| 争取形成的实质贡献 | 必须回答的检验问题 | 当前证据与下一步 |
|---|---|---|
| 跨时访问障碍的系统辨别 | 同一来源的当前恢复、固定读者与受限适配能否稳定区分失败模式？ | 合同、已知答案和旧模型分析已完成；需充分训练模型与seed确认 |
| 适合SNN的事件/概率几何分析 | 事件跨界、条件均值与单次状态的差别，是否解释平滑度量忽略的内容或任务效用？ | 命题、尺度律和有限SNN诊断已完成；阈值、噪声、观察精度扫描及骨架接口已补齐；正式跨模型预测比较待完成 |
| 诊断驱动的模型/读取改进 | 在未参与选择的数据及相同晚时刻，诊断推荐的改动是否胜过通用调参和相同成本对照？ | 已有冻结读者的留出访问收益；原生任务头和核心机制改动待正式训练 |

这些贡献目标保留充分空间；“方法能运行”“现象在一个旧模型出现”和“发现跨模型规律”分别验收。新发现的价值由可复核的设置、解释和正向设计作用建立。

### 实际执行状态

| 项目 | 本轮实际完成 | 解释范围 |
|---|---|---|
| 主模型 | 旧checkpoint 272ae6230942ebeb，原训练仅1次optimizer更新 | 同一冻结模型；本轮新增SNN训练0 |
| 主分析 | 9种载体观察×5个来源×16训练时刻×16评价时刻×4种映射 | 每种目标46080个风险单元；分类与连续恢复共92160个 |
| 读出 | 每个独立读者128个训练场景，2个共享draw，独特训练场景合计222 | 378次闭式矩阵求解；3060个按内容/目标/读者划分的逻辑拟合 |
| 主评价 | 8192个新独立场景 | 模型、任务、映射和选择冻结后生成 |
| 读者选择 | 6个候选，2048场景独立选择；45个留出配对比较 | 22个同时区间的下界为正；不能解释为记忆延长 |
| 延迟补充 | C写入后1/3/7/11步，6种映射，1024个独立场景 | 同权重、跨deadline观察；另有900个全族比较的重放认证 |
| 来源时钟 | 6种配对schedule，5来源，4载体，3年龄，共360行 | 同内容/噪声，区分来源、时间、年龄和可用性 |
| 有限尺度几何 | H1/C×first/last post×3扰动幅度，共12组 | 固定噪声与其他内容；48次正负扰动捕获；条件噪声另64×8重复 |
| 条件均值几何 | 8内容×128噪声重复，2来源×2尺度×2载体共8组 | 区分逐路径度量与条件均值度量；32次额外捕获，不将重复当独立内容 |
| 测试 | 67项通过，无skip；另完成120个细胞/骨架/分类重建捕获案例 | 结构/数学/因果前缀合同，不是性能排名 |
| v0.2补齐 | 540循环、54同lag、25相位比较、576扫描行；8阶段原检查点和9阶段未训练新配置集成通过 | 新配置为continuous、q=1、20步；其700个延迟比较哈希匹配，不算正式训练 |
| 作者源码 | 8个固定blob逐文件匹配；选定函数执行 | 无作者完整训练、EEG实验或硬件构建 |
| 正式benchmark | 0/435；audit_test未访问 | 正式跨架构、训练seed和真实数据仍待运行 |

没有下载真实数据集，也未使用用户A40。CPU环境Python3.12、NumPy2.3.5、SciPy1.17.0、Torch2.14.1+cpu。

## 3.6.1 首先固定内容身份：时刻、年龄与序列槽位

### 属于什么问题、为什么先做

若两个时刻的标签不再指同一内容，跨时失败不能叫遗忘。流式任务尤其容易把“第几个位置”与“哪个输入项目”混淆。

对来源i，写入时刻为τ_i，分析时刻为t：

\[
\mathrm{age}_i(t)=t-\tau_i,\qquad
\mathrm{available}_i(t)=\mathbf1[t\ge\tau_i]\mathbf1[i\text{确实出现}].
\]

当前取的是当步更新后的状态，所以写入当步已available。负年龄是尚未写入，不是遗忘。不存在的来源不能混入保持曲线。

本模型原训练要求H1与C，其余H2–H4为U；本节将U拆成具体来源以免平均掉年龄差异。以C为focal时，H1是cotarget G；F/G身份随问题变化，物理source ID不变。所有来源各4维，原始20维内容坐标保留。

### 对照文献与适配

Yang与Yu，2026，*Orthogonal-Rotational Dynamics Supports Efficient Encoding and Updating for Streaming Information in Working Memory*，DOI10.1523/JNEUROSCI.2220-24.2025，研究N-back中的项目与序列等级组织。2026-10-27已通过PMC12828889读到模型/训练、共享PCA投影、等级分离指标及W_hh操作和跨轴相关的结果；作者代码和补充文件仍未取得。这项工作按序列等级组织内容，我们进一步跟踪固定项目身份和事件年龄，并用独立场景读取及受限映射检验，而不从低维图直接断言完整内容保持。线上出版为2025-12，卷期为2026-01；文献表按2026卷期标注。

Sharma等ICLR2026把同一输入经历s次递推后的方向写成W_s=A^sB，使“内容是什么”与“内容多老”可以同时表示。本节进一步保留项目来源及写入schedule，避免在随机次序中按槽位误配标签。

### 代码和本轮实验

temporal_contracts.py实现：独立scene ID检查、来源可用性、event-age抽取和rank/source区分。超出记录的年龄直接报错，不用边缘复制补出不存在的记忆。

主分析schedule为H1/H2/H3/H4/C分别在2/5/8/11/12写入，记录0–15步。另有6个schedule固定H1/C、交换H2–H4；同一sample ID下内容和测量噪声逐值一致。来源保持不变，年龄对齐与固定日历读取分别比较。

360条结果保留“固定日历时刻目标是否已出现”字段。若某schedule下H2尚未写入，较差结果只能作为缺失内容控制。即使按年龄对齐，期间其他写入的顺序和干扰压力仍可能不同，不能将所有差异都归为时钟。

**设计用途：** 为后续事件触发读者、可变间隔输入、N-back、视频项目跟踪和多时相恢复提供来源合同。若要部署基于年龄的读者，还需从可见输入标签/时间戳维护年龄状态，不能免费取得分析元数据。

## 3.6.2 三种跨时量：当前可恢复性、固定读取与适配后的读取

### 类别与目的

只要“t训练、s测试”的读者失败，却没有同时检验s时刻独立训练的读者，就属于**可恢复性与读取迁移混淆**。

记r_t为固定接口，g_t为只用训练场景在t拟合的读者：

\[
E_{\rm within}(s)=E\ell(g_s(r_s),Y),\qquad
E_{\rm cross}(t,s)=E\ell(g_t(r_s),Y).
\]

矩阵第一轴为读者训练时刻，第二轴为输入状态的评价时刻。独立序列划分发生在所有时间点之前；同一序列的不同时间点不能分别进入训练和测试。

### 文献及作者代码

King与Dehaene，2014，*Characterizing the dynamics of mental representations: the temporal generalization method*，DOI10.1016/j.tics.2014.01.002，是方法入口。MNE GeneralizingEstimator固定v1.13.2，mne/decoding/search_light.py，blob3b5078649397d0f95b5188a4d6de3771e8d2cfb5。

核读GeneralizingEstimator与_gl_transform/_gl_init_pred/_gl_score。抽取两个路由函数、使用假估计器验证输出scene×source-time×target-time顺序，最大差0。没有安装运行MNE整套神经数据流程。源码某些score文档的维数描述不能替代实际数组检查。

Libby与Buschman，2021，*Rotational dynamics reduce interference between sensory and memory representations*，DOI10.1038/s41593-021-00821-9，其作者代码cross_auc_clf也使用训练时期×测试时期比较。我们保留其交叉读取思想，但不用一个时间窗平均替代全部状态轨迹。

### 适配和预算

代码：temporal_probes.py、run_temporal_review.py。

- 9观察：两层各pre/post/spike、两层各post+spike、两层post联合。自然维数为32或64；联合访问成本单列。
- 每来源16个固定随机二分类方向，同时恢复4维连续内容。
- 所有时刻/载体共享两组128场景训练索引；2个draw合计222独特场景。
- 使用固定ridge λ=.001，含截距。零分数按0.5错误处理。
- 每个view只用一个跨时间共用的RMS尺度；不逐时间/逐通道白化后仍声称原坐标旋转。
- 主模型头仍是原末时刻post回归头；本节新探针的访问与监督预算不同。

### 风险的理论分解

平方损失下，对当前时刻s的Bayes读者m_s(r)=E[Y|r_s=r]：

\[
R(g_t;r_s)=R^*(r_s)+E\|g_t(r_s)-m_s(r_s)\|^2/q.
\]

第一项是该观察接口的不可避免部分，第二项是该固定读者的额外失配。有限样本g_s也不是m_s，因此E_cross−E_within可能为负，不能裁零后当作精确机制贡献。

**当前结果：** 第一层post的H1同刻分类错误，在t=2约.1080，t=8约.3877，t=15约.4922。这个旧模型确实显示指定读者下的恢复下降，不能只讲“全是旋转”。C写入前约.5，写入后t12–15约.2035–.2108；写入前列是必要的无内容参照。

**研究增量：** 将同一内容的时间轨迹、观察权限和有限读者风险放进同一合同；既能检出读取失配，也保留真实的恢复下降。

## 3.6.3 对齐到底消除了什么：平移、增益、正交、反射和一般线性变化

### 先识别类别

对齐后改善只说明与该映射族兼容。不同映射族允许不同自由度；将它们统一叫“旋转”会丢掉机制信息。

所有映射使用独立、成对的校准场景拟合，方向为评价时刻s→源读者时刻t：

\[
\widehat r_t=(r_s-\mu_s)M_{s\to t}+\mu_t.
\]

| 映射 | 可以改变什么 | 解释 |
|---|---|---|
| raw | 不变 | 原读者的直接迁移 |
| center | 平移 | 平均活动/基线漂移 |
| O(D) | 正交方向，包括det=-1反射 | 长度和角度保持的坐标兼容 |
| SO(D) | det=+1纯旋转 | 与反射区分 |
| similarity | 正交加非负全局增益 | 可分离整体幅度变化 |
| affine ridge | 一般正则线性矩阵加平移 | 可以包含剪切和各向异性缩放，复杂度更高 |

主16×16矩阵使用raw/center/O(D)/affine四族；延迟补充使用六族。SO与similarity在已知答案和测试中也有单独控制。自对齐对角线固定恒等，不额外引入收缩处理器。

代码：temporal_alignment.py，fit_transport/transport/push_reader。push_reader将映射与冻结读者合成，核对其与显式先映射后读取逐值相同。跨时间比较不是将两个头的权重直接相减。

### 两个源码要点

SciPy1.17.0的orthogonal_procrustes返回的scale是AᵀB的奇异值之和，**不是similarity的乘数**。本轮例子返回约175.88，而将2X变回X的增益应为.5。源码固定blob e3584b85a0be556b1d43db88ba51447e9d62f1fc。

Procrustes的SVD解也可能有det=-1。反射控制中，普通正交对齐MSE约5.6e−33，纯旋转约1.99；子空间主角余弦却全部为1。**子空间相同不等于项目方向或标签朝向相同。**

### 可识别性、循环和混杂

校准交叉协方差秩不足时，零空间中的正交补全没有唯一解释。代码报告秩、维数、det和奇异值；少活跃脉冲时尤其不能给任意补全赋予生理意义。

对同一坐标变化，Q_ab Q_bc应与Q_ac兼容；cycle_error可检查已知旋转，不能仅靠成对拟合都不错就推断一个统一动力学。v0.2已完成真实冻结SNN的含截距循环及同lag映射迁移扫描；结果是受限映射描述的诊断，未假称已经建立其生成元。

还构造了总体几何与任务方向冲突的例子：整体状态MSE由49.758降至49.269，但原本完美的目标读者错误从0升至约.1479。校准使用与Gaussian总体前两矩完全一致的四点参照，评价用独立Gaussian样本。

这说明“状态重建更好”不能代替“目标读取更好”。它也给出正向设计规则：对齐需要按目标验收，并保留不适配的候选，避免不必要处理。

**相近与推进：** 受限对齐是已有工具；本节把具体映射自由度、秩、目标风险、来源身份和成本一起交付，形成可判别的跨时访问实验。

### v0.2补齐：含截距循环与同lag迁移

行向量映射写作T_c→b(x)=xW_cb+b_cb。连续应用c→b→a后，矩阵是W_cb W_ba，偏置是b_cb W_ba+b_ba；不能只乘矩阵而忽略均值。run_temporal_consistency.py冻结所有配对映射后，才生成2,048个新场景（development_test/7700000）。

9载体×20个时间三元组×3映射族得到540条循环记录，另有54条同lag迁移。每族180条循环中只有40条的三条交叉协方差全满秩。O(D)的留出预测差异中位数约.09217，affine约.001210；秩不足处不据全维矩阵补全推断真实旋转。

**center循环为零只是代数控制。** 三个时刻的均值取自同一校准池时，均值差自然相消；它既不要求保留目标，也不要求真实动力学是平移。已知旋转控制还显示：变速旋转的循环差约1.5e-31，但相邻等长间隔的矩阵差约.83586。因此循环一致不能代替时间推进规律检验。

同lag检查把t12→13的拟合映射应用到t13→14、t14→15，并与本地拟合映射在留出状态上比较。更一般的失败解释包括非线性、噪声、校准不足和状态分布变化；不能把受限线性拟合的时刻差异说成底层神经元参数时变。代码/完整结果：temporal_alignment.py、evidence/temporal_consistency。

## 3.6.4 内容特异几何：从稳定子空间到离散事件边界

### 为什么全体PCA或一个角度不够

高方差方向可能主要是时间、输入背景或其他项目。Murray等，2017，*Stable population coding for working memory coexists with heterogeneous neural dynamics in prefrontal cortex*，PNAS，指出单元动态与稳定群体编码可以共存。当前重读作者提供PDF的主要结果和方法段；官方分析代码入口未核实，不冒充其复现。

因此同时保留：固定头、每时刻头、共享时间读者和目标条件几何。方向的稳定需要与目标标签关联，不能从总活动轨迹稳定直接推出。

### 2025动态流形工作怎样用

Pellegrino与Chadwick，NeurIPS2025，*RNNs perform task computations by dynamically warping neural representations*，正式论文0572c069404875ccca76e822aaf48d50。其核心是从输入参数κ到状态x的映射及拉回度量：

\[
J_t=\partial x_t/\partial\kappa,\qquad G_t=J_t^TJ_t.
\]

本轮核读主文§2–3及相关公式，正式代码包URL尚未核实。我们复用内容参数化和局部几何的思路，但必须适配SNN的离散发放/重置；平滑ODE假设不能直接覆盖这些边界。

若噪声为常协方差Gaussian，位置族信息矩阵为：

\[
G_{\rm noise}=J^T\Sigma^{-1}J.
\]

可逆线性坐标变换同时作用于J和Σ时，该量不变；Euclidean G通常会变。当前SNN用经验条件噪声和正则逆得到的是**噪声归一化敏感性**，没有称为精确Fisher信息。

### SNN的关键命题：局部不变仍然可以全局有信息

设有限时间内的上游事件序列为E(z)，下游状态为S=F(E(z),S_0)，没有另外的模拟输入路径。只要某个邻域中E(z)不变，该邻域的下游状态就不变：

\[
\partial S/\partial z=0.
\]

但不同邻域可对应不同脉冲模式，因此S仍可区分内容。最小例子f(z)=1[z≥0]能够完美解码sign(z)，普通导数却几乎处处为0。

对Z~N(0,1)和中心差分：

\[
D_\delta f(Z)=\frac{f(Z+\delta)-f(Z-\delta)}{2\delta},
\]

\[
P(\text{事件跨界})=2\Phi(\delta)-1,
\qquad
E[(D_\delta f)^2]=\frac{2\Phi(\delta)-1}{4\delta^2}
\sim\frac{1}{2\sqrt{2\pi}\delta}.
\]

δ变小时，跨界事件更少；但其差分幅度更大，平均平方甚至发散。故“导数几乎处处零”“有限样本小扰动度量为零”“有限尺度平均很大”和“内容可读”可以共存。

这是标准量化映射性质在本研究中的推导和应用，不宣称量化不连续性首次被发现。它明确指出一种需要SNN适配的测量缺口。

### 从标量阈值推广到内容空间的边界积分

还可将尺度规律写成可用于后续理论分析的形式。设f在内容空间中分片常值，事件分区边界Σ局部光滑；密度p在边界连续，跳变量Δf有界。对单位扰动方向u，在边界交叉处不发生正测度退化、且可以交换积分极限的条件下：

\[
E\|D_{\delta,u}f(Z)\|^2
=\frac{1}{2\delta}\int_\Sigma p(z)|u^Tn(z)|\,\|\Delta f(z)\|^2\,dS(z)
+o(\delta^{-1}).
\]

理由是：正负扰动分处边界两侧的薄层厚度约为2δ|uᵀn|；跨界差分平方约为‖Δf‖²/(4δ²)，相乘后得到上式。分片平滑且有跳变时，该式仍描述非零跳变贡献的主导项，区域内部还存在平滑变化项；不能把它全称为零导数系统。

它给出三个可分因素：**有多少内容位于边界附近、扰动方向怎样穿过边界、跨界后状态跳动多大。** 因而大有限尺度度量既可能来自内容分辨，也可能来自少量巨大跳变，仍需目标读取验收。标量Gaussian阈值是该式的特例。本轮执行了标量精确参照和SNN有限尺度扫描；真实SNN的边界面积/法向估计尚未实现，不将此一般式当作已拟合的SNN定律，也不声称边界积分数学本身首次提出。

### 已实现的适配

run_temporal_extensions.py固定其他内容、噪声和schedule，对H1/C四坐标施加δ=.001/.01/.1的正负扰动。分别保存：

1. 有限尺度状态变化及G_δ；
2. 两种扰动之间的发放模式变化率；
3. 逐时间的内容敏感性、噪声归一化敏感性；
4. 与独立场景上的全局读出结果并列。

全部来源写入前的敏感性严格为0。当前C、δ=.001/.01时，所测64场景末层post度量为0；δ=.1跨过部分发放边界后变为非零。同一模型在独立场景上的末层C分类错误约.43，说明局部零敏感性不能单独等同于没有内容。

SG梯度也不作为真实前向局部度量：实际Spike在稳定阈值一侧的微小差分为0，而训练用surrogate backward可以非零。测试单独验证了这一差别。

### 必须再区分：逐路径几何与条件均值几何

上面的零导数指固定噪声实现的单次前向路径。若重复噪声并取条件均值，情况不同。对f(z,η)=1[z+η≥0]、η~N(0,σ²)：

\[
m(z)=E_\eta f(z,\eta)=\Phi(z/\sigma),\qquad
m'(z)=\varphi(z/\sigma)/\sigma,
\]

\[
I_{\rm Bernoulli}(z)=\frac{[m'(z)]^2}{m(z)(1-m(z))}.
\]

这是真实Bernoulli观测模型的分布信息；在z=0时为2/(πσ²)，虽然逐路径导数几乎处处为0。因而不能从单次脉冲的局部不连续，否定试次均值/概率编码的几何。比较神经科学PSTH或平均发放率与SNN单次状态时，尤其需要注明观察对象。

本轮追加run_mean_geometry.py：固定8份内容，各进行128个噪声重复，采用正负扰动共享噪声的方式，同时计算：

\[
G_{\rm mean}=\bar J_\delta\bar J_\delta^T,
\qquad
G_{\rm path}=R^{-1}\sum_rJ_{\delta,r}J_{\delta,r}^T.
\]

这里为配合代码，J_δ每行对应一个内容坐标，维数为q×D；与上文D×q的列切向约定互为转置。两者之差是有限样本Jacobian变化的协方差，半正定；已在8组SNN诊断中验证。J_δ始终是有限尺度差分。128重复给予了额外观察权限，不能伪装成单次、低延迟的部署接口，也不是1024个独立内容。

因此正式实验同时登记单次路径、条件均值和概率模型；根据被测对象选择度量，再与全局恢复、事件跨界及成本联系。这个区分使SNN特异分析更完整，也保留了将神经科学方法迁移进来的正向价值。

**论文价值与改进方向：** 建立适合离散时序系统的内容变化分析；比较编码的局部稳定、跨事件变化与全局可利用性，再检验阈值、辅助路径和读出设计怎样改变这些关系。正式实验将检验它是否比单一平滑度量更好解释跨模型结果。

### v0.2补齐：事件尺度、阈值、噪声与观察量化

run_event_scale_scan.py固定旧检查点，扫描threshold×{.8,1,1.2}、measurement sigma∈{0,.2,.5}、δ∈{.001,.01,.1,.3}以及浮点/2/4/8 bit导出观察。两来源、两post载体共576行描述统计，585次前向捕获，128个相同底层场景。阈值变体没有重新训练；量化使用原校准RMS的±4倍范围、有限级和饱和，内部算术保持浮点。

扫描保留整个历史的事件跨界、非零场景数、迹分位数和最大值。当前last_post/C在σ=.2、原阈值、δ=.001时，场景迹中位数、95%及99%分位均为0，但平均约9907.95。极少事件跨界和δ平方分母即可造成这种均方；它不是“内容信息极高”的证据。末时刻事件相同也不代表之前没有跨界。first_post仍可在固定事件模式内直接随原输入变化，不能误用“任意SNN状态的局部导数都为零”的说法。

实际字段、量化饱和比例及所有失败/变化均保留在evidence/event_scale_scan/report.json。这是对离散几何解释的补足，未把128场景的有限差分平均称为真实边界面积估计或普适幂律发现。

## 3.6.5 时间结构与更新：输入时钟不等于内部状态时钟

### 对照两类2025/2026工作

Qin等，2026 eLife reviewed preprint112013.1，*Rhythmic temporal structure organizes recurrent dynamics to support sequential working memory*，结合E/I RNN、短时突触状态和规则/抖动输入。其评审明确将当前证据评为incomplete。本文可用的是其具体方法和待验证机制，不将评审状态当作全部结论的否定。

固定作者commit3ff8416e0570bfd466a51fbefaaecbec2890f8df，审阅EIRNN.py和align.py选定内容。该模型支持显式传入hidden、syn_x、syn_u；分段连续调用需一起传递这些状态。其阶段依赖的重采样适合离线图示，不自动成为在线操作。

Yan等，Nature Communications2025，*Efficient and robust temporal processing with neural oscillations modulated spiking neural networks*，DOI10.1038/s41467-025-63771-x，提出周期门控的Rhythm-SNN。固定commit33c5e29ac2b4715cf7ddde4d9e0b891be32f12d8，核读选定更新和mask构造。

### 源码核查带来的适配

- 膜状态OFF时可冻结，但选定ALIF函数的a仍更新。CPU设备适配后的抽取检查中，v保持.2，a从.5变为约.45242。这是状态时钟范围，不直接判为作者错误。
- rhythmic_step明确提供“全部状态冻结”和“仅膜冻结”两种政策；振荡相位/步数属于运行状态。
- align.py的_slice_with_pad声称总返回指定长度，但对完全落在记录之外的窗口，本轮最小例子请求2、返回4。不能据此推断论文用了该窗口；我们的event-age读取直接拒绝越界。
- 原文相位分析明确使用因果复滤波；同时某些可视化用区间线性warping，二者权限不同。我们实现固定频带的过去依赖复滤波，并验证改变未来后缀不会改变过去相位信号。

### 我们测什么

时间轴分为：外部步数、实际输入事件数、项目年龄、每通道更新次数、内部振荡相位。当前完成年龄/schedule/blank-deadline及phase-conditioned读者扫描；完整Rhythm-SNN训练仍属正式机制扩展，不把我们自己的相位读者说成作者模型复现。

空白外部输入也不保证内部没有驱动：线性层bias、循环状态和辅助更新仍可能改变神经元。不能仅由输入为0就把所有后续变化解释成纯指数泄漏。

**设计用途：** 选择事件触发或周期更新策略时，同时检查存储和访问时钟；验证收益来自减少覆盖、改变读出相位，还是较短的梯度路径。梯度路径与前向内容保持仍分别测量。

### v0.2补齐：相位条件读者的独立选择与评价

phase_readers.py将状态扩展成[r,r cosφ,r sinφ,cosφ,sinφ,valid]，维数由D增至3D+3；相位无效时三个时钟项及交互项为0。六候选为pooled、calendar、source_age、input_phase、state_phase、phase_only；频率在所有池之前固定为1/8 cycles/step。

每个读者用128个独立标签，schedule0/1/2的可用时间只作为重复训练行；1,024个新场景在schedule3选择；冻结之后以2,048个新场景的schedule4/5平均风险评价。两schedule不是4,096个独立场景。5载体×5来源为25个同时比较，结果下界均未大于0。留出负增益保留，不能宣称相位条件读取在当前旧SNN上已经更优。

已知当前相位的旋转控制中，普通线性读者MSE约.99189，条件读者约.0001047，说明实现可处理这一已知访问问题。该控制的相位是额外给定观测，不是从未来或标签中估出的神经振荡。

input_phase使用输入事件标志的过去依赖24步滤波；state_phase使用载体群体均值的历史滤波，是额外历史非线性观察；source_age需要来源写入台账。三个权限与缓冲、参数成本不同，不能混在一起归功于某种细胞。详见evidence/phase_readers/freeze_contract.json和正式实验规格。

## 3.6.6 噪声、年龄与干扰：什么时候是真正的恢复瓶颈

### 一个可以精确算的参照

令行向量状态：

\[
r_t=a_t zQ_t+\sigma_t\epsilon_t,
\quad z,\epsilon\sim N(0,I),\quad Q_tQ_t^T=I.
\]

约定a_t≥0、σ_t≥0且二者不同时为0；符号方向归入Q_t。否则二分类Bayes式中的相关系数需要相应处理。

任意单位方向二分类与连续恢复的Bayes风险分别为：

\[
R^*_{\rm MSE}(t)=\frac{\sigma_t^2}{a_t^2+\sigma_t^2},
\quad
R^*_{01}(t)=\frac1\pi\arccos\frac{a_t}{\sqrt{a_t^2+\sigma_t^2}}.
\]

若源读者是其总体最优连续读者，α_t=a_t/(a_t²+σ_t²)，则固定源读者跨到s的精确风险为：

\[
R(t\to s)=1+\alpha_t^2(a_s^2+\sigma_s^2)
-\frac{2\alpha_ta_s}{d}\operatorname{tr}(Q_sQ_t^T).
\]

该式把读取朝向、增益和噪声分开；它减去当前Bayes风险得到非负失配项。代码也保留了此精确参照。

朝向变化影响固定读者，a/σ影响当前可恢复性。无噪声且a>0时，幅度再小也不必丢失Bayes信息；有限精度或固定读者可能仍失败。

当前独立参照对a=1,σ=.5得到MSE约.1995、二分类错误约.1475，理论值.2/.14758；a=.05、σ=0仍可无误恢复。它们是已知分布控制，不能直接当作SNN Bayes误差估计。

Ritter与Chadwick，2026 eLife reviewed preprint111445.1，*Efficient Working Memory Maintenance via High-Dimensional Rotational Dynamics*，使用线性Gaussian动态、均值/协方差及噪声/能量目标。当前核读主结果、Methods和有关补充段，官方代码仓库URL未核实；不能把本项目线性参照称为其优化网络复现。

我们实现连续线性系统的精确均值/协方差参照和Gaussian信息矩阵，以便把动态导致的噪声积累与纯坐标旋转分开。旋转会不会更优需要在指定动态、噪声输入和成本条件下比较，不能从瞬时几何角度直接推出。

### 时序叠加怎样接入

Sharma等，ICLR2026，*Temporal superposition and feature geometry of RNNs under memory demands*，正式作者仓库kashparty/iclr-rnn-superposition，固定commit26b00cbb9cc03bcb2b4f9d989ac3a9d48a5f6bf9。

核读原文§3–4及相关损失推导，取得Fig2/6 notebook，审阅数据、模型、损失和训练/选择代码。抽取四项损失函数与我们的独立矩公式比较，最大差2.78e−17，未执行作者训练。

对标量延迟读出系数a_s=C A^sB，记c_s=a_s−1[s=k]、输入均值m和完整时间协方差K，则有限时间的精确损失为：

\[
E(\hat y-y)^2=c^TKc+(c^Tm)^2.
\]

这里有限输入向量必须包含被恢复的目标分量。若目标尚未出现，当前合同将其单列为不可用来源；不能把未写入目标的零值约定混进遗忘曲线。代码对超出有限向量的target_age使用“该有限窗口内目标为0”的参照约定，正式保持分析只采用available时刻。

独立Bernoulli门控输入X=BU时，可以展开为任务项、投影干扰、均值修正与组合干扰。源码的ν取E[U²]，Uniform[0,1]为1/3，不能误用方差1/12。

本轮独立输入例子：四项之和.117514，与一般矩公式逐值一致，Monte Carlo约.118299。加入时间相关后，真实精确值约.224594；错误套独立公式会给.306071。我们的连续视频/自然信号适配因此保留完整时间协方差。

还必须检查可达和可观察模式：A=diag(.5,2)、B=(1,0)ᵀ、C=(1,0)，全谱半径2，但输入输出核仍为(.5)^s。复用谱半径结论时，应检验参与任务的模式，不能把所有隐藏模式等同于有效记忆。

### 完整状态的信息方向

若完整状态的未来只由当前状态和目标无关的创新噪声产生，且无新的目标相关输入，则数据处理给出I(Y;S_s)≤I(Y;S_t)。对任意已知未来输入U，更一般可比较I(Y;S_s|U)≤I(Y;S_t|U)，前提是给定S_t,U后后续机制不另取目标信息。

自然视频未来帧可能包含目标信息或解开旧混杂，不能无条件套“未来信息只能减少”。局部膜观察也不等于完整状态，局部可读性增加可以来自辅助状态迁移。

**设计用途：** 分别改变保持时长、后续写入次数、时间相关性、噪声、精度及参与任务的模式，再判断应该改衰减、输出接口、辅助状态还是干扰抑制。

## 3.6.7 让诊断导出可验证的设计增益

### 已完成的留出选择

6候选为末时刻固定头的raw/center/O/affine读取、跨时间共享头、逐时刻头。所有候选都包含在选择前，λ和p固定。2048独立场景选择每个view/source的一个方案；之后才生成8192确认场景。

跨时间共享头使用相同128个独特训练场景，在该内容available的时刻重复监督；不能把128×时刻数写成那么多独立标签。逐时刻头需要更多参数和时钟访问，两者均使用额外训练时刻的状态。

45个主配对比较中，22个同时95%区间下界为正。第一层post/H1的available-time平均错误，从末时刻头的.48842降为逐时刻头的.33010，降低约15.83个百分点，区间约[11.55,20.11]个百分点。

**这个增益不等于延长了记忆。** 平均包含H1刚写入的早期时刻；其末时刻同刻错误仍约.4922。它验证了读取合同，并说明不能用末时刻训练头评判所有早期状态。真正记忆延长必须在相同晚时刻、相同访问和预算下验证。

### 延迟补充得到的访问增益

使用同一原t15头，在新的成对校准状态上拟合t13/15/19/23→t15的映射，再评价新1024场景。t23的第一层post/C，原头错误约.46942，affine适配约.25909。t15的读取完全不变。

全900个既有比较同时构造区间，其中15个下界为正；上述first_post/C/t23的降低约21.03个百分点，区间约[6.72,35.35]个百分点。避免只给最大收益做单项区间；认证保留在evidence/delay_reader_certificates.json。它是已有场景/冻结读者的重放，不是新模型确认；与主45比较族各自给出95%，没有合称整个研究95%。

这一例子支持“指定线性适配能改善晚时刻访问”，尚不唯一定位成纯增益、旋转、内容生成或细胞优越。增加的512成对校准场景、矩阵/均值存储、时间索引和计算全部计费。

### 成本怎样落到实现

对D维接口，当前显式映射保存D²+2D个浮点数；D=32时为1088个，若用float32约4352字节/映射，不含校准缓存和读者。直接应用密集矩阵每次约D²次乘加，另有均值操作。这些是解析运算/存储计数，尚非设备实测功耗。

映射可以离线折叠进线性读者。若原读者为rW+b，组合后为：

\[
W'=MW,\qquad b'=b+\mu_tW-\mu_sMW.
\]

此时每个deadline保存Dk+k个读者参数，在线仍是Dk级乘加，但需要额外的时间索引、分时参数存储与512场景校准，不能称零成本。代码push_reader已验证与显式处理相同；尚未做设备速度/能量验证。只有一个deadline时，也不应保存多时刻冗余映射。raw/center、直接重训同预算读者和折叠适配应构成正式成本—收益对照。

### 如何转成正式设计实验

| 诊断 | 改动候选 | 最关键的验收 |
|---|---|---|
| 同刻好、固定跨时差、center有效 | 基线校准或读出截距适配 | 同晚时刻、同标签与访问 |
| O有效、方向/循环一致 | 受限时间条件读者或稳定投影 | 保留秩/反射/整体增益对照 |
| affine比O好 | 增益、各向异性或线性剪切适配 | 独立留出及复杂度控制；不直接称旋转 |
| 所测同刻读者均下降 | 核心保持/选择性写入或辅助状态 | 训练充分性、多个载体和噪声/年龄匹配 |
| 小尺度零、事件跨界后才变化 | 面向事件的接口/敏感性分析 | 与全局恢复、阈值及真实任务效用联合验证 |
| 节律收益依赖辅助时钟 | 明确aux更新策略、phase读出 | 同更新次数、功耗、梯度与前向控制 |
| U保留挤占目标或反之 | 任务条件选择、更新门控 | 目标收益与cotarget/U损伤同时报告 |

正式实验先用原435运行计划中的代表模型，不把本节控制计作已完成训练。再挑匹配K/L/width/delay和种子的模型开展3.6，最终迁移到分类及连续重建的原生任务头。图像状态保持完整空间轴；高维对齐需单列投影和缓冲成本。

## 4. 每篇原计划文献的覆盖与用途

| 原计划ID/文献 | 本轮采用内容 | 阅读/代码状态 |
|---|---|---|
| MEM-MURRAY／Stable population coding…，2017 | 稳定群体读者与单元动态并存 | 作者PDF选定结果/方法；官方分析代码未核实 |
| MEM-WOLFF／Dynamic hidden states…，2017 | 自然读出与目标盲探测分开 | 沿3.5审阅及固定作者代码记录，不虚增本轮执行 |
| INT-ROTATION／Rotational dynamics reduce interference…，2021 | 感觉—记忆编码、cross-time读取和干扰对照 | 选定原文与两个作者源码文件；未跑原神经实验 |
| INT-STREAMROT／Orthogonal-Rotational Dynamics…，2026 | 项目来源与序列等级区分 | 本次已读PMC方法/结果；补充文件和官方代码仍缺口 |
| GEO-TEMPORALSUPER／Temporal superposition…，ICLR2026 | 年龄方向、有限延迟损失、输入相关性 | 主文和相关附录、选定notebook；四项函数实际核对 |
| REC-RHYTHM／Rhythmic temporal structure…，2026 | 物理节律、事件warping、神经/突触载体 | eLife相关Results/Methods、EIRNN与align代码；reviewed preprint/incomplete |
| MECH-OSCILLATION／Efficient and robust temporal processing…，2025 | 周期状态更新与读出时钟 | 原文相关段、选定作者函数和mask；未跑完整模型 |
| CAND-OPERATIONS／Working memory operations emerge…，2026 | 保留优先级、更新和形状变化的候选接口 | 核实DOI/作者/摘要元数据；全文/代码未审计，不据题名补发现 |
| CAND-HIGHROT／Efficient Working Memory Maintenance…，2026 | 噪声协方差与时间几何的联合参照 | eLife相关Methods/补充；official代码URL未核实；incomplete评审状态保留 |
| MEM-REACTIVATION／Reactivating and reorganizing…，2025 | 再组织、噪声与可访问性竞争解释 | 沿3.5缺口/OSF记录，不伪称新取得完整代码 |

新增King/Dehaene2014、Pellegrino/Chadwick2025及SciPy源码，对应TGM合同、内容度量和对齐语义。原计划未取得的全文/代码不会阻止独立方法和控制实现，但不用于宣称已复现其机制。

## 5. 原文、代码与版本入口

| 工作 | 原文入口 | 代码入口 |
|---|---|---|
| 时间泛化 | https://doi.org/10.1016/j.tics.2014.01.002 | https://github.com/mne-tools/mne-python/blob/v1.13.2/mne/decoding/search_light.py |
| 稳定群体编码 | https://johndmurray.org/papers/murray_2017_pnas.pdf | 本轮官方代码未核实 |
| 旋转与干扰 | https://doi.org/10.1038/s41593-021-00821-9 | https://github.com/buschman-lab/RotationalDynamics |
| 流式正交旋转 | https://doi.org/10.1523/JNEUROSCI.2220-24.2025 | 本轮已读PMC方法/结果，官方代码未核实 |
| 时间叠加 | https://proceedings.iclr.cc/paper_files/paper/2026/hash/bfcb583d225b1db8d3ca2331f6785774-Abstract-Conference.html | https://github.com/kashparty/iclr-rnn-superposition |
| 节律与序列记忆 | https://elifesciences.org/reviewed-preprints/112013 | https://github.com/Bold2023/rhythmic_working_memory |
| Rhythm-SNN | https://doi.org/10.1038/s41467-025-63771-x | https://github.com/YinsongYan/Rhythm-SNN |
| 高维旋转维护 | https://elifesciences.org/reviewed-preprints/111445 | 文中称GitHub，但实际仓库URL尚未核实 |
| 动态形变 | https://proceedings.neurips.cc/paper_files/paper/2025/hash/0572c069404875ccca76e822aaf48d50-Abstract-Conference.html | 官方代码URL尚未核实 |
| 子空间操作候选 | https://doi.org/10.64898/2026.08.31.748237 | 全文/代码未取得 |
| 数值正交对齐 | https://github.com/scipy/scipy/blob/v1.17.0/scipy/linalg/_procrustes.py | 同左，blob及执行范围见manifest |

references/selected_source_manifest.json逐项保存8个文件的完整commit/tag、blob、路径和阅读深度。作者原文件不冒充本项目实现，未把整仓和预训练权重/数据一起打包。

## 6. 与两篇核心benchmark的关系

Ma等的Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects，arXiv2502.09449，提供时序通路/梯度和任务比较框架。Du等的Benchmarking spiking neural networks across sensing modalities on edge devices，arXiv2609.00026，提供跨模态、模型与部署比较背景。沿已核原文/审计范围进行对照，不据摘要推断它们完全没有几何或机制分析。

本节的明确比较单位是**同模型、同内容、同载体、不同时间与读取规则**。新增输出是：时间/来源合同、受限映射的解释、离散事件尺度问题、噪声/输入相关性的理论参照，以及可收费的读取改进。最终与L2/L4操作和原生任务验证结合，形成系统性时序能力判断与归因。

## 7. 代码阅读与复现

| 文件 | 本节功能 |
|---|---|
| temporal_contracts.py | 独立场景、来源/年龄/可用性、因果前缀 |
| temporal_capture.py | 时间载体导出；模拟输出不伪称spike |
| temporal_probes.py | 同刻/共享读者、场景损失、同时配对界 |
| temporal_alignment.py | 六族映射、可识别性、组合读者、主角/循环 |
| temporal_theory.py | Gaussian风险、任意时间协方差、可观察模式、连续噪声矩、事件尺度律 |
| temporal_clocks.py | 过去依赖相位、明确辅助状态时钟 |
| run_temporal_consistency.py | 含截距循环、同lag迁移及已知反例 |
| phase_readers.py、run_phase_readers.py | 六类因果/侧信息条件读者，独立选择与留出 |
| run_event_scale_scan.py | 阈值、噪声、δ和有限级观察量化 |
| run_model_interfaces.py、named_temporal_bank | 96分类+24原生图像重建捕获合同 |
| l36_suite.py | 带阶段journal的一键流水线 |
| retrieve_pinned_author_sources.py | 获取并Git blob核验8个选定作者文件 |
| run_temporal_controls.py | 已知答案控制 |
| run_temporal_review.py | 16×16主分析和冻结/独立选择 |
| run_temporal_extensions.py | 延迟、来源schedule和有限尺度几何 |
| run_mean_geometry.py | 单次路径与128重复条件均值的有限尺度几何 |
| certify_delay_readers.py | 重放全部900个延迟比较，构造同时界 |
| audit_temporal_sources.py | 8个固定源码及选定函数检查 |
| verify_l36_artifacts.py | 只读校验与冻结读者重放 |
| plot_l36.py | PNG/SVG科学图 |

运行目录中包含完整代码依赖和原checkpoint，默认不需旧3.5压缩包或作者训练环境。

~~~bash
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -v
python run_temporal_controls.py --output outputs/controls.json
python run_temporal_review.py --output outputs/temporal
python run_temporal_extensions.py outputs/temporal --output outputs/extensions
python run_temporal_consistency.py outputs/temporal --output outputs/consistency
python run_phase_readers.py outputs/temporal --output outputs/phase
python run_event_scale_scan.py outputs/temporal --output outputs/event_scale
python run_model_interfaces.py --output outputs/interfaces.json
python run_mean_geometry.py --output outputs/mean_geometry.json
python certify_delay_readers.py --primary outputs/temporal --extensions outputs/extensions --output outputs/delay_ci.json
python plot_l36.py
python verify_l36_artifacts.py
~~~

一键入口：python l36_suite.py --run-directory YOUR_RUN_DIRECTORY --output outputs/l36。主/补充/条件均值/认证等模型分析都显式传播同一run-directory。certify_delay_readers.py可用上述参数认证新输出；自定义模型须加--run-directory并通过与主冻结合同相同的哈希。不提供参数时读取包内证据。默认图表/验收读取包内已存证据。输出目录/文件默认不得已存在，防止无意覆盖已冻结记录。主全矩阵在本次CPU约1182秒；这是本次实现的运行记录，不是硬件速度benchmark。

### 池与权限

| 用途 | 样本 | split/首ID | 可见信息 |
|---|---:|---|---|
| 成对校准 | 512 | geometry_fit/7600000 | 状态、时刻与配对ID；映射不看目标标签 |
| 读者训练 | 512池，读者取128 | probe_train/7610000 | 相同内容标签；2draw共享所有接口/时间 |
| 选择 | 2048 | probe_val/7620000 | 六候选分类风险 |
| 主确认 | 8192 | development_test/7630000 | 只评价冻结方案 |
| schedule补充 | 1024×6配对 | development_test/7640000 | 来源/年龄分析，不是6个独立模型 |
| 长deadline校准 | 512 | geometry_fit/7650000 | 额外配对状态，成本登记 |
| 长deadline评价 | 1024 | development_test/7660000 | 冻结映射与原头 |
| 内容扰动 | 64 | geometry_fit/7670000 | 原始潜变量的离线诊断权限 |
| 条件噪声 | 64×8 | geometry_fit/7670100 | 全内容与schedule固定，仅测量噪声变化 |
| 条件均值几何 | 8×128 | geometry_fit/7690000 | 多次噪声重复；额外试次观察，不能当单次状态 |

## 8. 还需正式完成的实验与论文证据

1. 从正式435计划训练结果取代表细胞、骨架和独立seed；本轮120项覆盖只是时间导出/任务接口合同，不是训练覆盖。
2. 匹配项目年龄、干扰写入数、噪声和观察权限，检验同刻恢复下降与跨时失配的稳定规律。
3. 当前δ、阈值、观察bit精度和测量噪声扫描已执行；正式模型上进一步检验其对读出/任务效用的解释和预测，并与内部算术量化、过程噪声严格区分。
4. 在同一个晚时刻、同一访问和参数/校准预算下测试读出适配与核心机制改动，报告失败与非目标损伤。
5. 建立真实数据与分类/重建任务的对应；时间点不是独立训练seed，任务方向不是独立模型。
6. 3.7继续验证内容到达与实际接口，L4再做选择性干预/恢复；单靠本节TGM和对齐不推断唯一因果路径。

本节保留进取的研究目标：发现可迁移、能指导后来者改变模型的关系。当前交付使这些关系可以被明确检验，也区分了已经证明的参照、工程验收、旧模型发现和仍待确认的正式结论。

## 9. 本轮完整性与恢复记录

67项测试无skip；已保存证据的92160个风险单元、90个场景损失数组及9216个映射哈希均通过一致性检查。重新生成同一评价池前128个场景，90个冻结损失数组逐值一致，最大差0。这是同一记录的重放验证，不另算独立统计确认。

两套PNG/SVG图分别展示冻结模型证据和已知答案控制。旧3.5压缩包本次取回的字节缺尾，不能宣称已完整恢复；正文可读取。本包使用完整CRC校验的3.4代码依赖和与其逐字节匹配的checkpoint，包含需要的3.5自有桥接模块，独立运行不依赖旧3.5压缩包。来源和恢复范围见references/project_provenance.json。


v0.2交付新增540条含偏置循环、54条同lag迁移、25个相位比较、576行事件/噪声/阈值/观察精度面板，以及96分类+24原生图像重建接口检查。全部源于冻结旧模型开发分析或明确标为未训练的接口控制。参考文献、源码固定版本、测试、冻结合同、运行日志、图和验收均打包；正式科学验证的判据详见独立规格文件。


集成验收另实际运行了一份未训练continuous、width8、可见坐标1、20步的新配置，全9阶段通过；延迟证书自动覆盖其7载体×4时刻×5来源×5非raw映射=700比较。旧模型复用冻结主分析的8个阶段全部通过。这证明新配置尺寸和路径合同可执行，不增加正式训练模型证据。重复时间平均可能出现1+几个ulp的分类风险，本版仅对1e-12容差内的越界做截断，真实越界和非法比较族/置信参数仍拒绝。
