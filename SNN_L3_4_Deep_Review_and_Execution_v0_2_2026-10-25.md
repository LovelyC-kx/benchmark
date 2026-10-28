# SNN研究第三层 · 3.4 深入复审与执行稿 v0.2

日期：2026-10-25。主题：**同一冻结表示，为什么对不同任务、不同读取规则和不同样本量，会表现出不同的可学习性？**

阅读顺序：先读“本节论证”，再按3.4.1→3.4.4阅读。代码逐项对应，不需要先翻整个压缩包。原来的W0–W3保留；下述小节是对原计划的细化，不另造六层之外的新体系。

## 本节论证：我们要把什么讲透

3.3.3回答“状态对什么内容有依赖，以及某个固定读出误差与哪些内容有关”。3.4继续问：**这些内容的组织，能否预测新任务的学习代价，并指导改进读取？**

论文可以围绕一条可检验的论证组织本节：

> 历史内容进入状态，只解决了“存在依赖”；任务需要哪种内容组合、内容落在什么状态方向、采用什么学习规则、给多少标签，进一步决定可学习性。将内容分解与任务条件矩、状态谱和有限样本风险连接，可以解释差异，并产生在独立新场景上验证的读取改进。

这条论证中，四统计量和IPC等已有工具各有明确用途。研究价值来自：问题设置是否有辨别力；不同工具之间是否建立可验证的联系；是否发现稳定现象；能否指导后续设计。不会因为某个工具已有前例，就取消这种研究价值。我们也不把所有概率恒等式都标成首次提出。

### 本次完成与证据层次

| 层次 | 本次已有内容 | 应怎样使用 |
|---|---|---|
| 已知总体的理论与控制 | 有限p Gaussian精确条件分布；高阶方向翻转；谱与PCR噪声；连续值有限p MSE；高阶对齐方差控制 | 验证公式、实现和辨别能力，不能当成SNN群体发现 |
| 冻结SNN工程证据 | 1个旧模型，原训练只有1次optimizer更新；13个语义视图、39个F/G/FG组合；新评价场景8192/视图 | 流程验证与待正式检验的现象；13视图不是13个独立模型 |
| 分类读出训练 | 同p、同训练索引的Hebbian和固定λ ridge；MLP限C末层膜状态、球面任务的F/G/FG | 读出拟合不能计作SNN训练；固定超参数MLP也不是非线性能力上界 |
| 连续值补充 | 13视图、1248个闭式回归映射；独立8192连续目标评价场景 | 连接分类与数值恢复；尚非自然图像重建benchmark |
| 高阶对齐估计 | 9个“视图×角色”方差诊断；已知H3控制 | 推导成立与有限样本收益分开记录 |
| 正式benchmark | **0/435**；audit_test未访问；新SNN训练0 | 后续正式统计结论仍需要训练与复核 |

主checkpoint SHA256：
f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e。
本轮执行前后不变；CPU运行，未使用用户A40。原模型ID：272ae6230942ebeb。

## 一、先修正v0.1中的具体缺口

| 原问题或缺口 | 为什么会影响解释 | v0.2如何处理 |
|---|---|---|
| 将m/M计算后的权重Gaussian近似当成唯一更细参照 | 总体矩正确，不代表有限p错误率近似正确 | 保留原参照，新增不对训练权重作CLT近似的Gaussian精确条件分布 |
| 响应原点不为总体均值时，未把均值项写进有限p Gaussian训练律 | 固定中心化误差也会产生标签符号加权的随机项 | 显式加入响应均值与Rademacher和 |
| 只测随机球面任务 | 任务先验改变，平均统计量可能隐藏方向差异 | 增加坐标轴与等支持任务；严格区分未见方向与有限穷尽轴族 |
| 用拟合线性残差与标签相乘来判断高阶任务信号 | 线性拟合误差会混入诊断 | 已知Gaussian潜变量下先把标签的一阶Hermite项扣除，再独立交叉拟合 |
| MLP与线性读出未严格共享缩放和训练抽样 | 可能把预处理或抽样差异当成读出差异 | 统一geometry_fit标量RMS；共享嵌套p索引；区分训练抽样与初始化 |
| 没有直接在开发集选候选的对照 | 很难判断理论选择是否比直接试候选有价值 | 同时记录四统计量选择、开发任务矩选择、直接开发评价选择 |
| 单有平均收益，没有冻结后场景置信界 | 挑出的收益可能受评价噪声影响 | 先冻结选择/权重并散列，再产生新场景；按独立场景构造配对界 |
| 只有二分类，重建方向缺少对应推导 | 很难贯通文章既定的分类/恢复目标 | 增加任意向量连续目标的有限p Hebbian MSE恒等式与执行入口 |
| 只画状态能量，没有任务方向与读取规则 | 高能量不一定有目标，低能量不一定没容量 | 联接模式能量、任务对齐、总体线性容量、有限p读出与正则化 |
| 缺少“为什么某项估计没检出”的分析工具 | 小信号可能被估计方差淹没 | 推导任务特异的最优仿射控制变量，并保留实测无收益 |

---

## 3.4.1 · W0：先定义“谁在学什么”，再使用几何统计量

### 3.4.1.a 类别、目的与问题识别

这是**指定表示上的新任务学习问题**。识别标志是：主模型冻结；新目标或新任务方向单独定义；需要拟合一个读出；研究对象是该读出随p变化的泛化表现。

它与第二层“改神经元或状态机制后整个模型怎样变化”不同；与3.3.3“某因素在状态中的贡献”也不相同。这里先固定主模型，避免同时改变表示与学习器而无法解释。

每条曲线必须附带一个完整对象：

\[
\mathcal E=(\text{checkpoint},\,\text{输入/历史条件},\,\text{观测位置与时刻},
\,z,\,\pi(\theta),\,\text{读出规则},\,p,\,\text{预处理与预算}).
\]

缺少任何一项，“模型更容易学”就可能含义不明。

本轮内容角色是F、G、FG：F是指定focal源，G是其他源集合，FG是联合坐标。FG标签依赖F和G，不等于只测“纯F×G交互”。独立源包括H1–H4和C；每源4维Gaussian潜变量，输入经固定正交投影与噪声生成。C与H1在时间上的角色不同，不能仅按变量名解释。

### 3.4.1.b 核心近邻：Wakhloo、Slatton、Chung

**文献**：*Neural population geometry and optimal coding of tasks with shared latent structure*，Nature Neuroscience，2026，DOI [10.1038/s41593-025-02183-y](https://doi.org/10.1038/s41593-025-02183-y)。预印本：[arXiv:2402.16770v2](https://arxiv.org/abs/2402.16770v2)。

**已有贡献**：以共享潜结构的新任务为对象，将指定Hebbian学习误差与表示几何连接；研究最优编码，也包含非Gaussian模型和生物数据。不能将“首次考虑非Gaussian”“首次优化几何”作为我们的区别。

**代码核对**：作者utils.py固定commit 9674a6cb0c39926d70746024ce17de35f87df8f9，blob ca103732e125e9cb1282e80e667b661c5f83b39d。旧包保留选定四函数AST提取运行与数值对照记录；未运行作者完整训练。本次又核读该工具文件及正式版有关optimal codes/Gaussian simulations的Methods。确切来源见references/。

**我们的适配**：将已有几何公式接到SNN内容角色、膜状态/脉冲观测、时间条件和新任务曲线；引入可识别的高阶/有限样本参照与独立改进验证。

### 3.4.1.c 四统计量必须保持作者定义

令中心化联合二阶矩为：

\[
\Psi=\operatorname{Cov}(r),\quad
\Phi=\operatorname{Cov}(r,z),\quad
\Omega=\operatorname{Cov}(z),\quad Q=\|\Phi\|_F^2.
\]

\[
PR=\frac{(\operatorname{tr}\Psi)^2}{\operatorname{tr}\Psi^2},
\qquad c=\frac{Q}{\operatorname{tr}\Psi\,\operatorname{tr}\Omega}.
\]

\[
f^{-1}=
\frac{\operatorname{tr}\Omega\,
\operatorname{tr}[(\Phi^\top\Phi)\Omega^{-1}(\Phi^\top\Phi)]}{Q^2},
\]

\[
s^{-1}=
\frac{\operatorname{tr}\Omega\,
\operatorname{tr}[\Phi^\top(\Psi-\Phi\Omega^{-1}\Phi^\top)\Phi]}{Q^2}.
\]

对应参考：

\[
E_4(p)=\frac1\pi\arctan
\sqrt{\frac{\pi}{2pc^2PR}+f^{-1}+s^{-1}-1}.
\]

这里的f/s名称与代码返回值逐项对照，避免tuple顺序误配。潜变量奇异时显式给状态；不静默换伪逆后继续套原公式。Q=0时，公式参照可到机会水平，但不能据此证明非线性信息不存在。

代码：geometry_generalization.py中的sample_geometry、geometry_terms、predict_error；readout_design.py中的transported_geometry。

**原公式的任务分布与近似条件必须写在结果旁**。本轮轴族、等支持族仍展示E4，身份是球面理论的参照曲线，不是声称作者理论适用于任意先验。

### 3.4.1.d 为什么增加任务先验

“记住F”只说明内容范围，未说明F中的哪种函数是目标。新任务方向本身可以改变学习难度。

| 任务族 | 定义 | 能识别什么 | 本轮解释 |
|---|---|---|---|
| sphere | θ从单位球面抽样，y=sign(θᵀz) | 同一先验下新方向泛化 | 开发方向与评价方向独立 |
| axis | θ=e_j，逐坐标读出 | 是否某些坐标更容易读 | 有限轴族穷尽；不能称为未见轴任务 |
| equal_support | 恰有k个非零坐标，幅值1/√k，符号随机 | 任务组合稀疏/稠密程度 | 实际SNN本轮取k=D；控制实验另扫k |
| Hermite连续目标 | 已知Gaussian输入的正交多项式 | 线性记忆与高阶变换能力 | 分布不同必须更换正交基 |
| 向量连续目标 | y∈R^q | 数值恢复及重建接口 | 3.4.3.e详细推导 |

代码：task_structure.py的support_tasks、hermite_task_mean；run_focused_review.py的task_directions。等支持任务可能重复且正负标签等价，报告记录unique_directions_up_to_sign。当前C联合状态、F角色：sphere 16/16、axis 4/4、equal_support 16/7个独特方向。不能把16条列数当成16个独立任务。

**本部分的研究收益**：把“表示好不好”变成“对哪类任务、在哪个学习预算下好”。后续神经元机制的改进若只服务某些任务方向，这里能显示收益范围，而不是一个平均分掩盖掉。

---

## 3.4.2 · W1：先校准预测，再解释预测与观测的差距

### 3.4.2.a 总体矩恒等式与Gaussian近似要分开

任意平方可积表示，y∈{−1,+1}，固定表示原点：

\[
m_\theta=E[yr],\quad M=E[rr^\top],\quad
w_p=\frac1p\sum_i y_i r_i.
\]

训练场景IID，故严格有：

\[
Ew_p=m_\theta,\qquad
\operatorname{Cov}(w_p)=\frac{M-m_\theta m_\theta^\top}{p}.
\]

对独立测试点(x,y)，margin条件均值为y mᵀx，方差为
[xᵀMx−(mᵀx)^2]/p。用这两个矩拟合正态，得到TM参照：

\[
E_{MR}\approx E_{x,y}\,
\Phi_N\!\left(
-\frac{y\,m^\top x\sqrt p}{\sqrt{x^\top(M-mm^\top)x}}
\right).
\]

**矩关系严格，最后的正态错误率仍是近似**。训练权重可能偏态、有离散质量或重尾。v=0单独处理；score=0按半个错误计。代码：task_moments.py。

### 3.4.2.b 不对训练权重作CLT的有限p Gaussian参照

**来源**：联合Gaussian条件分解；Azzalini的skew-normal表示为相关概率背景；SciPy 1.17.0的skewnorm随机生成实现也使用等价的潜在正态分解。我们没有将skew-normal或半正态分解称为新分布理论。新增价值是把它写成适合本benchmark、可含响应均值和真实测试池的有限p校准器。

设训练(r,z)联合Gaussian，E z=0、E r=d，给定独立测试特征x与任务θ。令：

\[
v_\theta=\theta^\top\Omega\theta,\quad
A=\frac{x^\top\Phi\theta}{\sqrt{v_\theta}},\quad
B^2=x^\top\Psi x-A^2,\quad D=x^\top d.
\]

存在相互独立的g,e∼N(0,1)和S∈{−1,+1}，使得：

\[
\operatorname{sign}(\theta^\top z)\,x^\top r
\overset d= A|g|+D S+B e.
\]

证明：先将xᵀ(r−d)沿标准化θᵀz分解为A g+B e；再乘sign(g)。sign(g)与|g|独立，sign(g)e仍是独立标准正态，遂得上式。

p个训练样本的score之和为：

\[
A H_p+D S_p+B\sqrt p\,e,\quad
H_p=\sum_{i=1}^p|g_i|,\quad S_p=\sum_{i=1}^p S_i.
\]

因此对测试标签y：

\[
R_p(x,y)=E_{H_p,S_p}\,
\Phi_N\!\left[-\frac{y(AH_p+DS_p)}{B\sqrt p}\right],
\]

B=0按确定性margin和半平局计算。注意1/p不改变分类符号。

零响应均值、p=1还有解析式：

\[
R_1(x,y)=\frac12-\frac y\pi\arcsin
\frac{A}{\sqrt{x^\top\Psi x}}.
\]

**为什么均值项不能省**：我们只能用独立geometry_fit估计中心，不会神奇地知道总体中心。相对于固定中心，Gaussian代理的d可以非零，p个标签的随机正负会把均值误差累积为D S_p。

**“精确”的边界**：指给定真实Gaussian训练参数的分布恒等式；数值积分、有限测试参照池、参数估计各有自己的误差。Gaussian代理训练可以配真实非Gaussian测试池，后面用于逐步替换比较。并未假设真实SNN状态已经Gaussian。

代码：finite_gaussian.py的finite_gaussian_forecast。已知控制8组，含均值0/非0×p=1/8/32/128。独立直接Gaussian训练2048次，与精确条件法最大风险差约0.001882；内层积分使用4096抽样或p=1解析积分。MCSE和内层界分别输出，不合并成总体保证。

### 3.4.2.c 把风险差拆开，找到后续分析入口

**相关文献**：

1. *Neural networks trained with SGD learn distributions of increasing complexity*，Refinetti、Ingrosso、Goldt：预印本2022、ICML 2023、JSTAT 2025。以匹配较低阶统计量的替代分布分析学习复杂度。不能只标“2025新工作”。
2. *Gaussian universality in neural network dynamics with generalized structured input distributions*，Bae、Jeong：arXiv:2405.00642v4，2025-11修订，JSTAT 2026。Gaussian等价取决于结构、标准化与条件，不能因“看起来非Gaussian”就断定失效。

**代码适配为什么不能直接复制**：

| 来源 | 已核读文件与版本 | 需要改的地方 |
|---|---|---|
| sgoldt/dist_inc_comp | commit ed7cfe7e6020873f39dcf965f8af7b2a44c72094；README、censoring.py、cdatasets.py | 作者class-conditional图像clone匹配的是类别条件均值/协方差；每通道处理、jitter、uint8裁剪/量化会改变实际抽样矩。我们的joint(r,z)代理与标签对齐贡献代理是不同对象 |
| peardragon/GaussianUniversality | commit 10138a9468f9a360efdd6f1d5d175e48e6a492c6；README、utils/gaussian_mixture.py、utils/model.py | 逐坐标标准化不等于全白化；代码有未设种子的default_rng。我们显式冻结种子、协方差和拟合池；未运行作者全套训练 |

定义五个风险：

| 符号 | 训练表示 | 测试表示 | 用途 |
|---|---|---|---|
| E_RR | 真实 | 真实 | 实际指定读出风险 |
| E_MR | 以真实m/M匹配标签对齐训练贡献的正态参照 | 真实 | 训练贡献形状参照，原TM |
| E_GR | 联合Gaussian训练代理，有限p精确法 | 真实 | 同二阶联合Gaussian替换 |
| E_GG | 同一Gaussian训练代理 | Gaussian代理 | 同族参照 |
| E_4 | 四统计量近似 | 对应理论平均 | 最简参考 |

于是恒等地：

\[
E_{RR}-E_4=
(E_{RR}-E_{MR})+(E_{MR}-E_{GR})
+(E_{GR}-E_{GG})+(E_{GG}-E_4).
\]

这四项依次提示：训练贡献形状、任务条件矩、测试分布替换、Gaussian内部近似/平均化。它们是**有符号的风险对比**，有限参照池与参数估计也会进入差值；不把四项命名为唯一、互相独立的因果贡献。

本轮C末层膜状态、F、p128：
E_RR=0.443146，E4=0.432082，差0.011064。
四段约为+0.002534、−0.005833、+0.017119、−0.002756。正负抵消正是需要逐步参照的原因：总差小，不代表每一步都合适。

代码：approximation_diagnostics.py的prediction_gap，及run_focused_review.py保存的独立forecast记录。

### 3.4.2.d 从误差诊断进一步给理论界

固定测试点x，令q=y_train r_trainᵀx，
μ=mᵀx，σ²=xᵀ(M−mmᵀ)x，ρ=E|q−μ|³。
若σ>0且ρ有限，IID Berry–Esseen给出：

\[
|R_{\rm true}(x)-R_{\rm normal}(x)|
\le \min\left(1,\frac{0.56\,\rho}{\sigma^3\sqrt p}\right).
\]

再对独立测试分布平均即可。平局通过左右CDF的中点处理；σ=0回到确定性情形。0.56取保守常数，不追求最优。参考Shevtsova 2011，[arXiv:1111.6554](https://arxiv.org/abs/1111.6554)。

这带来两个实际策略：p小、偏度/尾部大或margin方向方差近退化时，优先用精确Gaussian控制或直接经验曲线；p增大仍不贴合时，检查分布/矩替换，而非笼统归咎“样本还不够”。

代码berry_esseen_diagnostic用样本矩得到的是**plug-in诊断**。未给ρ和σ做同时置信控制时，输出不能称为已认证真实误差上界。总体定理、参数估计与风险抽样三个误差来源要分列。

**本部分的研究收益**：预测不准本身变成研究对象；能区分理论参照需要修正的环节，并指导需要哪种分布控制和更多采样。

---

## 3.4.3 · W2：把内容结构连到任务方向、状态谱和连续恢复

### 3.4.3.a 与3.3.3的严格连接

设相互独立的因素X=(F,G,U,E)，平方可积表示有功能ANOVA：

\[
r=\mu+\sum_{\varnothing\ne S}r_S(X_S),\qquad
E[r_S\mid X_{S\setminus\{i\}}]=0.
\]

若任务y只依赖X_T，那么S含T外因素时，E[y r_S]=0。因为可以先对该额外因素积分，利用零条件均值。故：

\[
m_y=E[yr]=E[y]\mu+\sum_{S\subseteq T}E[y r_S].
\]

平衡分类E y=0时均值项消失。另一方面，各独立ANOVA成分仍进入M及训练权重波动。

因此：**与目标无关的历史成分，可能不推动平均学习方向，却增加给定规则的有限样本波动。** 这并不意味着任何读出都一定被同等干扰；ridge、白化和非线性头可改变利用方式。对目标有关的高阶成分也不能直接统称噪声。

这就是从3.3.3进3.4的必要桥梁。3.3.3给内容成分，3.4追问它们如何影响一个明确学习器。正式实验再由第二层干预增加/移除状态机制，第四层用机制实验验证，不在3.4单独宣称全部因果链。

### 3.4.3.b 用正交化标签识别高阶任务对齐

对z∼N(0,Ω)，令：

\[
u_\theta=\frac{\theta^\top z}{\sqrt{\theta^\top\Omega\theta}},
\quad \kappa=\sqrt{2/\pi},\quad
h_\theta=\operatorname{sign}(u_\theta)-\kappa u_\theta .
\]

Gaussian条件期望给出E h=0、E[z h]=0。定义：

\[
\eta_\theta=E[rh_\theta]
=m_\theta-\kappa\frac{\Phi\theta}{\sqrt{\theta^\top\Omega\theta}}.
\]

因此任意预先固定、独立拟合的线性控制A都有：

\[
E[(r-Az)h_\theta]=\eta_\theta.
\]

它不要求r是Gaussian，也不要求A恰好是总体OLS；已知Gaussian潜变量法则才是关键。对自然数据未知潜变量，不能把估计协方差带入后照称总体精确。

与“残差r−Âz乘原标签”相比，这里预先剔除了标签的一阶Gaussian成分。用独立池a/b估计η：

\[
\widehat P_\theta=\widehat\eta_{\theta,a}^\top
\widehat\eta_{\theta,b}.
\]

条件于固定控制映射，其期望是||ηθ||²，避免同池平方范数的正偏。负估计保留；负值不是“负信息”，也不是证明η=0。

代码：task_structure.py的gaussian_label_residual、crossfit_higher_alignment。

**同类文章关系**：Hermite分解/正交多项式是已有方法；Wakhloo等工作也并非只看线性模型。我们的推进是将“超出Gaussian一阶任务预测的贡献”明确连到SNN内容角色、任务先验与后续有限p风险，并给出能实际估计的独立池方案。

### 3.4.3.c 一个可解析且辨别力强的任务先验控制

取独立标准Gaussian z_i：

\[
r_i=a z_i+H_3(z_i),\quad
H_3(z)=\frac{z^3-3z}{\sqrt6},\quad \|\theta\|=1.
\]

同一表示的Ψ=(a²+1)I、Φ=aI，与任务先验无关。Gaussian条件矩给出：

\[
m_{\theta,i}=\kappa\left(a\theta_i-\frac{\theta_i^3}{\sqrt6}\right),
\qquad
\eta_{\theta,i}=-\frac{\kappa\theta_i^3}{\sqrt6}.
\]

于是：

\[
\theta^\top m_\theta
=\kappa\left(a-\frac{\sum_i\theta_i^4}{\sqrt6}\right)
=\kappa\left(a-\frac1{\sqrt6\,PR_\theta}\right),
\quad PR_\theta=\frac1{\sum_i\theta_i^4},
\]

\[
\|\eta_\theta\|^2=\frac{\kappa^2}{6}\sum_i\theta_i^6.
\]

等支持k时：

\[
\theta^\top m_\theta=\kappa\left(a-\frac1{\sqrt6 k}\right),
\qquad \|\eta_\theta\|^2=\frac{\kappa^2}{6k^2}.
\]

a=0.1时临界k=4.08248：k≤4为负，k≥5为正。D16控制已跑：k1精确投影−0.24595，k5为+0.01464，k16为+0.05943；实测与解析式相符。k4的MC误差接近均值，因此该点的正负依据解析式，不依赖一次随机估计。

**它证明什么**：同一二阶几何并未描述任意任务先验下的所有学习方向；方向的集中程度可选择性暴露高阶贡献。不能把它解释为原球面任务定理在任意先验下失败，也不能仅凭teacher方向投影推出所有θ下的分类极限；后者仍需计算实际margin风险。

本控制适合在论文做简洁理论例子，并在SNN中检验是否出现对应的任务方向依赖。代码：hermite_task_mean；run_v0_2_controls.py。

### 3.4.3.d 进一步推进：高阶对齐的方差最优仿射控制

为什么做：有限采样下η可能很小，“没检出”可能来自估计方差。控制变量的目的应是降低估计成本，且保持同一个被估计量。

令q=(1,zᵀ)ᵀ，v_h=1−κ²。Gaussian法则严格有：

\[
E[q h_\theta]=0,\qquad
E[q q^\top h_\theta^2]
=v_h\begin{pmatrix}1&0\\0&\Omega\end{pmatrix}.
\]

证明要点：沿uθ方向分解z；E h²=1−κ²，且
E[u²h²]=1−4κ²+3κ²=1−κ²；正交分量独立，交叉项由对称性消失。

考虑贡献(r−Lq)hθ。其均值始终是ηθ，故最小化协方差迹等价于加权最小二乘：

\[
L_\theta^*=
E[r q^\top h_\theta^2]\,
\left[v_h\,\operatorname{diag}(1,\Omega)\right]^{-1}.
\]

有精确“超额方差”关系：

\[
\operatorname{trVar}[(r-Lq)h]
-\operatorname{trVar}[(r-L^*q)h]
=\operatorname{tr}[(L-L^*)D_h(L-L^*)^\top]\ge0.
\]

这对总体L*成立。估计出的L不保证优于普通控制，所以实现采用独立fit池估计，独立validation池在none/ordinary/half-weighted/full-weighted中选择，再在两个未参与选择的moment池估计η交叉功率。

已知H3控制的最优斜率为：

\[
a+\frac{2\kappa^2}{\sqrt6(1-\kappa^2)}.
\]

a=.1时约1.53045；独立大样本贡献方差由1.36136降至0.68296，约降低49.83%。**当前9个SNN诊断全部选回ordinary_linear，额外收益0**，并且高阶交叉功率有负值，原样保留。

代码：higher_order_controls.py、run_higher_order_controls.py。该推导是本项目新补的针对性估计方案；其原则来自经典控制变量/加权最小二乘，全球优先性尚无证据。即使正式SNN不需要它，仍能明确给出“何时值得花更多样本估计高阶信号”的成本分析。

### 3.4.3.e 状态能量、任务容量与有限学习如何连接

**近期文献1**：Abdalla、Rontani，*Organization of computation in reservoir computing*，[arXiv:2607.17858v1](https://arxiv.org/html/2607.17858v1)，2026-07-20。重点读§2.1/2.2的状态分解与模式-任务投影。其ESN分析已经指出低能量模式可承载显著任务容量。未找到并核实官方代码仓库；本包按方程独立实现，没有冒称复用了作者仓库。

**近期文献2**：Haruna、Nakajima，*Memory Uncertainty Relation and Harmonic Memory in Random Recurrent Networks*，Physical Review E 113,055304，2026，DOI [10.1103/2vtp-m1yk](https://doi.org/10.1103/2vtp-m1yk)，[arXiv:2605.24628v1](https://arxiv.org/html/2605.24628v1)。核读§III.1/III.2。官方代码[taichiharuna/mur](https://github.com/taichiharuna/mur)，commit 5c4d2f6dfb936041d289704400a2dec307e77647；完整核读README及两个选定分析脚本。README明确未固定PRNG种子；未运行全仓训练/仿真。

**基础与代码来源**：Dambre等，*Information Processing Capacity of Dynamical Systems*，2012，DOI10.1038/srep00514；Schulte to Brinke等，*A refined information processing capacity metric allows an in-depth analysis of memory and nonlinearity trade-offs in neurocomputational systems*，Scientific Reports，2023，DOI10.1038/s41598-023-37604-0，官方[Zenodo代码](https://zenodo.org/records/7688574)。

固定原点下M=UΛUᵀ，对单位方差、零均值目标y，m=E[ry]：

\[
C_y=m^\top M^+m=\sum_{\lambda_i>0}
\frac{(u_i^\top m)^2}{\lambda_i},
\qquad e_i=\frac{\lambda_i}{\sum_j\lambda_j}.
\]

总体矩且支撑相容时C_y是最佳无截距线性回归所解释的目标方差。中心化、目标归一化、截距约定必须与公式一致。

**e_i与C_y的组成不同**：一个模式能量小，分子也可能恰好与λ成比例，仍能有完整容量。已知控制r=(.1F,G)，F/G独立标准Gaussian：F方向能量仅0.9901%，总体线性恢复F容量仍为1。

但finite-p Hebbian不会自动采用最优逆协方差。该控制的p128分类错误，raw Hebbian约0.37496，已知白化后约0.02084；p1约0.49370→0.24228。这是已知控制，不是SNN结论。它说明“总体上能读”和“少样本下给定规则能学”可相差很大。

Haruna的harmonic关系提供桥梁。沿总体Hebbian方向、仅拟合一个标量增益的容量为：

\[
H_y=\frac{\|m\|^4}{m^\top M m}\le C_y.
\]

由Cauchy–Schwarz：
(mᵀm)²≤(mᵀMm)(mᵀM^{-1}m)。等号要求m支撑上的相关特征值一致。在有效支撑上精确白化可闭合此总体间隙；有限样本估计白化仍可能放大噪声。

**可逆变换不创造信息**：r'=Br、B可逆时，总体无限样本线性容量不变；但Hebbian决策使用r_trainᵀBᵀB r_test，度量改变，所以finite-p风险可变。这给我们的读取改进一个明确解释。

### 3.4.3.f 为什么噪声后“记忆提高”需要检查正则化

Haruna等研究输入噪声与PCR的相互作用。本项目先构造更简单的**固定表示+独立观测噪声**控制，不与改变SNN内部动态的输入噪声混淆。

若给r加独立各向同性噪声、方差ν：

\[
C(\nu)=\sum_i\frac{(u_i^\top m)^2}{\lambda_i+\nu},
\quad
C'(\nu)=-\sum_i\frac{(u_i^\top m)^2}{(\lambda_i+\nu)^2}\le0.
\]

不限制支撑时容量下降。但若PCR只保留λ_i+ν>τ，噪声可把某个目标模式抬过**绝对阈值**。例M=diag(.01,1)、m=(.1,0)、τ=.02：

| 噪声方差ν | PCR容量 | 不限支撑容量 |
|---|---:|---:|
| 0 | 0 | 1 |
| .009 | 0 | .5263 |
| .011 | .4762 | .4762 |
| .015 | .4 | .4 |

因此PCR分数上升，不等于系统获得更多信息。相对阈值、绝对阈值、ridge、截断支撑应分开报告。真实SNN若输入噪声改变动态，M和m本身也会变化，必须另测。

代码：task_spectrum.py的crossfit_task_spectrum、population_linear_capacity、harmonic_capacity_reference、pcr_noise_scan。谱基由独立geometry池拟合，任务功率由moment_a/b交叉乘积估计；其signed capacity proxy可超出[0,1]，不作为已知总体IPC。重复特征值附近单个模式不唯一，应优先汇报谱块/累计量，而非赋予每个任意特征向量生物意义。

### 3.4.3.g IPC作者代码审查之后，怎样适配SNN

本次没有下载Zenodo原始实验数据，只取出并校验三个源码成员：capacity_calculation.py、subtract_encoder_capacity.py、test_capacity.py；SHA256见references/。读了任务生成、Legendre递推、容量核心、collect以及编码器减除的相关函数，未复现NEST仿真、未逐行审计所有iterator循环。

| 观察到的实现约定 | 我们为何调整 | 本项目处理 |
|---|---|---|
| Uniform输入配Legendre目标 | 我们控制潜变量为Gaussian | Hermite目标；输入分布变化必须重新验证正交性 |
| collect对传入活动做原地中心化/逐通道标准化 | 可能改变调用者数组；全静默通道可能除零 | 拷贝数组；独立拟合变换；显式处理零方差；各读出共享同一变换 |
| 选定旧代码调用scipy.linalg.pinv(cond=...) | 当前SciPy API不兼容 | 本项目独立实现显式谱支撑；记录软件版本 |
| standalone cov_capacity的默认R_inv=None随后调用.any | 独立调用路径缺少保护；collect正常传矩阵时不触发 | 不据此宣称作者结果无效；我们显式校验输入 |
| delay/degree搜索有停止/剪枝约定 | SNN可能出现非单调可访问性 | 当前预定p网格全部保存；完整时延研究归3.6 |
| 编码器capacity减除带变换和max(0,·) | 不是唯一因果分解 | 不把其减除量当成SNN动态净贡献；回到第二层受控机制对照 |

另一个关联是Inubushi、Yoshimura的*Reservoir Computing Beyond Memory–Nonlinearity Trade-off*，2017，DOI10.1038/s41598-017-10257-6；其线性/非线性单元混合思路可启发第二层机制对照，但不能把其条件下的权衡直接说成所有SNN的必然规律。本轮未核实对应官方代码仓库。

**这一部分的独立研究价值**：把“何种内容落在哪些模式”连到“哪些有限样本读者用得好”，并识别正则化造成的能力变化。这比只报告PCA图或总体IPC更接近改进建议。

### 3.4.3.h 从分类走向连续值恢复：有限p精确MSE

这是本轮新增、与文章重建目标直接相关的接口。设y∈R^q，r∈R^d，训练与测试IID，表示原点固定；只要求相关矩有限：

\[
W_p=\frac1p\sum_i r_i y_i^\top,\quad \hat y=\alpha W_p^\top r.
\]

定义：

\[
M=E[rr^\top],\quad C=E[ry^\top],\quad
T=E[\|y\|^2rr^\top],\quad Q=E\|y\|^2,
\]

\[
S=\|C\|_F^2,\quad A=\operatorname{tr}(C^\top MC),
\quad B=\operatorname{tr}(MT).
\]

则无需Gaussian：

\[
\boxed{
R_p(\alpha)=\frac1q
\left[Q-2\alpha S+\alpha^2
\left(A+\frac{B-A}{p}\right)\right].
}
\]

**推导**：测试风险展开为Q−2α tr(CᵀE Wp)+α²E tr(WpᵀM Wp)。E Wp=C；二重训练和中i≠j给A，i=j给B，因此预测平方项为(1−1/p)A+B/p。除以q即每目标维MSE。

这里B−A≥0，是训练映射在M加权下的方差。因此无标量重新校准时：

- Q：目标尺度；
- −2αS：预测与目标的交叉项；
- α²A：总体平均映射的预测能量；
- α²(B−A)/p：有限训练样本增加的预测能量。

这些是风险分解，不把每一项都单独称为非负“误差贡献”。只有最后的训练方差项保证非负。

若总体矩已知，最佳公共标量为：

\[
\alpha_p^*=\frac{S}{A+(B-A)/p},\qquad
R_p(\alpha_p^*)=\frac1q\left[Q-\frac{S^2}{A+(B-A)/p}\right].
\]

从数据估计该增益要单独拟合与计费；使用评价标签调增益不允许。当前使用独立moment池2048个连续标签，不能称作仅p标签的方法。

**为什么分类结论不能代替连续恢复**：

1. r=y=z∼N(0,1)：任意p时Hebbian分类方向都正确，分类错误为0；但固定α=1回归MSE=2/p，oracle公共标量后的MSE=2/(p+2)。
2. 向量r=y=z∼N(0,I_D)：α=1的每维MSE=(D+1)/p。
3. H3与Gaussian code可匹配M、C、Q，却有不同混合四阶矩T；因此连续恢复的finite-p方差也不同。

一维r=.1z+H3(z)、y=z时，M=1.01、C=.1、Q=1，与匹配Gaussian code相同；但TH3=7+.2√6+.03≈7.51990，TGaussian=1.03。二阶几何相同并不足以决定该连续读出的有限p风险。

代码：continuous_risk.py的continuous_moments、exact_continuous_risk；run_continuous_extension.py。这套恒等式来自均值估计的标准风险展开；本项目新增的是面向同一SNN表示的分类/恢复比较、混合矩控制及预算清晰的执行方案。

### 3.4.3.i 当前连续值执行与边界

新评价场景ID从5,700,000开始，与分类主评价池分离；13视图共享预定p计划，每个冻结回归映射在评价生成前保存。1248个闭式映射不等于1248次SNN训练。原生任务头保留，用来识别“原任务会不会做”和“冻结状态还能否支持重新读取”。

F角色，p128，本轮结果：

| 观测 | 原生头连续MSE | raw Hebbian MSE | 独立矩估计增益后 | 同p固定λ ridge |
|---|---:|---:|---:|---:|
| C末层post | 1.5787 | 2.1825 | .9678 | 1.0388 |
| C两层post拼接 | 1.5787 | 8.4885 | .8142 | .4004 |
| H1末层post | 1.2974 | 2.3028 | 1.0008 | 1.2254 |

“C两层post拼接”确切是第一层与最后一层的post拼接，64维；**不是膜电位+脉冲拼接**。这一点在collect代码中核实。

raw Hebbian的尺度依赖明显，较大MSE不等于该SNN丢失了相同程度的信息。当前几何统一RMS只保证读出比较预处理一致，不能令未校准Hebbian自动成为最优回归。ridge比标量增益更好时如实保留。

SNN的风险预测用样本矩，所以是plug-in预测，不是说估计值与4次训练抽样平均必须严格相等。连续MSE无上界，不能套用后面的分类Hoeffding区间。

后续接重建：可令y为连续信号/像素向量，用该线性参照分析表示与少样本恢复；完整重建头仍应报告相应指标和预算。该公式不能直接变成U-Net等训练解码器的风险公式，也不能直接推出平均PSNR。

---

## 3.4.4 · W3：怎样做实验，才能把分析转化为可信的改进

### 3.4.4.a 冻结与独立采样合同

末层post/spike各32维，两层post拼接64维。联合观察增加访问维度，正式比较需要同时报告状态访问、读出参数和计算预算；当前表格用于定位读取瓶颈。

主分类执行13个语义视图：C末层post的6种schedule；H1末层post；C末层spike；C两层post拼接；其中C末层post/两层post各补轴族和等支持族。每视图测F、G、FG，共39个组合；同一个checkpoint，不能当独立架构样本。

| 池 | 场景数 | split/首ID | 用途 |
|---|---:|---|---|
| geometry | 512 | geometry_fit / 5,000,000 | 中心、统一标量RMS、协方差、候选谱变换 |
| moment_a | 1024 | probe_train / 5,100,000 | 任务矩与高阶对齐的一半 |
| moment_b | 1024 | probe_train / 5,200,000 | 独立另一半；候选开发矩可合并两池 |
| forecast | 512 | development_test / 5,300,000 | 预评价风险参照/开发候选选择 |
| reader_train | 512 | probe_train / 5,400,000 | 读出p子样本，p=1/8/32/128 |
| primary_classification | 8192 | development_test / 5,600,000 | 权重/选择冻结后产生的分类评价 |
| continuous_evaluation | 8192 | development_test / 5,700,000 | 连续恢复评价；不套分类置信界 |

更早ID5,500,000的小规模整合运行只保留为工程历史；不是本次新主证据。分类主池也不是封存audit_test。相同scene可被多个任务共享，统计时以scene为独立单位，不把任务列数相乘当样本数。

所有读出在同一geometry_fit变换后训练；每个p、每个训练draw使用同样索引；各p嵌套。4个训练draw与2个MLP初始化种子分开保存。p128单个读出只用128个标签；4个draw合起来用348个独特场景标签。报告的是各独立读出风险平均，**不是将4个读出部署成一个ensemble再说只用了128个标签**。

### 3.4.4.b 读出、选择器与标签预算

| 方法/参照 | 需要什么 | 本轮地位 |
|---|---|---|
| identity Hebbian | 每任务p标签 | 明确学习规则，理论对接对象 |
| 四候选谱预条件Hebbian | geometry池；指定选择策略；每任务p标签 | identity、η=.1/1/10，保持可逆 |
| fixed ridge | 同p；固定λ=10^-3 | 读取基线，未调参，不能称为最优ridge |
| MLP | 同p；width16、tanh、80epochs、两个初始化 | 仅C末层post/sphere的F/G/FG已运行；固定训练规则，不是非线性上界 |
| 四统计量选择 | 带潜变量的geometry池 | 有结构标注的诊断/设计方法，记录其成本 |
| 开发任务矩选择 | 2048个moment场景的开发任务标签、forecast池 | 新任务方向上迁移选出的变换；轴族没有未见轴承诺 |
| 直接开发候选选择 | 各候选拟合开发任务头，再评forecast | 必须保留的现实对照；不能只与随机候选比 |
| 真实任务m/M或连续增益参照 | 额外moment标签/连续目标 | oracle诊断预算，不能称纯p标签方法 |

谱变换为：

\[
B_\eta=U\,\operatorname{diag}
[(\lambda_i/\bar\lambda+\eta)^{-1/2}]\,U^\top,
\quad \eta>0,
\]

另做无关分类符号的全局归一化。它不改变表示所含信息，改变Hebbian相似度度量。代码：readout_design.py。所有候选、随机候选平均、三种选择器、无效/负收益都输出。

成本也不是附属项：当前CPU记录中，C联合状态的直接开发选择约.009秒，理论+任务矩选择合计约.106秒；它们还不含整个数据生成和SNN前向成本。**本轮不能声称理论选择更省计算**。正式实验应统一计入潜变量标注、场景、前向、拟合与选择成本，并比较收益随预算变化。

### 3.4.4.c 条件配对收益保证

给定冻结的主模型、任务集合、训练样本集合、读出权重和选择结果，对每个新独立scene j，把任务/读出draw平均后的损失差记为：

\[
d_j=\ell_{\rm base}(j)-\ell_{\rm selected}(j)\in[-1,1].
\]

scene之间IID，任务和读出之间可以相关。Hoeffding与union bound给出K个预定比较的同时区间：

\[
\Delta\in[\widehat\Delta-\epsilon,\widehat\Delta+\epsilon],
\qquad
\epsilon=\sqrt{\frac{2\log(2K/\delta)}{N}}.
\]

本轮K=13×3×4×3=468，N=8192，δ=.05，ε=.0490071。无需假设任务独立。这个保证条件于已冻结对象，不能升级成跨SNN训练种子、所有架构或所有未来任务的保证。

代码：paired_readouts.py的paired_gain_certificate。selection_freeze.json及文件SHA保存选择、任务方向、训练索引/权重摘要；新评价池随后才产生。每scene损失与冻结权重保留于NPZ，便于复算。

### 3.4.4.d 已运行结果如何解释

球面任务、F角色、p128：

| 冻结表示 | identity Hebbian错误率 | 四统计量所选谱Hebbian | 同p fixed ridge | 所选相对identity的收益及同时区间 |
|---|---:|---:|---:|---|
| C末层post | .44315 | .43256 | .43375 | .01058；[−.03843,.05959] |
| C两层post拼接 | .36972 | .26901 | .23199 | .10071；[.05170,.14972] |
| H1末层post | .49131 | .49068 | .49077 | .00064；[−.04837,.04964] |

C末层post的固定MLP错误率为.44896；没有借此宣称“MLP不如线性”的一般结论。C联合状态ridge好于所选Hebbian，说明我们保留了更强的简单读取方法，而非把弱基线提升包装成全面胜利。

39个组合的p128四统计量选择：34个正平均收益、5个负平均收益；4个区间下界为正。组合共享模型与场景，不能把34/39当独立成功率。

值得正式检验的现象是：**同一个旧SNN，在不同位置/角色上的读取瓶颈不同；同一可逆变换家族有时有效、有时基本无效。** 这为后续机制和读出设计提供可测假设。当前一更新checkpoint不足以决定正式训练后的现象是否存在。

### 3.4.4.e 正式实验怎样接，不把六层责任混在一起

W0–W3是顺序依赖的研究工作包，不是四种新benchmark任务。

| 工作包 | 正式实验动作 | 核心图/分析 | 对后续的指导 |
|---|---|---|---|
| W0 | 对齐目标、观测、统计量、任务先验、读者与预算；保留本文实现与作者参照 | 任务/统计定义表；版本与代码核对 | 避免更换定义后还沿用原结论 |
| W1 | Gaussian已知控制、真实表示代理替换、finite-p精确/近似参照；矩样本量扫描 | 风险预测误差与四段参照；BE诊断 | 指定哪里需要高阶/有限样本分析 |
| W2 | 接3.3.3内容分量；任务支持/方向；能量与任务谱；分类/连续恢复 | 内容→学习方向/波动→曲线；低能量可读性 | 指出值得保留、抑制或重新组织的状态方向 |
| W3 | 冻结后新任务/场景；同p同索引；ridge调参与固定基线分开；读取改进验证 | p曲线、选择器收益/失败、成本-收益 | 给出有条件、可复用的设计建议 |

正式训练后先在预定代表性配置上做完整分析，再扩展；新主模型seed、任务方向seed、读出训练抽样seed与MLP初始化seed分开。由既定正式清单决定神经元/骨架配置，不临时把13个观测视图冒充神经元覆盖。

需要特别回答的问题：

1. 神经元时间常数/自适应/额外状态改变的，是目标内容进入状态、目标谱位置、无关内容波动，还是指定读出的度量匹配？
2. 跨膜状态/脉冲观测，差异来自信息本身还是可学习性？3.4提供读出合同，载体本身的细化在3.5。
3. 随历史间隔增长，先下降的是内容依赖还是特定读出？跨时刻重编码与访问轨迹由3.6展开。
4. 更强的内部信息是否到达原生头？跨层通信与原生头到达由3.7展开。
5. 在等预算下，调整读出是否足够，还是必须更改神经元或训练目标？第二/四层负责机制干预，3.4提供可检验预测。

这些问题可支撑高影响力论证：识别何种“更多记忆”转化为实际用途、何种只增加学习负担，并给出可验证的修正。是否形成普遍或架构特异的规律，由正式结果决定；研究问题和实验力度不应先被缩成只做图示工具。

---

## 二、独创性应该体现在哪里

| 层面 | 已有工作与我们相近的部分 | 我们要形成的具体贡献 |
|---|---|---|
| 研究问题 | 几何影响泛化；RC研究记忆/非线性；SNN benchmark比较任务/设备 | 统一判断保留内容、任务需求、有限学习和实际利用之间的断点 |
| 设置 | 随机共享潜结构任务、正交多项式目标、Gaussian代理 | 保留SNN历史角色与时序条件，联合任务先验、载体/层位、分类/连续目标和读取预算 |
| 理论连接 | Hebbian矩、Hermite、harmonic/OLS、控制变量均有来源 | 将内容分解接到方向/波动；finite-p精确Gaussian桥；任务方向翻转；连续混合矩风险；η的最优仿射控制 |
| 实验 | 测总体平均分、容量或能量 | 同表示、同p同索引的配对读者；逐步分布替换；新场景冻结验证；正负收益与实际成本 |
| 发现 | 低能量模式有容量已被研究 | 在正式SNN上定位何种状态机制导致何种读取瓶颈，检验其跨任务/种子稳定性 |
| 设计增益 | 白化、ridge、预条件已有 | 根据诊断选择何时/何处调整读取或状态机制，并验证收益及失败条件 |

“高阶工具有前例”不否认新的科学发现；“在SNN上运行一次”也不自动形成新发现。本节的主张应围绕可解释、可复现、能指导后续设计的完整证据链组织。

本轮已拿到的是推导、可运行实现、已知控制和单checkpoint的流程证据。正式论文可以保持有力度的假设与目标；结果层面则按已实际获得的证据写。

## 三、文献与代码索引：本轮实际读到哪里

详细commit、blob、SHA256与适配理由见references/l34_v0_2_source_audit.json。下面是查阅入口，不宣称每篇全文与全仓库均已复现。

| 文献 | 原文/代码入口 | 本轮核读范围 | 接入位置 |
|---|---|---|---|
| Wakhloo等，2026 | DOI10.1038/s41593-025-02183-y；arXiv2402.16770v2；原作者utils固定blob见审计 | 正式版相关Methods及工具文件；原四函数对照记录保留 | W0，E4，W1 Gaussian控制，W3读取设计 |
| Abdalla/Rontani，2026预印本 | https://arxiv.org/html/2607.17858v1 | §2.1/2.2及相关定义；官方代码未核实 | W2模式能量/任务组织 |
| Haruna/Nakajima，2026 | https://arxiv.org/html/2605.24628v1；https://github.com/taichiharuna/mur | §III.1/III.2；README及2脚本 | W2 harmonic与PCR控制 |
| Refinetti等，2022/23/25 | https://proceedings.mlr.press/v202/refinetti23a.html；https://github.com/sgoldt/dist_inc_comp | Gaussian clone有关方法；完整选定3文件 | W1匹配分布参照 |
| Bae/Jeong，2024/25/26 | https://arxiv.org/html/2405.00642v4；https://github.com/peardragon/GaussianUniversality | 标准化/结构相关段落；完整选定3文件 | W1等价条件与生成器 |
| Schulte to Brinke等，2023 | https://doi.org/10.1038/s41598-023-37604-0；https://zenodo.org/records/7688574 | 容量/减除有关方法；选定3源码成员的相关函数 | W2容量估计与实现适配 |
| Dambre等，2012 | https://doi.org/10.1038/srep00514 | 原计划已有基础；本轮经上述近期原文核对相关定义，未再次全文审计 | IPC总体解释 |
| Inubushi/Yoshimura，2017 | https://doi.org/10.1038/s41598-017-10257-6 | 混合单元有关方法与记录；未核实官方代码 | 连接第二层机制对照 |
| Azzalini，1985；SciPy | https://www.azzalini.it/SN/abstracts.html；https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html | 原作者书目/潜正态背景；本地SciPy 1.17 _rvs实现 | W1精确条件律背景 |
| Shevtsova，2011 | https://arxiv.org/abs/1111.6554 | IID Berry–Esseen常数与条件 | W1近似误差界 |
| Hoeffding，1963 | https://doi.org/10.1080/01621459.1963.10500830 | 独立有界和的尾界 | W3配对场景收益界 |

## 四、代码阅读与复现顺序

| 顺序 | 文件/主要入口 | 先读什么 |
|---|---|---|
| 1 | geometry_generalization.py | 二阶联合矩；PR/c/f/s；公式条件 |
| 2 | task_moments.py | 任意表示的任务m/M；正态权重参照 |
| 3 | finite_gaussian.py | A/B/D分解；p1解析；p>1积分与平局 |
| 4 | approximation_diagnostics.py | 五风险相减；BE plug-in |
| 5 | task_structure.py | 任务支持、正交化标签、H3解析例子 |
| 6 | higher_order_controls.py | 同η的控制变量；独立拟合/选择/估计 |
| 7 | task_spectrum.py | 能量与任务功率、harmonic、PCR噪声 |
| 8 | continuous_risk.py | 向量目标风险恒等式与增益 |
| 9 | readout_design.py、paired_readouts.py | 候选、选择、共享索引、独立初始化、配对界 |
| 10 | run_focused_review.py | 冻结合同与13视图分类执行 |
| 11 | run_continuous_extension.py、run_higher_order_controls.py | 连续恢复与估计方差补充 |
| 12 | tests/test_theory.py、tests/test_v0_2.py | 已知答案、边界条件、抽样/初始化分离 |

### 推荐命令

安装requirements-runtime.txt；纯理论部分安装requirements-reference.txt。本次验证环境Python3.12.14、NumPy2.3.5、SciPy1.17.0、Torch2.14.1+cpu。输出目录必须不存在，避免混入旧证据。

~~~bash
python -m unittest discover -s tests -v
python verify_v0_2_artifacts.py

python run_v0_2_controls.py --output outputs/known_controls.json

python run_focused_review.py evidence/source_smoke_run \
  --output outputs/classification --smoke \
  --all-schedules --carriers --task-priors --mlp

python run_continuous_extension.py evidence/source_smoke_run \
  outputs/classification --output outputs/continuous

python run_higher_order_controls.py evidence/source_smoke_run \
  outputs/classification --output outputs/higher_order_controls
~~~

保存证据另通过54个Python文件AST与97个JSON检查；792个分类scene-loss数组、1164个权重参数数组、39个训练索引合同及39个连续权重组均复核一致。verify_v0_2_artifacts.py复核已打包证据，不重新拟合模型。

本轮分类MLP累计7680个vectorized optimizer steps，覆盖1536个单任务头拟合；限上述3个角色组合，不是39组合全做MLP。当前v0.2试验不合并旧v0.1的1920/1920步或更早整合运行来凑样本。

### 输出与历史

- evidence/validated_v0_2_thirteen_views：本轮主分类、冻结合同、每scene损失、权重及报告。
- evidence/validated_v0_2_continuous_native：原生头注释与连续恢复；独立5,700,000场景。
- evidence/validated_v0_2_higher_order_controls：9项高阶估计控制与已知H3方差结果。
- evidence/known_controls_v0_2_continuous.json：当前完整已知控制。
- evidence/test_log_v0_2.txt：55项测试通过，含Torch配对检查，无skip。
- figures/：从保存JSON重建的图；绘图脚本为根目录plot_v0_2.py。
- history/v0_1_code：旧代码快照；旧evidence和旧3.5–3.7内容均保留，不计新增独立证据。

## 五、到这里3.4完成了什么，哪些是下一阶段

本轮完成的是**可审阅的研究设计、来源对照、必要推导、实现修订、已知控制与冻结SNN集成验证**。本节已有足够明确的实验合同，可以直接连接正式训练后的诊断。

下一阶段仍需正式训练模型、多主模型seed、充分矩样本量扫描、调参ridge/更强读出及公平成本对照，才能把候选现象写成正式架构发现。自然数据集下载继续由用户安排，本轮只采样项目既有生成器并取选定公开源码，未下载实验数据集。

接3.5时，应沿着本节已经固定的“任务/读取合同”讨论膜状态、脉冲与其他状态载体；接3.6/3.7再分别讨论跨时间访问与跨层到达。这样第三层各部分有明确职责，同时共享同一条论文论证。



## 六、图件说明

figures/L3_4_v0_2_known_controls.png与SVG显示六个已知答案控制：有限p Gaussian、任务支持翻转、低能量目标、PCR噪声、有限样本读取、分类与连续恢复。它们校准分析工具，不代表正式训练后的SNN发现。

figures/L3_4_v0_2_frozen_SNN_evidence.png与SVG显示单checkpoint的分类风险、全部39组合的条件收益区间与连续恢复对照。分类图使用同p标签，连续标量增益额外用2048个moment标签；图中明确标注。两张图均由plot_v0_2.py从保存JSON重建，未重新选择结果。
