> **执行状态更新（2026-10-29）**：本文保留原研究记录；当前代码入口、训练/验收状态、数据清单以[当前状态](docs/CURRENT_STATUS.md)和[执行手册](docs/SERVER_RUNBOOK.md)为准。旧“完成”不自动表示全量训练完成。

# 3.3.3 非线性交互、条件编码与实际读出代价

2026-10-25，v0.2复审修订。承接3.3.1角色分解、第三层深化复盘、3.3.2子空间与读出协议。

性质：可审阅的研究构思、方法推导、独立实现、已知答案校准与旧checkpoint接入验收。正式benchmark训练仍为0/435拟议配置。本节没有新增SNN训练；诊断读出的拟合与MLP更新另行登记。


**本次复审的实质变化。** 本节保留既定问题、因素合同、读出族、原任务损失及8个旧模型视图，补上小交互估计的结构校准、显式情境门控和全部标签预算。随后以3.4把表征成分连到未见分类任务的学习方向、学习波动及可预先选择的读出改进。原始v0.1全文保存在代码包的`history/original_protocol_v0_1.md`，供逐项追踪。

| 本次实际发现的问题 | 已采取的修订 | 对研究的正向增益 |
|---|---|---|
| 方差加法闭合，仍可能出现虚假小交互 | 新增对称差分估计，并保留原估计、独立重复和收敛参照 | 增加一个对“交互存在”有辨别力的控制，减少被估计噪声引导的机制解释 |
| 新估计不一定更适合所有主效应 | 加入二值纯主效应反例，报告两种方法的性能差异 | 针对待识别对象选估计器，避免把一次局部改善当通用优越性 |
| 共享线性读出加情境one-hot，只能改变截距 | 新增复用六个专家头的显式斜率门控及错误门控控制 | 区分情境缺失、映射改变和表示不可读 |
| “N个训练样本”遗漏选参标签 | 代码登记训练＋选择的场景数和标量标签数 | 后续学习效率比较有真实预算基础 |
| 交互功率与任务用途之间仍缺连接 | 3.4增加任务条件矩、Gaussian匹配参照和预先选择的谱读出 | 可以解释何种混合影响学习、预测修改是否有效，而非停在表示图像 |

本次已通过63项测试，并对8个已存视图重新计算主估计，不需要再次拟合SNN；另完整运行一次修订后的schedule0流程验证新的默认路径。所有分析仍复用原来仅做过1次优化更新的同一checkpoint，正式benchmark训练仍为0/435。

## 0. 在整篇论文中，本节究竟承担什么

本节属于L3“表征组织与访问”，服务论文机制解释主线P2。问题是：**一阶编码和几何不足以解释恢复差异时，内容是否以非线性或条件依赖方式参与表示；这种参与对指定任务读出到底产生什么后果？**

| 已有部分 | 已回答的问题 | 本节继续补什么 |
|---|---|---|
| 3.3.1 | 谁是焦点F、其他共同目标G、未请求内容U；联合编码和有限二阶诊断怎样做 | 非线性不能只等于加几个二次项；需要完整的组间作用与条件均值检查 |
| 深化复盘 | 主效应/总效应的pick-freeze、重复风险、事件路径边界 | 不能把多个总效应相加；需要将交互各记一次，并接到实际误差 |
| 3.3.2 | 一阶编码子空间、条件协方差、实际线性头和观察限制 | 一阶方向不稳定或均值抵消时，不能据此判内容不存在 |
| 本节3.3.3 | 组间非线性依赖、有限读出恢复、条件编码及误差归属 | 为后续L2机制交叉与L4操作提出可证伪预测 |

本节不提前完成3.4的Wakhloo四统计量及Hebbian泛化，不代替3.5载体、3.6跨时间、3.7通信子空间，也不把观察分析升级为L4核心状态因果确认。

建议正文论证顺序：先给出线性分析遗漏的现象，再用非线性读出与条件控制判断是否可访问，最后用固定任务头的误差分解判断其实际后果。不要先堆一排非线性指标，再临时解释哪个“显得有趣”。

## 3.3.3.1 先认清三类问题：依赖、可读、实际利用

### 类别怎么识别

| 类别 | 识别问题的问法 | 本节的证据 | 单独不能得出的结论 |
|---|---|---|---|
| 函数依赖 | 改变F及其与其他因素的组合，表示r会怎样变？ | 功能ANOVA主效应、交互和总效应 | F一定可以恢复；变化一定服务任务 |
| 指定规则下可读 | 一个声明了权限、容量与训练预算的读出能恢复F吗？ | 留出线性、二次特征、MLP读出及学习曲线 | 所有读出都能；有限probe失败证明没有信息 |
| 既有系统利用 | 原任务头怎样把这些变化变成预测和误差？ | 原头预测、输出/误差ANOVA、独立条件重复 | 某个输入或神经方向删掉后一定改善 |

三者的方向也不同：编码研究内容到状态，probe研究状态到目标，原头分析研究系统当前实际使用的映射。不能互相替代。

### 贯穿本节的两个相似但结论不同的例子

令F、G独立、均匀取±1。

**例A：只观察r=FG。**

\[
\mathbb E[r\mid F]=0,\qquad
\operatorname{Var}(r)=1.
\]

F的单因素平均效应为0，但F参与的总效应为1。与此同时，给定r仍无法区分F的符号：每个r都对应两个等概率F。因此所有平方损失读出中最佳预测为0，恢复F的总体最小MSE为1。

**例B：观察r=(FG,G)。**

F的组主效应仍为0，但合法读出可以用：

\[
F=r_1r_2.
\]

线性读出不能恢复F，二次读出可以精确恢复。增加的G是可观察参照，绝非分析者偷偷提供真值。两个例子说明，交互依赖、缺少参照导致的歧义、线性读出限制是三种不同原因。

另一个边界：r=F²−1对对称连续F有强确定性依赖，却丢失符号。不能见到“非线性目标功率”就宣布完整目标已保存。

代码：calibration_cases.py；tests中的pure_product、no_reference_sign_ambiguity、quadratic_legal_reference_restores_focal等检查。

### 文献怎样进入

Rigotti等2013《The importance of mixed selectivity in complex cognitive tasks》说明非线性混合选择性可支持更灵活的线性分类。我们借它提出“交互也可能有用”的问题，不将它改写成“所有混合都提高所有任务”。本轮核实正式记录与摘要，全文入口受阻，未声称重新精读其补充方法或复现神经记录。

Langdon与Engel 2025《Latent circuit inference from heterogeneous neural responses during cognitive tasks》的Fig.7比较了可解码方向与行为有效方向：更易解码不保证沿该方向的活动改变能驱动行为。本轮读取其相关正文与操作论证，作为“可读与利用分开”的依据；当前实现没有拟合作者latent circuit，也没有移植其连接扰动。

## 3.3.3.2 分解的随机变量必须先固定

### 四个因素组

固定checkpoint、来源存在性、精确排列pi、观察接口和期限末端时刻，写：

\[
X=(F,G,U,E),\qquad r=f(X;\pi),\qquad p=g(r).
\]

| 因素 | 内容 | 本版怎样抽样 |
|---|---|---|
| F | 当前焦点来源中任务请求的坐标 | 既有独立标准Gaussian坐标 |
| G | 其他来源中仍被任务请求的坐标 | 与F独立，继续保持共同目标身份 |
| U | 可见但没有被任务请求的坐标 | 包含同一来源未请求坐标和其他未请求来源 |
| E | 五次写入的完整测量扰动张量 | 独立Gaussian；作为一个组，不拆成40个因素 |

F/G/U逐坐标定义，不能简单按“当前来源/历史来源”替换。比如demand_only中K=2但可见4维，同一焦点来源的另外两维属于U，不能被无意固定为F。

当前合成生成器具有明确乘积分布。projection和固定RMS缩放在同一配置内保持不变；混合输入重新渲染合法来源身份和写入时刻，仅x进入SNN，潜变量、组标签和排列编号不进入原模型。

当前SNN前向确定。若以后加入随机阈值、随机发放或评价期dropout，额外随机性必须进入明确的独立因素和配对合同；不能让未记录的随机状态破坏共享因素条件。

当某组不存在时去掉该因素。例如全部可见内容均被请求时U为空；sigma=0时E是常量，不作为活动因素。活动组数m≤4，完整子集数2^m≤16。保留空组会把有限样本交叉协方差误差伪装成不存在因素的效应，因此代码直接移除。

### “完整”二字的范围

本版完整覆盖的是**声明的因素组之间的作用**，不是所有潜变量坐标的完整高阶分解。

F组的主效应可以是F内部任意复杂函数；G组可能包含多个历史，其内部交互被合在一起。E组也包含多个写入与传感通道。若要追问某条具体历史或某个噪声来源，须在筛选后的少量设置中重新分组、登记成本，不能把当前15项自动命名成所有神经机制。

组交互阶数也不是多项式阶数。F^7仍是F组的主效应；F×G×U是三组交互，即使它只是三次多项式。阈值函数不光滑也可作平方可积的功能分解，不要求替代梯度或真实导数存在。

### 相关数据怎样变

真实视频、自然语音和预测性历史往往相关。独立替换可能产生非法场景；乘积分布的正交性、唯一分配和误差归属不能原封不动使用。

此时应回到L1，明确合法条件生成、目标依赖和条件采样，或使用明确说明参考分布的分解。把相关坐标任意Gaussian化或随机置换，不能自动恢复因果或合法语义。当前runner明确只接入既有独立合成合同；外部数据适配未完成。

代码：factor_sampling.py::factor_schema、splice、full_lattice、conditional_panels。

## 3.3.3.3 从单因素总效应推进到完整组间作用

### 总体定义

以下h可以是表示r、固定读出p，或预测误差e。固定平方可积向量函数h(X)，在声明的独立因素分布下定义：

\[
h_\varnothing=\mathbb E h,
\qquad
h_S(X_S)=\mathbb E[h\mid X_S]
-\sum_{T\subsetneq S}h_T(X_T).
\]

每个非空成分对其中任何一个因素积分均为0，不同S/T的函数成分在总体上正交：

\[
\mathbb E[h_Sh_T^\top]=0\quad(S\ne T),
\qquad
C_S=\mathbb E[h_Sh_S^\top].
\]

于是：

\[
\operatorname{Cov}(h)=\sum_{S\ne\varnothing}C_S.
\]

对四组，保存4个主效应、6个二组交互、4个三组交互和1个四组交互。每个交互只出现一次。

### 为什么上一轮的主/总效应还不够

主效应I_i=tr(C_{i})，总效应：

\[
T_i=\sum_{S:i\in S}\operatorname{tr}(C_S).
\]

F×G同时计入T_F和T_G。纯FG例子里两者均为1，但总表示方差只有1。把总效应直接相加会重复计账。

主/总效应适合筛查“是否存在被平均抵消的依赖”，完整组间分解用于区分它是哪一类交互；两者用途不同。

### 不另拟合复杂网络，怎样估计每项

取独立完整场景A、B，对每个子集S构造：

\[
H_S=(A_S,B_{-S}).
\]

在独立补全条件下：

\[
V(S)=\operatorname{Cov}(h(A),h(H_S))
=\operatorname{Var}(\mathbb E[h\mid X_S])
=\sum_{T\subseteq S,T\ne\varnothing}C_T.
\]

v0.1使用`legacy_covariance`：对称化样本交叉协方差，分母N−1，空集强制为0，总量为A侧样本协方差。该估计总体无偏，继续保留用于主效应参照与敏感性分析；本轮发现的是它在小交互上的有限样本泄漏，不是总体定义错误。

v0.2对交互分析增加主路径`estimator='symmetric_difference'`。令`Sym(M)=(M+Mᵀ)/2`，对每个子集定义：

\[
\widehat V_D(S)=\tfrac12\operatorname{Sym}\{\widehat{\operatorname{Cov}}[h(A),h(H_S)-h(B)]
+\widehat{\operatorname{Cov}}[h(B),h(H_{S^c})-h(A)]\}.
\]

两个括号项的总体期望均为V(S)；各数组按自身行均值中心化，交叉协方差分母N−1。所有hybrid仍使用原有完整子集，不增加前向。空集结构为0；全集变为`0.5*Cov(h(A)-h(B))`，因此**新旧方法的有限样本总量也不同**，不能以A侧方差检验新估计闭合。

对严格加性函数`h=mu+sum_i h_i(X_i)`，`h(H_S)-h(B)`对S是可加的集合函数，另一差分也一样；Möbius反演因此在任意有限配对设计上给出全部二阶及更高组交互为0。若h忽略某因素，包含该因素的交互也精确为0。这两条结构性质能帮助识别伪交互。

A/B实际空子集交叉协方差继续单独保存。该检验不能单独证明生成器独立；独立性还来自种子、场景ID和重渲染合同。

这一做法借鉴经典相关采样/差分敏感性估计，特别是Owen《Better estimation of small Sobol’ sensitivity indices》中的小指标问题。我们的双向中心协方差实现不冒充该文三输入Correlation2估计器，也不把差分方法本身宣称为新理论。

对V作Möbius反演：

\[
\widehat C_S=\sum_{T\subseteq S}(-1)^{|S|-|T|}\widehat V(T).
\]

完整子集要求最多16组前向，而非对20个内容坐标枚举2^20种组合。空子集就是B，全集就是A，可直接复用。

另用互补子集平方差计算每个因素总效应：

\[
\tfrac12\mathbb E[(h(A)-h(H_{-i}))(h(A)-h(H_{-i}))^\top].
\]

两种估计在总体上对应同一总效应，有限样本不要求恰好相同。与上一轮单因素pick-freeze相近的是共享因素交叉协方差；新增的是完整子集与互斥交互项，而非宣称发明了Sobol方法。

### 有限样本必须保留什么

总体C_S是PSD，但交叉协方差差分的估计可能不定，迹也可能为负。代码保留负数和估计区间，不裁成0、不投成PSD后继续声称原估计无偏。

总方差为0时，归一化比例记为未定义。极小总方差会让比例不稳定，正式解释优先看绝对功率、估计区间和独立复测。多个成分区间跨0时，不能靠归一化柱状图强行命名机制。

这一部分的数学属于Hoeffding/Sobol功能分解及Monte Carlo敏感性分析的已有基础，不作为本研究原创定理。

代码：functional_anova.py::decompose、power_summary。

## 3.3.3.4 表示有交互，为什么还要分解输出和误差

### 同一表示，非线性读出能改变交互的归属

对r=(FG,G)，r的第一坐标含FG交互；读出g(r)=r1*r2后得到F，输出只剩F主效应。

因此，非线性读出的效应不能由状态协方差左乘、右乘一个固定W得到。应对每个合法混合输入先重新计算g(r)，再作同一功能分解。

在线性固定头p=Wr+b的特例中有：

\[
C_S^p=W C_S^r W^\top.
\]

本版对原头实际核对该运输关系。它是矩阵运算合同，也说明哪些状态变化被既有头抑制；不能把大状态交互自动记为大输出误差。

### 再往前一步：直接对误差函数分解

对于指定真值y(X_T)，定义：

\[
e(X)=g(f(X))-y(X_T).
\]

将e按相同因素分解：

\[
R=\frac1q\mathbb E\|e\|^2
=\frac{\|\mathbb Ee\|^2}{q}
+\frac1q\sum_{S\ne\varnothing}\operatorname{tr}(C_S^e).
\]

这样可以包含任意非线性固定读出和任意平方损失目标。对每个H_S必须重算自己的y(H_S)；把A的标签复制给所有混合输入，会把目标变化本身错误计为干扰。

这里的“归属”是函数误差在指定输入分布下的正交分解，不是删掉某因素的因果收益。输入、网络或头改变后，其他成分也可能一起改变。

### 焦点目标只依赖F时，哪些项对应非focal变异

若y=y(F)，所有包含G/U/E的误差成分在总体上等于相应输出成分。因此：

\[
\frac1q\sum_{S:S\not\subseteq\{F\}}\operatorname{tr}(C_S^e)
=\frac1q\mathbb E\operatorname{tr}\operatorname{Cov}(p\mid F,\pi).
\]

剩余的总体偏置与F单组误差合在一起，等于：

\[
\frac1q\mathbb E\|\mathbb E[p\mid F,\pi]-y(F)\|^2.
\]

这与上一轮条件风险恒等式一致。本节用两套独立抽样估计：完整pick-freeze分解与F固定重复。两者有限样本有差异，需要区间与样本量分析，不能要求数值完全相同后才承认理论。

G自身也有真值，另以T={G}评价G头，另构造G固定的重复面板。F固定面板中的G会变化，不能把它直接当成G固定风险分析。

还要保持原任务的权重：父训练有anchored/uniform两种目标，不能把F/G两个平均MSE直接相加冒充原总损失。本版另保存原头全部请求坐标的原任务目标记账。若原损失为sum_j w_j e_j²，先把每个误差坐标乘sqrt(q*w_j)，再调用统一的每坐标MSE分解，便精确保留权重形式。分析统一采用float64，原训练标签/前向的float32舍入差另按数值精度理解。这里的“目标因素内部误差”仍是错误，不等于有用表示。

### 一个具体已知答案

四因素独立、零均值、单位方差，固定预测：

\[
p=F+0.5G+0.7FG+0.2E,\qquad y=F.
\]

则误差是0.5G+0.7FG+0.2E：

| 互斥误差成分 | 总体MSE贡献 |
|---|---:|
| G | 0.25 |
| F×G | 0.49 |
| E | 0.04 |
| 合计 | 0.78 |

F×G只计一次。F的总效应包含这项，并不因此意味着这项帮助恢复F。

若改成预测p=FG，目标也为FG，则风险为0，此时FG属于合法目标计算。若同一固定预测p=FG仍被要求输出F，MSE为2；不要把这个数与“只观察FG时最佳读出的最小MSE=1”混淆。

### 有限样本怎样既保留无偏目标又精确记账

**保留的v0.1/A侧估计。** IID样本上，样本均值平方包含估计噪声。对e(A)的样本协方差使用N−1分母，定义：

\[
\widehat B=\|\bar e\|^2/q-\operatorname{tr}(\widehat{\operatorname{Cov}}(e))/(Nq).
\]

则：

\[
\widehat B+\sum_{S\ne\varnothing}\operatorname{tr}(\widehat C_S^e)/q
=N^{-1}\sum_i\|e_i\|^2/q.
\]

这是一条精确的有限样本代数记账关系。B和单项允许为负。**代数闭合并不能证明每个单项估得准。** 需要下面的不确定性、重复与已知答案校准。

穷举有限总体的校准使用总体分母N，不作上述IID样本均值偏差修正；代码将两类设计显式分开。

**v0.2配对主路径的记账。** 对误差e的差分估计，改用：

\[
\widehat R_{AB}=\frac{1}{2Nq}\sum_i(\|e(A_i)\|^2+\|e(B_i)\|^2),\quad
\widehat B_D=\widehat R_{AB}-\frac{\operatorname{tr}(\widehat V_D(\mathrm{all}))}{q}.
\]

于是`B_D + sum_S tr(C_D,S)/q = R_AB`。由于A/B独立，`E[0.5 Cov(eA-eB)]=Cov(e)`，B_D是总体均值误差平方的无偏估计；有限值可以为负。代码另存两侧均值平方的平均与其有符号差，**这不是旧的Var/N修正**。Bootstrap重采整条A/B及其所有hybrid，始终使用同一估计路径。两种主量都针对相同总体风险，却不是同一份有限样本MSE；各自的区间、总量和偏置项必须成套读取。

代码：error_risk_budget、conditional_risk、run_nonlinear.py中的两个目标组与固定读出循环。

## 3.3.3.5 从依赖到可读：读出族、学习曲线与权限

### 本版实际接入的读出

| 读出 | 本版实现 | 用途 |
|---|---|---|
| 原任务头 | 冻结模型中的真实W,b；joint时把最后层头正确嵌入 | 系统当前实际使用 |
| 线性ridge | 原观察全部坐标，独立选择正则 | 线性可访问性的有限样本参照 |
| 完整二次特征ridge | 线性项、所有平方和交叉乘积 | 检查低阶非线性参照/组合解码 |
| 小MLP | 单隐藏层64，ReLU，AdamW；所有预声明初始化保留 | 检查更一般但仍有限的可访问性 |

二次特征包含D+D(D+1)/2列；off-diagonal乘sqrt(2)，全部使用同一训练集均值与单一观察RMS尺度，不逐轴任意放大。它是声明的读出特征类，不把神经状态假定成Gaussian或Hermite正交码。

ridge目标为平均SSE加lambda权重平方；lambda={0,1e−5,1e−3,0.1,1}。特征数大于样本数时使用对偶解，lambda=0用最小范数lstsq。截距不受惩罚。

MLP学习率固定0.003，weight decay={0,0.001}，每10个epoch及最后epoch在选择集检查；只由选择误差选择衰减与停止epoch。最大epoch为正式分析200、smoke80。默认初始化seed=1,2,3全部保留；smoke只用seed1。不能从这三个seed里只报告最好的一个。

### 训练样本量与容量为什么必须一起报告

默认读出学习曲线N={64,256,1024}；smoke为{32,128}。相同的拟合/选择/评价场景用于比较各读出族；选择集不进入拟合，评价集不选择超参数。

本版记录原观察维数、特征/隐藏维数、实际参数量、拟合样本数、选择样本数、候选数、epoch、更新数和运行时间。不同族没有自动实现完全等参数/等计算；它们回答“此声明规则下是否能恢复”，不能据此宣布所有非线性读出公平优于线性读出。

例如D=32、q=4时，线性头132参数，二次头2244参数，64隐藏MLP为2372参数。joint接口D=64时二次参数迅速增加，MLP与二次头的预算关系也变了，不能沿用last_post下的相近预算印象。

一个probe成功，是存在这种有限规则可读的证据；若原头差而probe好，优先考虑读出适配限制。多个probe都失败时，先检验训练充分性、样本量、观察权限及结构歧义，不能证明信息彻底消失。

### 标签预算复审修订

本节的N是拟合样本数，不能直接称作“N-shot”。例如smoke的128个训练场景还使用64个选择场景，实际有192个有标签场景；32训练点对应96个有标签场景。每个目标维度也对应一个标量标签。v0.2在ridge和MLP返回值中登记`total_labelled_scenes_for_fit_and_selection`及`total_labelled_target_scalars`；旧结果的元数据在复审中补齐。多个正则候选复用同一选择集，不能把候选数再乘进独立场景数。

3.4另给真正每个新任务仅p个标签的固定超参曲线；其大规模预测/开发任务标签预算另列，不交给最终读出训练。

### 不给额外真值参照

不能为了让FG可解，就把潜变量G直接交给原先只允许看r的probe。例B中的G必须来自合法观察坐标。若额外提供排列、来源真值或历史标签，要登记为额外分析权限，单列结果。

对当前checkpoint，所有probe只见last_post或joint_post；并不读取源潜变量作为输入。joint比last多32维状态访问权限，性能差不能直接解释成单元更强。

代码：decoding.py::fit_probe、fit_mlp、predict；run_nonlinear.py::native_map。

## 3.3.3.6 条件编码：排列混合为什么可能制造“零编码”

### 先用反例识别问题

令条件c独立均匀取±1，r=cF。汇总后一阶相关为0；每个固定c内部却能用F=cr恢复。

| 情况 | 可能原因 | 下一步 |
|---|---|---|
| 分条件线性可读，跨条件固定读出差 | 条件依赖编码或映射失配 | 固定reader的跨条件矩阵，比较同条件重新适配 |
| 共享合法非线性读出可读 | 条件线索可能已隐含在观察内 | 保持输入权限，检查非线性与样本量 |
| 只有额外提供c后可读 | 分析者条件能解歧义 | 记作额外权限，不能当原系统恢复能力 |
| 分条件与合法非线性读出都差 | 当前读出族/观察/训练仍可能不足 | 先排除这些，再讨论结构性丢失 |

### 本版六种排列怎样实际运行

继续固定合法写入时刻：

```text
[2,5,8,11,12]  [2,5,11,8,12]  [2,8,5,11,12]
[2,8,11,5,12]  [2,11,5,8,12]  [2,11,8,5,12]
```

这是H1,H2,H3,H4,C各自的写入时刻，不是把sample_id模6当作排列。原生成器每六样本块随机重排，取相同余数不能固定排列。

相同潜变量和测量扰动在六种排列间配对，训练/选择/评价用途之间则独立。

本版输出：

1. 每种排列内独立拟合的线性、二次和MLP读出；
2. 固定源排列读出迁移到全部目的排列的6×6误差矩阵；
3. 共享读出，输入仅为原观察；
4. 共享读出，显式附加六维排列one-hot，标明额外分析权限。

共享读出本版为线性/二次两族。条件特异MLP有跨排列迁移，但共享MLP尚未接入，不将它写成已经完成。

### 显式情境门控：修补原对照的解释缺口

`linear([r, one_hot(c)]) = W r + b_c`只让截距依赖情境，无法表示情境特异斜率`W_c r`。二次特征会包含`r_i*one_hot(c)`，因此它可能仅仅补出门控，不能一概解释为“状态需要二次解码”。

新增`gated_specialist_predictions`：

\[
g(r,c)=\sum_{j=1}^{6}1[c=j](W_jr+b_j).
\]

直接把六个已拟合线性专家换回原坐标系，以正确情境和循环错位情境分别选头；参数块相同、不增加拟合。此门控仍享有额外的排列身份，不能当作原系统自然可用的读出。

旧schedule0—5的存储结果中：F正确门控MSE=0.96232048，错位门控=0.96810300；G分别为1.04416381和1.04310163，G没有改善。这些是小模型流程结果，显示该控制能区分不同目标，不作为情境门控普遍有益的证据。

### 样本和参数如何比较

共享与六个specialist系统均使用六排列×N个配对场景的呈现行，总行数相同；其独立潜变量场景只有N个，不是6N个。specialist系统有六个头，共享系统一个头，分别登记总参数。

同一场景的不同排列不能随机分散到训练和评价中。共享结果若估计区间，须以独立场景为组，不能把6N行当IID。本版保存每个场景跨排列平均的MSE供后续成组比较，没有据此宣称共享读出具有普遍显著优势。

这节是排列条件控制，不是跨时间重新编码的完整研究。年龄随排列变化的来源要结合3.2既有年龄匹配控制；不可把矩阵差异统一命名为神经旋转。

代码：context_audit.py；完整六排列运行后自动生成cross_schedule_audit。

## 3.3.3.7 对SNN机制真正有用的进一步分析

### 一个可以严格推导、又很容易被忽略的零参照

假设每次写入j先经独立无状态编码器phi_j，再进入线性状态递归与线性观察。固定写入时刻后，末态可写为：

\[
r_T=c+\sum_j K_j\phi_j(Z_j,E_j).
\]

这个式子允许编码器非线性，却没有跨写入内容的函数组合。若F和G属于不同写入，任何同时含F和G的功能ANOVA成分都为0。含同一写入未请求坐标的F×U则仍可能非零。

当前无norm/dropout的连续控制，两层核心均为线性膜递归、无发放/复位，符合这一类加性展开。这个简单零参照只在这些明确条件下成立；不能外推到带乘法门、非线性状态、输入相关缩放或其他骨架的所有连续模型。

因此正式比较中，不只问总交互强不强，还应区分：

- 同一次写入内，编码器已经产生的非线性；
- 不同写入内容之间，在核心中形成的组合；
- 核心表示仍加性，但非线性probe自己产生的输出交互。

本版已知答案“独立写入非线性编码＋线性状态”保留F×U效应而跨写入F×G效应为0。另有完全无记忆的静态阈值AND单元，F、G主效应和FG交互各为1/16，说明有交互本身也不能证明具有时序机制。

此外，已在实际交付骨架中作代数校准：复制同一旧checkpoint的全部权重，仅把副本的单元前向改为continuous，采用float64。对64组合法输入计算四角有限差分r(Fa,Ga)−r(Fa,Gb)−r(Fb,Ga)+r(Fb,Gb)。跨写入FG的最大差不超过7.2e−15。将可见4维内容的前2维标为请求、其余归U后，F×U可以来自同一编码器调用，副本对应差分最大约1.0355。

这验证的是当前代码满足推导中的加性零参照，并确实区分同写入与跨写入组合。它是**同权重的前向算子覆盖**，不是重新训练后的连续模型基线，也不是硬复位优劣的科学结论。K2仅改变该参照中的分析角色，不计为已训练K2配置。原checkpoint保持不变；结果见evidence/actual_architecture_additivity_reference.json和run_additivity_reference.py。

以上是已有线性叠加/功能分解性质的推论与校准，不作为原创SNN定理。

### 沿既有L2/L3计划实施，不另开无限网格

| 候选问题 | 首先看什么 | 必要比较 | 结果如何指导下一步 |
|---|---|---|---|
| 阈值是否将历史内容变成条件门控 | 状态FG/FGU等项与事件统计；输出是否同样变化 | continuous、threshold、analog_hard_ref、lif_hard、lif_soft；同编码器/同预算 | 若仅probe输出出现交互，不归因给核心；若核心出现，继续事件路径控制 |
| 某种复位是否降低有害影响 | 原F头非focal条件变异与F误差 | 同时保留G自身恢复与任务误差、原头/重训头对照 | 若F改善伴G崩溃，不叫更好的多内容管理 |
| 测量扰动是否被内容门控 | FE、GE及高阶含E项 | 相同噪声强度与信号幅值；E显式分组 | 平均条件方差相同也可有不同门控结构，继续局部条件重复 |
| 一阶编码弱但非线性可读 | 线性/二次/MLP学习曲线，合法参照和条件 | 固定接口、独立选参、容量/样本登记 | 优先检查读取适配；不急于增加状态或宣布遗忘 |
| 输出变化很小但状态交互大 | 原头运输与非线性实际输出 | 原头、全部共同目标头及未来时刻 | 可能当前被抑制或暂时不使用；未来作用留3.6/3.7 |

当前8视图接入没有执行上述细胞正式训练交叉，不能从单个极小lif_hard模型的交互估计给细胞排名。

### 交给L4的预测要写成什么

例如，若一个设置的原F头误差主要来自F×G输出项，则预先预测针对该通路的操作将降低F的非focal条件变异；同时要求G头自身可恢复、目标增益没有全面崩溃，并与等幅/随机/工作点匹配操作比较。操作强度在选择数据定好，再到新场景、新seed或新排列检验。

如果作用只存在于某个重新训练的probe而原头不使用，L4预测应改成读出适配，而不是宣称删除状态就改善原模型。若干预破坏补偿参照，预期甚至可能反向；3.3.2的r=(F+U,U)反例仍然适用。

functional ANOVA不标定唯一神经通路，统计交互也不等于某条具体突触连接。完整状态、事件路径与恢复操作仍按L2/L4合同确认。

## 3.3.3.8 可靠性、抽样和数值验收

### 固定的数据用途

| 用途 | 默认样本 / smoke | split | 首个场景ID |
|---|---:|---|---:|
| probe拟合 | 1024 / 128 | probe_train | 3100000 |
| probe选择 | 512 / 64 | probe_val | 3200000 |
| 自然评价 | 4096 / 256 | development_test | 3300000 |
| ANOVA独立A | 4096 / 256 | development_test | 3400000 |
| ANOVA独立B | 4096 / 256 | development_test | 3500000 |
| F或G固定重复 | 256×8 / 32×4 | development_test | 3600000 |
| 条件成分函数补全 | 64×16 / 8×4，可选 | development_test | 3700000 |

同用途的六排列复用内容/噪声形成配对；不同用途ID区间分开。所有重复补全RNG显式纳入split，即使以后复用相同ID，也不让不同split共享重复噪声。排列编号故意不进入补全RNG，便于配对不同排列。

本版固定规模位于这些区间间隔内；未来若更改到超过10万场景，需重新分配ID区间并校验，不直接沿用。

### 哪个区间是什么不确定性

对整个A/B/全部hybrid匹配行成组Bootstrap，不能对不同mask分别打乱。默认200次，smoke32次；报告逐项百分位区间和相同功率单位下的max-deviation联合近似区间。

这描述固定模型、固定读出、固定因素与排列下的Monte Carlo误差，不包含训练seed、超参数选择或事后选择交互的不确定性；也没有声称普遍严格覆盖率。本版不输出15个未校正p值后挑显著者讲故事。

正式分析预先指定关键聚合量：F头非focal变异、F×G项、含E项、三组及以上交互、G自身误差。其余细项先作探索，新的留出条件用于验证。

### 独立条件重复如何校验

固定T所需目标因素，重采样其他因素，B次重复后计算：

\[
\widehat R_{mean}
=\operatorname{MSE}(\bar p,y)-\widehat V_{within}/B,
\qquad
\widehat R_{var}=\widehat V_{within}.
\]

两者之和精确等于面板直接MSE；修正项允许负。与ANOVA聚合量的比较是独立估计器交叉检查，不能把它们的差异当作额外新物理效应。

### 条件成分“函数”也接入，但不夸大

对同一批base场景，为每个子集S固定X_S，用独立补全估计m_S(X_S)=E[h|X_S]，再逐场景Möbius反演得到成分函数估计。全集必须重放同一完整输入，空集则全部补全。

代码保存每个条件均值的Monte Carlo方差/B与估计成分。成分之和重构base响应是代数性质；有限B下的估计成分不保证真正正交，不能直接把其样本平方当无偏总体功率。主功率结论仍来自前面的独立pick-freeze协方差，函数面板用来检查条件形状及筛选后续局部分析。

不把这些组函数硬贴成“这群神经元专门编码FG”；也不拿有限补全结果重新定义3.3.2的一阶子空间。

代码：conditional_mean_lattice；--nested选项。

### 已实际通过的校准

1. 53项检查：完整子集、主/总效应重复计数、纯二组/三组交互、非线性解码转换、线性协方差运输、有限负值、误差偏置修正、条件重复、空因素、合法混合、跨split重复独立、条件权限、probe预算与MLP随机状态。
2. 四个独立二值因素的全部16×16个A/B组合穷举，交互分解有精确总体答案；固定有害头风险0.78，每项0.25/0.49/0.04。
3. 合法观察(FG,G)的独立训练/选择/评价：线性probe MSE约1.00156，二次probe约1e−30；MLP两个初始化分别约1.47e−10、2.61e−9。它们是人工已知答案，不是SNN任务结果。
4. 同一概率预测例子中，准确率100%，Brier为0.0525，说明Brier分解不能替代准确率分解；二坐标重建已知误差为0.065。
5. 已知Gaussian误差函数0.5G+0.7FG+0.2E，每种N运行16个独立估计seed。非focal总变异真值0.78，其估计RMSE为：N=128时0.2131，512时0.1243，2048时0.0554，8192时0.0277。它们是估计器seed，不是SNN训练seed；不能从这些数保证普遍收敛速度或区间覆盖。

这组结果也说明smoke的256对场景只适合验收流程。即使风险加法闭合到机器精度，单个高阶项仍可能明显不稳定。

## 3.3.3.9 逐篇文献怎样进入、作者代码究竟看了什么

以下只列对本小节有具体职责的文献；原研究索引的其余文献仍可调用。新的检索不追补为10/24快照已完成的阅读数。

| 文献 | 本轮真实阅读范围 | 用于本节什么 | 独立实现/复用边界 |
|---|---|---|---|
| Rigotti等，The importance of mixed selectivity in complex cognitive tasks，Nature 497,585–590，2013，DOI10.1038/nature12160；https://www.nature.com/articles/nature12160 | 正式记录与摘要；全文本轮受阻 | 非线性混合可有任务价值的基础动机 | 未重跑神经记录或作者模型；混合不自动等于本任务受益 |
| Park等，Tensor Product Neural Networks for Functional ANOVA Model，ICML/PMLR267,48041–48085，2025；https://proceedings.mlr.press/v267/park25d.html；方法选读https://arxiv.org/html/2502.15215v4 | 2.2–3.2尤其sum-to-zero/可识别性；下述源码 | 提醒组件分解需声明参考测度与识别约束 | 当前已知乘积分布直接做Monte Carlo分解，不把普通加法网络分支命名为唯一成分，不移植TPNN |
| Park等，Bayesian Neural Networks for Functional ANOVA Model，ICLR2026；https://proceedings.iclr.cc/paper_files/paper/2026/file/29cd189d6e68a1b2ea7c4bff619fcb39-Paper-Conference.pdf | 摘要、2.3–3的零均值定义、高阶组合成本与方法设置 | 高阶项预先截断和组合成本的近期方法参照 | 本版仅4组完整枚举；未执行MCMC、后验或作者代码，不宣称得到贝叶斯不确定性 |
| Langdon、Engel，Latent circuit inference from heterogeneous neural responses during cognitive tasks，Nature Neuroscience28,665–675，2025；https://www.nature.com/articles/s41593-025-01869-7 | 主文相关机制论证、Fig.7及操作解释 | 将可解码与实际行为作用分开，为L4提出更强验证 | 本节未拟合latent circuit或移植其连接操作 |
| Xiong等，In-context superposition: human-like working memory interference in large language models，arXiv2604.09670v3，作者标注COLM2026；https://arxiv.org/html/2604.09670v3 | 第3节、A.5.7–A.5.13及下述选定源码 | 必须面对的近邻：内容重叠、读出对齐、抑制操作已存在 | 本研究需进一步区分非线性组作用与指定误差；操作参数独立选择并验证迁移，不能只换成SNN |
| Dang、Chen、Constantinidis，The cost of efficiency in flexible neural representations，bioRxiv v1，2026-05-19，DOI10.64898/2026.05.18.722885；https://pmc.ncbi.nlm.nih.gov/articles/PMC13228564/ | 摘要、引言与三策略问题设置；不是完整方法审计 | 共享空间的资源节约与正确表现可能有张力，候选解释要和任务后果一起看 | 未审计/运行其RNN训练与神经刺激代码；不把预印本发现当SNN普遍规律 |
| Sobol′2001，Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates；https://doi.org/10.1016/S0378-4754(00)00270-6 | 经典方法入口与定义核对，沿用深化复盘的基础 | 主效应、总效应、独立因素分布 | 完整组子集与向量误差实现为本项目独立实现；数学不是新发明 |
| Jansen1999，Analysis of variance designs for model output；https://doi.org/10.1016/S0010-4655(98)00154-4；Saltelli等2010，Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index；https://doi.org/10.1016/j.cpc.2009.09.018 | 正式入口和估计器背景核对；未声称本轮重新阅读全文 | 互补平方差及设计/估计器需要区分 | 没有移植作者敏感性库，也不宣称本文采样器优于其估计器 |

### 2025 ANOVA-TPNN作者代码审阅

固定仓库https://github.com/ParkSeokhun/ANOVA-TPNN ，commit：

```text
4a4dec0173f9b693b0317adb20b0aca1eb39ab84
```

逐行读models/tpnn.py完整文件和models/README；tpnn.py Git blob为18b8f040be4decd0c77dc97b62fb7b1bfc298113。重点是forward中用训练批次bins的比率构造零均值约束，save_id_constants保存评价用常量，以及数据依赖初始化和小于1000样本时的提示。

因此，若未来移植，需要核对参考测度、批次估计、数值epsilon与冻结常量，不能只复制网络结构后宣布已得到同一个可识别分解。本轮没有读完所有trainer/ensemble依赖，没有执行作者代码，也没有将作者源码打包。该commit晚于论文方法版本，不等于确认论文实验使用了完全相同提交。

### 2026 in-context superposition作者代码审阅

固定仓库https://github.com/sakimarquis/in-context-superposition ，commit：

```text
9ff5fd3b94a953ad35c6505c440495c4afbe57c5
```

逐行读analysis/summary_causal.py完整文件，以及analysis/causal_remove.py第1–190行。对应blob分别为c65bb44b76013dd6a7d0f2b6802f9b77bf3212ef、08410ebba103c039e6d88c05ea6e945d0e9c4b37。

核对到两点：其选定实现使用同一cohort比较基准和操作；summary取扫描中最佳准确率并将负收益截为0。论文v3自己也明确说明选择与评价同批试验，独立迁移仍待检验。

我们保留配对思想，但正式研究登记有符号效果；操作由选择数据决定，再到新试验/新条件评价。这是需要执行的验证要求，不是当前已经完成的超越。未执行作者LLM、未审计全部依赖；该仓库提交与v3论文更新时间不同，不冒称逐字对应其最终发布实验。

## 3.3.3.10 与现有benchmark的相近、差异和可能贡献

| 比较对象 | 相近之处 | 本节具体推进 | 不能据此声称 |
|---|---|---|---|
| Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects，原核查arXiv2502.09449v1 | 都重视时序路径与任务表现、机制对照 | 把路径改变继续拆成哪些内容的非线性参与、哪类读出可访问、原头误差中哪项改变 | STP没有机制分析，或本节已经在其全部任务上取胜 |
| Benchmarking spiking neural networks across sensing modalities on edge devices / Soul，arXiv2609.00026及官方仓库 | 都需要一致模型/数据/评价合同 | 增加内容角色、非线性依赖、指定误差和读出权限分析 | 当前合成任务已超越其多模态/边缘部署证据 |
| Sharma等ICLR2026 temporal superposition，沿用3.3.2代码核查 | 都联系内容方向和任务损失 | 加入非线性组作用及非线性读出，不默认SNN存在固定Wh^sWx方向 | 作者线性方向或特定损失分解适用于所有SNN |
| Xiong等COLM2026 in-context superposition | 内容共存、几何重排、读出和抑制 | 在受控独立因素中检验非线性重编码/歧义/实际误差，并保住共同目标，预先要求留出验证 | 首次提出重叠导致干扰、首次将干扰与读出相连 |
| 功能ANOVA/TPNN等 | 使用主效应、交互、识别约束和分布条件 | 将同一SNN的任务角色、核心机制与固定误差对应起来 | 新发明ANOVA、Möbius反演或条件风险公式 |

本轮STP、Soul与Sharma的比较沿用3.3.2已审阅的版本和范围，没有冒称本轮再次逐行读完这几篇原文/仓库。

### 真正值得争取的突出性质

候选贡献不是“我们算了更多指标”，而是形成可被反驳、对设计有用途的规律：

1. **识别恢复受限发生在哪一环。** 一阶看不见→合法非线性可读→原头未利用，与真正结构歧义区分。相应建议分别是读出适配、补充参照/状态或改善写入，不一律增加神经元。
2. **解释多内容任务中的选择性代价。** 某机制降低F头的非目标输出作用，同时是否保留G头的内容和行为，区别于简单删除历史。
3. **定位跨写入组合的来源。** 独立编码器的同写入非线性、核心事件组合、非线性读出引入的组合分开，并用连续加性控制和事件路径分析验证。
4. **让解释作出新预测。** 在新训练seed、排列、等待长度、工作点或架构上检验哪些误差项改变，而非只拟合已有模型的相关图。

只有正式实验和留出验证支持时，这些才是论文贡献。当前协议/代码质量不能代替发现本身。

## 3.3.3.11 论文应该怎么写、放哪些证据

### 正文建议是一段论证，而不是独立指标目录

先选一个在预定开发screen中出现的一阶解释失效案例，用同样本量与接口检查；然后依次呈现表示组作用、合法读出学习曲线、实际头误差和共同目标保留；最后在独立条件重复中确认并提出留出预测。

不能看完所有seed和所有任务后只选一个最好看的案例。开发阶段可以探索，确认阶段用预先锁定条件和新模型。

| 证据 | 正文职责 | 附录补什么 |
|---|---|---|
| 表示F主效应/含F交互与读出恢复并列 | 解释一阶弱不等于遗忘，依赖不等于可读 | 全部互斥项、负估计、样本量和区间 |
| 原F头的误差成分与独立条件重复 | 将非线性组织落到任务后果 | 逐项偏置修正、估计器差异与完整风险 |
| G恢复及任务误差 | 排除通过损害其他有用目标获益 | 每个目标头、角色变体与权重 |
| 六排列迁移和额外条件权限 | 区分条件重编码、共享读出和额外线索 | 配对场景、呈现数、参数和全部矩阵 |
| 核心机制对照及新条件预测 | 支持设计指导 | 训练seed、工作点、失败范围和L4确认 |

### 可写入论文的方法段落草案

We characterized nonlinear dependencies using a grouped functional ANOVA under a controlled product distribution. For each frozen model, terminal observation and presentation schedule, visible latent coordinates were assigned to the focal target, other requested targets or unrequested content. Measurement perturbations formed an additional explicit factor. Inactive factors were removed. The complete subset lattice therefore contained at most sixteen legal input combinations per independent base pair. This decomposition was complete across the declared groups, without claiming exhaustive resolution of interactions within each group.

Closed-effect covariance matrices were estimated from matched pick-freeze responses and converted into exclusive group components by Möbius inversion. Finite negative estimates were retained. The same analysis was applied separately to the representation, each fixed prediction function, and prediction error with the correct target recomputed for every hybrid input. For targets depending only on a specified factor subset, error components involving other factors were related to conditional prediction variability and cross-checked using independently sampled repeated scenes.

We assessed recovery using the original task heads and independently trained linear, quadratic and small neural readouts. Hyperparameters and stopping epochs were selected without evaluation data, and all registered initialization seeds were retained. Presentation schedules were analyzed separately before fixed-reader transfer and shared-reader comparisons. Shared comparisons reported both presentation counts and unique scene counts; an explicit schedule input was labeled as additional analysis access. Uncertainty estimates resampled complete matched base-pair rows and described sampling variability conditional on the fixed predictor. Statistical interaction was not interpreted as information loss, a unique circuit mechanism or the benefit of deleting a factor.

这是方法草案，不预写任何SNN细胞优劣。正式论文需按最终实际执行的样本量、读出族与分析范围修订。

## 3.3.3.12 工程交付、真实完成状态和复用方式

### 本轮新代码

| 文件 | 已实现 |
|---|---|
| functional_anova.py | 完整组子集协方差、Möbius成分、互斥误差、配对Bootstrap、条件风险、嵌套条件函数 |
| factor_sampling.py | 逐坐标F/G/U、E组、合法混合、空组移除、含split的独立补全 |
| decoding.py | 线性/二次ridge、MLP、独立选参、预算登记、导出NumPy推断 |
| context_audit.py | 六排列迁移、共享线性/二次、额外排列权限与场景级MSE |
| task_objective.py、annotate_objectives.py | 原anchored/uniform任务权重运输及已有冻结观察的补充记账 |
| run_nonlinear.py | 冻结checkpoint的完整接入、原头映射、两目标组、重复与可选嵌套函数 |
| calibration_cases.py、run_calibrations.py | 精确有限总体、歧义/参照、门控与无记忆阈值、Brier/重建校准 |
| run_monte_carlo_reference.py | 4种N×16估计seed及独立Bootstrap示例 |
| run_additivity_reference.py | 实际骨架同权重前向参照；float64跨写入零差分与同写入非线性对照 |
| tests/test_nonlinear.py | 53项检查，PyTorch可选测试单独标记 |

parent_provenance.json记录17个字节不变的本项目父文件：既有13个父Python文件、bridge、原sampling和checkpoint/config两个文件。它们不是17次训练，也不是上游论文作者源码。

### 真实接入结果

复用既有lif_hard checkpoint 272ae6230942ebeb，它原本仅训练1次优化更新。SHA256为：

```text
f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e
```

本轮C/last_post六排列、H1/last_post排列0、C/joint_post排列0，共8个视图，仍只有1个SNN checkpoint。

另有上述同权重前向覆盖的加性参照，两种数据角色合同、各64对场景；单独登记，不加入8个原模型视图，也不计为训练基线。

- 原头嵌入最大误差约3.97e−7，为float32前向与float64分析算术的差异。
- 原头的ANOVA协方差运输误差不超过2.0e−17。
- 误差风险记账误差不超过8.9e−16；条件重复记账误差不超过4.5e−16。
- C/H1角色交换后，对应hybrid响应误差0；joint观察最后32维与单独last_post响应误差0。
- 8视图拟合96个probe，六排列共享控制另8个，总104个fit调用；其中MLP为32个fit、SNN接入部分共5120次读出优化更新。
- 已知答案校准另外保留两个MLP初始化。它们与上述probe都不是SNN训练，不计为正式模型seed。
- 新增SNN训练0、SNN优化更新0、audit访问0；正式benchmark仍0/435。

先前临时CPU依赖缺失导致一次流程在形成报告前停止，补齐依赖后重跑到新的验证路径；该失败不计为额外模型/实验结果。最终证据目录和校验以evidence/verification.json为准。

当前小模型和很小分析样本可出现负高阶项、原头差或probe差，均不作为成熟训练模型的机制发现。正式模型训练、跨训练seed稳定性、结构干预和外部任务结果尚未完成。

### 可以直接运行的命令

以下均为PowerShell可用的单行。输出路径必须尚不存在。

```powershell
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -v
python run_calibrations.py --output outputs/known_answers.json --with-mlp
python run_monte_carlo_reference.py --output outputs/mc_reference.json
python run_additivity_reference.py evidence/source_smoke_run --output outputs/additivity_reference.json
python run_nonlinear.py evidence/source_smoke_run --output outputs/c_nonlinear --all-schedules --focal C --nested --smoke --device cpu
python run_nonlinear.py evidence/source_smoke_run --output outputs/h1_nonlinear --schedule-index 0 --focal H1 --smoke --device cpu
python run_nonlinear.py evidence/source_smoke_run --output outputs/joint_nonlinear --schedule-index 0 --observation joint_post --smoke --device cpu
```

去掉--smoke只扩大对输入checkpoint的分析，不会训练SNN。接入正式模型时替换run_directory；--device控制SNN前向设备，本轮实际只验证CPU，未使用用户A40。MLP诊断读出在CPU拟合，成本单独记录。

若仅看NumPy方法，可安装requirements-reference.txt运行校准与多数测试；其中一个PyTorch测试会被跳过。绘图另装requirements-figures.txt。

### 分类和重建目前到哪里

通用误差函数API支持固定概率的Brier平方风险、向量或像素MSE；已有两个已知答案校准。准确率、BCE、PSNR、SSIM不遵守本MSE加法式，单独保留任务指标。分类的最小必要方向与图像的合法相关场景仍需各自合同。

例如分类y=sign(theta^T z)只要求该任务方向。独立标准Gaussian下可先作明确正交坐标变换，把theta方向标为必要因素、正交分量另归未请求内容，再逆变换回原输入渲染；此时不能沿用当前轴对齐回归的F定义。这个原生分类采样适配尚未实现，不把“固定整个z后测变异”冒充最小充分任务条件。

当前SNN runner只接入连续潜变量回归，尚未声称接入原生分类checkpoint或真实图像重建。不能把通用数学API支持写成这些正式任务已全部跑完。

### 本小节已完成与仍待完成

已完成：本节研究问题/类别、条件推导、文献与选定作者代码核查、组间完整分解、合法读出、排列权限控制、独立估计器校准及旧模型8视图接入。

仍待正式实验：3.2既定细胞/预算训练、关键交互的更细来源分组、充分样本和多训练seed、留出条件预测、L4机制确认、真实分类与图像任务适配。

因此3.3.3已经从设想推进为可执行初版，但不等于整个3.3或第三层已全部完成。后续按原层级继续，不以这节新增功能擅自扩大成另一套训练网格。


## 3.3.3.13 本次复审：哪些创新可以真正做强

**研究贡献可以来自已有工具形成的新连接、实验设置和可检验规律。** 关键是新增认识与预测能够被证据支持。我们应保留方法出处，同时主动把这些方法变成对模型诊断与改进有价值的工作，而非因每个工具都有前人就停止推进。

### A. 把“有交互”推进为“怎样改变任务学习”

当`y=y(F)`平衡且因素独立时，Hoeffding成分的条件零均值性质给出`E[y r]=E[y r_F]`。含G/U/E的交互不能直接贡献该Hebbian平均学习方向，却可进入`E[rrᵀ]`和有限样本学习波动。这个连接在3.4给出完整推导、一般目标子集变体、反例及代码校准。

这不要求所有混合都应删除。二次读出可把`(FG,G)`解成F，联合目标也可需要交互；更强读出若把状态组织转换为适合目标的成分，原来不适合线性读出的混合可以成为可用计算资源。于是研究对象从“交互越少越好”变成**交互—任务—读出之间的匹配**。

对应新实验：同一checkpoint/同一载体，比较F、G、FG任务族，追踪m方向、M波动、p标签学习曲线及固定原头风险。再把读出族和预处理当作已登记的操作，检验能否改善未见任务。

### B. 估计问题转化为机制排错能力

32个独立估计seed的校准结果如下。所有数字均为已知函数，非SNN发现。

| 已知函数／N | 原估计FG项RMSE | 差分估计FG项RMSE | 关键解释 |
|---|---:|---:|---|
| 严格Gaussian加性／128 | 0.104881 | 约2.2e−16 | 原方法有限样本可生成伪交互；新方法满足结构零 |
| 真FG交互／512 | 0.092687 | 0.058376 | 该校准下小交互估计有所改善 |
| 纯三组交互／512，真FG为0 | 0.147673 | 0.136375 | 新方法仍有估计噪声，不能保证所有零项都精确 |
| 二值纯F／128，F主效应RMSE | 0.008886 | 0.080037 | 新方法在这个非零主效应上更差，保留原估计有必要 |

实验决策：组交互的第一诊断用差分路径，主效应与总量同时报告原路径，关键结论要求独立重复/样本量扩展。不能按照哪个结果更符合预期临时选估计器，也不能把新方法称作统一最优估计器。

代码：`run_estimator_review.py`；完整12行结果及全部seed在`evidence/estimator_review_v0_2_full.json`。

### C. 让“条件编码”成为具体架构/读出建议

若共享线性差、专家好、显式正确门控接近专家而错位门控差，则优先研究条件依赖读取；若二次读出在不给情境时也好，再检查合法观察是否自带可解混参照。若所有受限读出都失败，先检查载体、时间窗口与抽样量，再决定是否投入机制修改。

这些是可检验的诊断分支，不是保证每个模型都属于某一固定类型。后续可以在未参与规则构造的任务/细胞上测试它是否节省搜索、改善性能或定位失败。

### D. 与最接近工作的区别怎样写

我们不把现有benchmark概括成“只有准确率”。原有时序、模态和边缘评价分别提供任务、学习和部署价值；时序编码文献也已经做了计数控制与时序扰动。我们的突出方向是**内容角色明确的能力认证、成分到学习代价的连接、载体/时间/通路定位，以及利用诊断预测修改效果**。真正的论文贡献由这条链支持的新规律和留出增益构成。

对已有方法的复用不是问题；把复用当作未经核查的发明才是问题。本文应主动争取新的可区分设置、新的机制解释和新的设计结果，不能把工作停在工具清单或谨慎措辞上。

### E. 已做和下一步

- 63项测试通过，包含10项新增结构/预算/门控测试；原53项保留。
- 原8个观察视图完成差分估计、配对区间、原任务加权风险重算，最大新风险记账误差约2.2e−16。
- 修订runner另完整执行一次C/last_post/schedule0，12次probe拟合、640次MLP更新；这是既有语义视图的流程重验，不是额外独立模型或新正式实验。
- 存储视图重算及显式门控本身0次新probe拟合；checkpoint散列保持不变。
- 3.4已接入9个视图、3类目标组，并完成Gaussian、非Gaussian、全零和平局参照；3.5—3.7已有配套独立实现与接入控制。详见新的后续协议。
- 未完成：435配置正式训练、多SNN训练seed、真实分类/图像数据结果及L4机制操作验证。这些仍是最终论文实验的一部分，不能由本轮工程验收代替。

### 本次新增文献/代码入口与读取深度

1. Owen，2012预印本/2013正式《Better estimation of small Sobol’ sensitivity indices》，https://arxiv.org/abs/1204.4763 。核读小指标相关采样与中心化论证；不宣称重现其中所有估计器。
2. SALib，`src/SALib/analyze/sobol.py`，commit `43e4847b99900cfc2c9acf8ffdd668416fb5d4f9`，blob `ab53bccaf4e2c497289d26acbe2457432c18f2ec`。读`analyze/first_order/total_order/second_order`及采样布局；核对差分乘积、方差归一化与bootstrap，未执行其完整分析器。我们的向量全子集差分协方差为独立实现。
3. Wakhloo等2026正式论文及3个核心源码文件的对照，详见3.4。原文已经检验非Gaussian网络与生物记录，不能声称我们首先将它用于非Gaussian码；我们推进的是任务条件方向、方差预算与SNN具体载体/时间条件下的预测、失败解释和设计用途。

旧版第3.3.3.9的论文与源码记录继续有效；其中明确写明的“摘要/选读/全文受阻”不因本次复审自动升级为全文精读。
