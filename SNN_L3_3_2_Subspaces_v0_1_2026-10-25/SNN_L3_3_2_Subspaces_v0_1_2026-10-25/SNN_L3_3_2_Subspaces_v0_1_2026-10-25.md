# 3.3.2 编码子空间、条件协方差与指定读出的限制

2026-10-25，v0.1。承接3.3.1和第三层深化复盘。性质：研究协议、理论推导、可审阅实现及流程验收；不是完成正式训练后的科学结果。

## 0. 本节属于什么、为什么要做

本节属于第三层的“表征组织与读出后果”。3.3.1已经将当前考察目标F、其他共同目标G、未请求内容U及未解释部分分开；本节进一步判断：这些内容的一阶编码占据哪些方向，方向估计是否可靠，它们的强度、条件变异和指定读出如何共同联系恢复误差。

本节在正文中应承担一段完整论证：**同样保留了某份内容，表示组织不同，指定读出所遇到的限制也可能不同；必须把组织、可读性和实际利用分别测量。**

内部顺序固定为：

1. 明确编码方向、解码方向与协方差度量；
2. 独立估计方向、选择维数并检查稳定性；
3. 测几何重叠、幅值和合适的随机参照；
4. 在明确定义的条件协方差下分析一阶编码；
5. 用具体读出的映射和误差验证几何的任务含义；
6. 限制观察、固定读出和重新拟合读出，测学习曲线；
7. 检查排列、维数、样本量与解释边界。

上述顺序对应下文3.3.2.1至3.3.2.7，不按“夹角、PR、MI各写一节”的方式组织。Wakhloo的四统计量及Hebbian泛化预测仍留在3.4；跨时编码、载体和通信分别留在3.5–3.7，核心状态干预确认在L4。

## 3.3.2.1 三种方向首先不能混为一谈

### 目的

避免把一个分类器或回归头的权重当成“信息保存的位置”，也避免把投影到编码子空间预设为有效的改进方法。

继续联合编码模型：

\[
r=\mu+A_F\widetilde F+A_G\widetilde G+A_U\widetilde U+e,
\qquad \hat F=Wr+b.
\]

| 对象 | 定义 | 回答什么 |
|---|---|---|
| 编码列方向 | A_F的列及其列空间 | F的线性变化怎样对应群体观察变化 |
| 解码方向 | W的行及其行空间 | 这个读出怎样组合观察以预测F |
| 协方差度量下的编码方向 | Sigma^{-1/2}A_F | 相对于声明的变异结构，编码方向怎样排列 |

编码方向是内容到观察，解码方向是观察到目标。它们不必重合。Sigma必须附带条件，例如给定F和排列的平均条件协方差，不能只叫“noise”。

### 一个应贯穿本节的例子

令F、U独立、零均值、单位方差，观察为：

\[
r_1=F+U,\qquad r_2=U.
\]

则：

\[
A_F=\begin{bmatrix}1\\0\end{bmatrix},\quad
A_U=\begin{bmatrix}1\\1\end{bmatrix},\quad
W=\begin{bmatrix}1&-1\end{bmatrix}.
\]

编码方向夹角45度，却能精确读出F=r1-r2。第二个坐标与F的边际线性相关为零，但能提供抵消U的参照。

若只保留目标编码方向，就只剩r1=F+U，最佳均方误差从0变成.5。由此可见：

- 非正交编码不自动意味着无法恢复；
- 解码权重偏离编码方向不自动意味着头没有对齐目标；
- 编码子空间的正交补不自动等于无用方向；
- 观察限制损害性能，不能立即解释为删掉了同等数量的直接目标信号。

另一反例：r=F、W=-1。编码与读出行空间完全相同，目标MSE却为4。子空间指标不保留符号、增益和目标坐标映射，必须检查WA_F。

### 文献与我们的转换

Haufe等《On the interpretation of weight vectors of linear models in multivariate neuroimaging》，NeuroImage 2014，区分forward patterns与backward filters，并解释读出权重可能用于抑制非目标来源。本轮读取作者机构版第2.3–2.4节及相关讨论。我们利用已知潜变量直接估计A_F，再将W的目标映射和抑制作用单独分析，不把W当编码定位结果，也不将作者变换公式未经条件检查套到不可靠的头上。

代码：subspace_geometry.py::basis；readout_geometry.py::mapping_diagnostics。以上反例进入测试和run_calibrations.py。经典区分不是我们的原创；本研究增量在于将它落实到相同SNN接口、任务角色与机制比较中。

## 3.3.2.2 怎样估计子空间，怎样避免噪声也被算成方向

### 先固定实验对象

固定checkpoint、观察层/接口、期限末端时刻、来源存在性、内容维数和一种合法排列。本轮主观察为last_post；另接入joint_post检查额外观察权限，不能把二者当作同维度观察。

对同一自然输入，仅改变F=C或F=H1的分析标签，不改变模型状态，也不改变联合A。改变需求L后分别训练得到的模型属于另一类比较。

当前主数据是独立标准Gaussian内容，因此内容坐标有明确单位，主几何使用已知Omega=I。若迁移到不同尺度或相关潜变量，A_F A_F^T需改为A_F Omega_F A_F^T，并重新检查因素系数可识别性。

### 估计、选择和复测分开

axis_fit上联合拟合全部可见内容；axis_selection选择每种角色编码的压缩维数；独立axis_replica重新估计联合编码，用于复测。所有矩阵都保持同一神经坐标，不能先把不同角色人为正交化再测它们是否正交。

对某角色的编码矩阵作SVD：

\[
\hat A_F=U_FS_FV_F^\top,
\qquad Q_F=U_F[:,1:k_F].
\]

Q_F只定义所选的一阶线性子空间。不同角色内部的基可以各自正交，但角色之间不施加正交约束。

我们区分三个数：

| 数 | 含义 | 不能解释为 |
|---|---|---|
| 内容坐标数 | 请求或输入的潜变量维数 | 这些维数全部已经被编码 |
| 数值秩 | 给定数值容差的非零奇异值数 | 真正信号维数 |
| 验证选择的压缩秩 | 在独立数据上保留多少模式进行预测描述 | 生物内在维数或显著性结论 |

### 实际选择规则

候选k从0开始，直到该角色矩阵的数值秩。验证时其他角色项固定为axis_fit的估计，只替换当前角色的截断编码。计算每个k的逐场景表示预测MSE。

默认采用预先声明的paired one-SE规则：先找验证MSE最低的k，再选与其平均误差差异不超过该差异一个标准误的最小k。它是保守压缩的启发式，不是显著性检验；验证集不能被称作最终测试。

**k=0是合法结果。** 此时编码子空间为空，夹角与相应重叠记为未定义，不返回“90度、无干扰”，也不强行保留一根轴。它只说明当前数据和估计规则没有选择非零一阶压缩描述，不能证明目标信息不存在。空投影捕获功率为0也只是定义结果，不能作为不同角色没有相互影响的证据。

若截断恰好切在近简并奇异值块中，代码记录边界谱隙。单根奇异向量可在简并块内任意转动，此时应讨论整个稳定块的投影，而非给某一根轴命名。

### 独立性落实到哪里

主重叠比较Q_F^(A)与Q_G^(B)，同时反向比较Q_F^(B)与Q_G^(A)。同一拟合集的重叠保留为次级描述。这样避免共同拟合误差让两个方向看起来特殊。

复测沿用已选择的秩，不使用评价集重新选。Bootstrap独立重采样两个拟合集的场景，给出固定所选秩条件下的分布；它没有覆盖完整选择过程，也不代表独立训练seed的不确定性。

代码：fit_axes、select_role_rank、independent_replica、fixed_rank_bootstrap_overlap。dPCA明确允许不同因素的编码/解码轴非正交，并提醒强行QR可能引入顺序依赖；本节沿用这一边界，当前方法不是完整dPCA复现。

## 3.3.2.3 方向重叠、功率重叠和随机参照要分别报告

### 纯几何重叠

给定正交基Q_F、Q_G，Q_F^T Q_G的奇异值为主夹角余弦。定义：

\[
O_{F\to G}=\frac{\|Q_F^TQ_G\|_F^2}{k_F},
\qquad
O_{G\to F}=\frac{\|Q_F^TQ_G\|_F^2}{k_G}.
\]

两者分母不同，一般不相等。例如一维F空间完全包含于三维G空间时，前者1，后者1/3。必须保存维数、全部主夹角、两个归一化量，不能只给一个未说明分母的“overlap”。

若k_F+k_G>D，至少有k_F+k_G-D维交集；高重叠可能部分由维数强制。若两个空间都满D维，重叠1是必然结果，不能证明所有内容都发生竞争。

### 功率重叠

纯夹角忽略信号强度。以独立replica估计的G线性编码协方差S_G为例：

\[
\eta_{G\text{ in }F}=\frac{\operatorname{tr}(Q_F^T S_G Q_F)}{\operatorname{tr}(S_G)}.
\]

另报告同秩最佳捕获量归一化：

\[
AI(Q_F,S_G)=\frac{\operatorname{tr}(Q_F^T S_G Q_F)}{\sum_{j=1}^{k_F}\lambda_j(S_G)}.
\]

前者问G总线性编码功率中有多少落入F空间；后者与同秩最优PCA捕获量相比。它们都不是信息量。当前S_G=hat A_G^(B)hat A_G^(B)T仍有有限样本估计误差，平方幅值不自动无偏；应与预测表现、复测和样本量共同解释。

Elsayed等2016的alignment index提供同秩方差捕获的参照。2026预印本《Partitioning Neural Co-Variability》附录F沿用了同类归一化。我们的实现同时保留总功率分母和同秩最优分母，并将协方差对象改成有明确内容角色的一阶编码成分；不将它们包装成新公式。

### 四种对照有不同职责

| 对照 | 本版构造 | 回答什么 | 不能宣称什么 |
|---|---|---|---|
| 同维Haar随机子空间 | iid Gaussian矩阵QR | 仅由环境维数与所选秩会期待多少重叠 | 真实神经统计的完整零模型 |
| 协方差塑形随机子空间 | orth(C^(1/2)G) | 若方向受观察协方差约束，参照如何变化 | 精确复制Elsayed Supplementary Note 3采样器 |
| 保留方差最接近的Haar候选 | 只用选择集C从候选中选 | 随机方向的能量差能否被减小 | 未匹配成功时仍叫能量匹配 |
| 同秩同保留方差构造 | 低/高协方差特征空间混合 | 控制秩与总保留方差后比较观察限制 | 均匀条件Haar零分布，或已匹配完整噪声谱 |

各向同性Haar下：

\[
\mathbb E\,\operatorname{tr}(P_FP_G)=k_Fk_G/D,
\qquad \mathbb E O_{F\to G}=k_G/D.
\]

所以高维空间里低重叠本来就很常见；要结合对应维数的参照。

### 同秩同保留方差怎样实际构造

当2k<=D时，取C最低k维基Q_lo与最高k维基Q_hi，并分别在内部作随机正交旋转。两组相互正交，且在C度量下交叉项为零。令其保留方差分别为E_lo、E_hi，目标E位于两者之间，取：

\[
t=(E-E_{lo})/(E_{hi}-E_{lo}),\quad
Q=\sqrt{1-t}Q_{lo}+\sqrt t Q_{hi}.
\]

则Q^TQ=I，tr(Q^T C Q)=E。若k>D/2，先对D-k维正交补匹配总方差减E，再取补空间。不可行目标明确报错，退化等方差情形单列处理。

这只是明确、可校准的一种构造；所匹配的只有秩和总保留方差。代码保留构造方式，不将它视作无偏抽样或显著性分布。

代码：overlap、covariance_capture、null_overlaps、variance_matched_basis，以及run_geometry.py::projection_spaces。

## 3.3.2.4 信号强度、条件协方差和条件数怎样进入

### 同角度也可能有不同可读性

A_F和.1A_F占同一子空间；在相同外部噪声下，它们的恢复误差可以不同。两个几乎平行但线性独立的方向在无噪声时仍可能精确区分，却需要很大读出权重，因此对噪声、有限样本与权重约束敏感。

因此保存完整奇异值谱、非零模式条件数、是否列满秩和选择边界谱隙。秩不足时不把非零模式条件数伪装成整个目标映射条件良好。

### 两个协方差对象必须分开

固定排列，令m(F)=E[r|F]，本节直接用独立重复估计：

\[
\bar\Sigma_{cond}=\mathbb E\operatorname{Cov}(r\mid F).
\]

另一方面，将拟合的一阶目标成分减去，得到线性focal余项协方差：

\[
\Sigma_{lin-rem}=\operatorname{Cov}(r-\mu-A_F\widetilde F).
\]

对固定A_F，在总体上有：

\[
\Sigma_{lin-rem}=\bar\Sigma_{cond}
+\operatorname{Cov}(m(F)-\mu-A_F\widetilde F).
\]

第二项可包含目标自身的非线性编码，也可包含一阶拟合误差。有限样本中本节用不同独立数据估计这些量，不要求样本估计逐项恰好相等。

例如r=(F+e1,F²-1+e2)，测量噪声方差均.04：真实条件协方差是.04I，一阶余项第二维方差却为2.04。将后者全部叫噪声，会把确定的非线性目标成分一起算进去。

### 估计与正则化

noise_fit和noise_selection各自使用独立的focal固定重复场景。重复内重采样所有其他可见内容和测量扰动，固定目标请求坐标与精确排列。所有重复跟随场景，不跨折划分。

对平均条件协方差作：

\[
\hat\Sigma_\gamma=(1-\gamma)\hat\Sigma+\gamma\tau I,
\quad\tau=\operatorname{tr}(\hat\Sigma)/D.
\]

gamma候选为0、.01、.1、.5、1；特征值另有相对tau的1e-6数值下限。用独立selection协方差最小化logdet(S)+tr(S^{-1}C_val)。这是二阶矩的协方差评分；不因此声称SNN数据是Gaussian。

若估计的条件协方差完全为零，程序返回度量未定义，不凭空添加单位噪声后声称获得有限信息量。有限重复修正后的between协方差可以不定，本节不会拿它直接做无正则逆或功率分数。

### 白化几何的解释边界

对声明的正定Sigma，定义：

\[
B_F=\Sigma^{-1/2}A_F,\quad B_G=\Sigma^{-1/2}A_G,
\quad J_F=A_F^T\Sigma^{-1}A_F.
\]

报告B_F/B_G子空间的角度及J_F的谱，称为“该协方差度量下的一阶编码几何”。在SNN中它不是完整Fisher信息，也不是所有非线性读出的容量。异方差和非线性条件均值都可能使全局一阶描述不足。

只有匹配的独立线性Gaussian生成模型r=A_F F+n、F~N(0,I)、n协方差Sigma中，才有：

\[
\operatorname{Cov}(F-\hat F_{opt})=(I+J_F)^{-1}.
\]

已知总体协方差与信号一同可逆变换时，该度量保留相关几何。有限样本下向I收缩依赖原坐标，并不保持一般可逆变换不变性；本轮专门测试这一边界。

代码：choose_covariance_metric、noise_metric_geometry。现有Wakhloo工作帮助明确条件和受限读出问题；其原四量与Hebbian任务预测不在这里执行。

## 3.3.2.5 真正联系任务：检查W如何处理编码与条件变异

对预测q个F坐标的固定头，分别保存：

\[
WA_F-I,\qquad WA_G,\qquad WA_U,
\qquad q^{-1}\operatorname{tr}(W\bar\Sigma_{cond}W^T).
\]

前者包含目标增益和坐标混合不匹配；中间两项描述其他内容的一阶注入；最后一项是该固定线性读出的条件输出变异。

即使W的行空间与编码空间高度重合，也可能有错误的符号或增益；即使两者夹角明显，W也可能恰好借其他方向抑制干扰。r=(F+U,U)的例子恰好同时说明这两点。

对留出数据，仍用上一轮五项精确误差记账：

\[
\hat F-F=(WA_F-I)\widetilde F+WA_G\widetilde G
+WA_U\widetilde U+We+(W\mu+b-\bar F).
\]

所有交叉二阶矩与偏置保留，不把几个正的平方范数直接相加称为实际总误差。

再用focal固定重复面板计算：

\[
\mathbb E\|\hat F-F\|^2
=\mathbb E\|\mathbb E[\hat F\mid F,\pi]-F\|^2
+\mathbb E\|\hat F-\mathbb E[\hat F\mid F,\pi]\|^2.
\]

这一步不要求整个SNN线性Gaussian。有限重复的均值误差减去组内方差/B，负的修正估计保留；直接MSE与两项之和作数值核对。

H1等共同目标的自然数据恢复也单独报告。focal固定重复中G会变化，因此不能把同一面板中的G当作固定目标，硬套上述两项公式。本版仅对真正固定的focal目标做此条件风险分解。

### 与Sharma等ICLR 2026的具体关系

该工作用年龄相关编码方向及其在指定读出上的投影联系损失。本轮读取固定commit中iclr_figure_02_and_06.ipynb的全部10个代码单元：BatchedRNN、数据生成、compute_ws_vectors、compute_as_values、四项损失、训练记录及作图。

需要保留的细节：

- notebook的线性特征方向为W_h^s W_x；这不等于非线性SNN普遍存在同样的固定编码方向。
- 该示例通过tie_readouts=True令W_y=W_h^k W_x，以便清晰展示；论文附录也讨论untied设置。我们的实际任务头独立训练，不能默认它与A_F绑定。
- 该notebook的nu=1/3明确注释为Uniform输入的二阶矩E[U²]；不能未经核对就替换成方差1/12。
- 该notebook从5次示例训练中选择最终理论损失最低的历史用于展示。我们不把这种可视化选择流程用于benchmark平均结论；正式实验需保留预先声明的全部seed。
- 本轮只读源代码，未执行作者训练，也未将该notebook中的损失公式移植成非线性SNN通用定理。

本节转换的核心是：从“内容方向看起来是否分离”进一步落到“这个实际读出如何处理它们”，同时保留SNN的条件和非线性余项。

## 3.3.2.6 观察限制、固定读出、重新拟合读出及样本量

### 观察限制怎样定义

对独立估计的正交基Q，保持训练折均值mu，定义：

\[
r_Q=\mu+QQ^T(r-\mu).
\]

本节只在已经采集的观察上作这个限制，不把投影状态重新注入SNN递归。它是观察/读出诊断，不能作为核心状态因果干预的充分证据。

比较：完整观察、F编码子空间、独立replica的F空间、F正交补、G/U编码空间，以及同维Haar、最近方差候选和同秩同方差构造。

### 三种评价不能合并

| 评价 | 怎么做 | 回答什么 |
|---|---|---|
| 原任务头固定 | 将r_Q送入原W,b | 原系统的既有映射对限制是否敏感 |
| 自然全观察probe固定 | 在完整自然观察上拟合后锁定，再测r_Q | 一个已适配的线性诊断读出是否受损 |
| 限制后的probe重新拟合 | 独立训练/选择数据上学习Q^T r读出 | 这种观察限制下，指定学习规则可恢复多少 |

前两者损害而第三者恢复，支持读出适配限制的解释；仍需检查幅值、分布和观察差异。第三者也差，不能仅凭一个有限线性probe就证明所有信息消失，应调用3.3.1非线性诊断及后续载体分析。

### 一个必须遵守的总体结论

在同一完整观察r上，所有线性读出的函数类包含先作固定线性投影后再读出的函数类。因此总体、无约束最佳线性风险满足：

\[
\inf_{W,b}R(Wr+b)
\leq\inf_{V,c}R(VQ^Tr+c).
\]

投影不能创造原观察不存在的信息。投影后有限样本probe变好，可以来自参数减少、正则和估计方差改变；原头变好也可能只是避开了它原先错误使用的方向。不能将这些结果写成总体信息增加。

### 实际学习协议

probe_fit、probe_selection与evaluation独立。所有观察限制使用axis_fit确定的同一个完整观察均值与标量RMS尺度；不对每根轴分别标准化，避免悄悄改变欧氏度量。

ridge目标采用平均平方误差加lambda范数罚；调用父实现的sum-SSE形式时alpha=N*lambda。lambda候选0、1e-5、1e-3、.1、1，仅由probe_selection选择。

默认训练样本数64、256、1024；smoke为32、128。限制后的输入维数可以降低，但目标的全部原请求坐标仍保留；不能投掉难恢复的目标后重新计算一个更容易的分数。

每行记录观察维数、读出参数量、训练样本数、正则、负R²和相对同N完整观察的成对MSE差。高维补空间与低维F空间参数预算不同，必须看同维随机对照，不能只比较两条最漂亮的曲线。

代码：affine_restrict、fit_projected_probe、paired_error_interval和run_geometry.py。单元测试检查了共同尺度下ridge对正交基旋转的不变性、空观察常量读出和完整观察no-op。

## 3.3.2.7 排列、可靠性、任务与论文中的地位

### 排列与跨条件

主分析在一种排列内部完成；额外支持全部六种合法排列。六种排列复用相同内容与测量扰动，仅改变H2–H4的呈现时刻；它们是同一checkpoint的成对条件视图，不是六个独立训练。

每种排列分别拟合和选择编码子空间，不先汇总不同条件。跨排列重叠只作条件依赖描述；若任一所选秩为0，则该角度未定义。跨时解码失配和编码旋转在3.6进一步处理，本节不凭跨排列图宣布普遍旋转机制。

### 可复用的数据合同

下表为默认独立场景数量，smoke数量括号内。每一行使用不同scene_id区间；重复内目标、排列不变，其余内容与测量扰动独立重采样。跨排列故意复用同一行的数据，以支持成对比较。

| 用途 | 场景数 | 首个scene_id | split |
|---|---:|---:|---|
| 编码方向拟合 | 1024（128） | 2000000 | geometry_fit |
| 秩与几何选择 | 512（64） | 2100000 | geometry_fit |
| 独立方向复测 | 1024（128） | 2200000 | geometry_fit |
| 一阶focal余项协方差 | 1024（128） | 2300000 | geometry_fit |
| 条件协方差拟合 | 256×8（32×4） | 2400000 | geometry_fit |
| 条件协方差选择 | 128×8（16×4） | 2500000 | geometry_fit |
| 自然probe拟合 | 1024（128） | 2600000 | probe_train |
| probe选择 | 512（64） | 2700000 | probe_val |
| 自然评价 | 1024（128） | 2800000 | development_test |
| 条件风险评价 | 256×8（32×4） | 2900000 | development_test |

本版复用父生成器和checkpoint，增加声明清楚的独立分析样本；不改变模型可见输入x，不向模型提供latent或场景标签，不访问audit。

Bootstrap场景区间只反映固定checkpoint内的抽样不确定性。固定秩Bootstrap不是完整post-selection区间；跨训练seed的稳定性必须由独立训练支持。正式报告全部预先声明的条件及seed，不以最优seed取代总体结果。

### 分类与重建的边界

代码初版只接入连续坐标回归、来源全部存在、期限末端last_post/joint_post，因为这些接口能严格映射到当前原任务头。

二分类需先按sign(theta^T z)定义任务方向及其他内容的角色；完整z并非全部必要目标。重建需依赖合法场景生成和区域真值，不能把自然相关历史独立替换成不可能的视频。像素MSE可用相应条件风险逻辑，准确率、BCE、SSIM不能直接套本MSE加法记账。

### 在论文里怎么组织

正文建议把本节压成一个论证单元，图的职责如下：

| 面板 | 读者应看懂什么 |
|---|---|
| A：定义与反例 | 编码轴、读出轴、参照方向的区别 |
| B：独立估计与稳定性 | 这些子空间有可靠的一阶编码依据 |
| C：维数/功率/协方差参照 | 重叠不是高维几何或能量差的自动产物 |
| D：指定读出与条件风险 | 哪个几何变化真的联系任务错误 |
| E：观察限制与学习曲线 | 影响来自内容访问、噪声参照还是有限读出学习 |

不需要在正文放所有原始角度矩阵。完整谱、全部随机对照、选择曲线、协方差敏感性和负结果可放补充材料，但保持结果可追溯。

## 4. 与已有benchmark和最邻近研究的具体关系

| 工作 | 相近之处 | 本节新增的研究职责 | 不应声称什么 |
|---|---|---|---|
| Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects，arXiv:2502.09449v1 | 共同重视时序路径、单元/学习规则及任务有效性 | 从路径是否重要，继续拆到保存的内容、一阶组织和固定读出误差 | STP没有机制分析，或任务数量更多就等于超越 |
| Benchmarking spiking neural networks across sensing modalities on edge devices / Soul，arXiv:2609.00026及官方仓库 | 共同要求跨模型的一致协议和可复核评价 | 在系统能力评价中增加内容角色、观察权限、条件协方差和读出映射这一层解释 | 仅凭当前合成小模型已经超越其多模态与部署证据 |
| Sharma等，ICLR 2026 temporal superposition | 共同研究需求、方向及读出投影的联系 | 对非线性SNN分开统计编码、实际独立头、条件变异和不完整一阶描述 | 固定W_h^s W_x或作者特定损失可直接成为所有SNN的定理 |
| Wakhloo等，Nature Neuroscience 2026 | 共同将潜变量几何与规定读出联系 | 为后续理论检验提供条件清楚的几何和经验读出对照 | 本节已复现其四量泛化公式或预测图像PSNR |
| Haufe、dPCA、Elsayed等基础方法 | 使用编码/解码区别、因素空间和方差捕获 | 接入同一SNN的需求、接口、机制操作与留出预测 | 将这些经典公式重新命名为原创 |

STP本轮复读其v1的任务、STP定义和评价组织；Soul本轮读取官方README全文，当前README blob SHA为1c0cb70ff8405898795cc2ef06c24bbdf5fada75。Soul arXiv正文重新访问受阻，本轮未将README核对冒充重新阅读全文。

## 5. 文献与作者代码的核查范围

| 来源 | 本轮具体核查 | 如何采用 | 尚未做什么 |
|---|---|---|---|
| Haufe等，NeuroImage 2014 | 作者机构PDF的forward/backward模型、2.3–2.4节和相关讨论 | 解释编码与滤波权重区别；建立噪声参照反例 | 未执行作者数据实验或移植变换代码 |
| Kobak等，eLife 2016 | 不同因素轴不强制正交、讨论及QR顺序问题；沿用3.3.1所声明的代码核查 | 各因素内部SVD，因素之间保持原关系 | 完整dPCA条件网格与作者求解器复现 |
| Elsayed等，Nature Communications 2016 | 正式原文、alignment index定义与协方差参照随机空间思路 | 两种功率归一化与明确随机控制 | 未逐式复现Supplementary Note 3采样器 |
| Sharma等，ICLR 2026 | 原文3.2–3.4、相关读出设置和附录；一个固定notebook的全部代码单元 | 内容—读出乘积、tied/untied边界、矩与展示选择规则 | 未执行作者训练，未移植作者损失到SNN |
| Wakhloo等，Nature Neuroscience 2026 | 沿用既有固定版本方法/作者函数审阅，本轮不扩大为完整复现 | 条件对象与规定读出，留出3.4理论预测 | 本节不运行四量理论 |
| Thomas、Zhu、Cullen、Charles，Partitioning Neural Co-Variability，arXiv:2605.06995v1，2026-05-07预印本 | 摘要、结构化协变问题及附录F的alignment公式 | 指明同类方差捕获公式已有2026应用；检查协方差对象 | 没有将其Poisson matrix-normal模型套到确定性SNN膜态，未审作者代码 |
| Libby、Buschman，Nature Neuroscience 2021 | 官方作者介绍与此前索引的问题定位；本轮原文访问受阻 | 保留感觉/记忆避免干扰的问题动机 | 不将官方摘要核对说成方法/代码复现，时间旋转分析后续处理 |

RNN作者代码固定commit：26b00cbb9cc03bcb2b4f9d989ac3a9d48a5f6bf9；文件iclr_figure_02_and_06.ipynb，Git blob b6d9bbfe94941bcb496b1392036a343270b92a98。阅读代码单元0、1、3、5、7、9、10、12、13、14。没有执行作者的任何cell。包中只保存来源/哈希/用途记录，不分发该notebook。

主要原文入口：

- Haufe作者PDF：https://doc.ml.tu-berlin.de/bbci/publications/HauMeiGoeDaeHayBlaBie13.pdf；DOI：https://doi.org/10.1016/j.neuroimage.2013.10.067
- dPCA：https://elifesciences.org/articles/10989
- Elsayed：https://www.nature.com/articles/ncomms13239；PDF：https://www.nature.com/articles/ncomms13239.pdf
- Sharma：https://proceedings.iclr.cc/paper_files/paper/2026/file/bfcb583d225b1db8d3ca2331f6785774-Paper-Conference.pdf；作者代码：https://github.com/kashparty/iclr-rnn-superposition
- Wakhloo：https://www.nature.com/articles/s41593-025-02183-y
- 2026协变预印本：https://arxiv.org/html/2605.06995v1
- Libby作者入口：https://www.timbuschman.com/Publications/rotational-dynamics-reduce-interference-between-sensory-and-memory-representations
- STP：https://arxiv.org/html/2502.09449v1
- Soul：https://arxiv.org/abs/2609.00026；官方代码：https://github.com/yudi-mars/Soul

## 6. 哪些值得作为我们的创新，哪些必须由结果决定

本节不把SVD、主夹角、alignment index、白化、线性MMSE或交叉验证叫作原创方法。

本研究可以形成的具体增量是：在同一SNN系统和受控内容需求下，将独立估计的一阶编码、条件变异、实际读出、观察限制和样本学习难度连接起来；对候选机制给出不同预测，而非只给事后相关图。

建议预先登记三类命题：

1. 若某种改动真正改善“其他有用历史对当前读出的不合适影响”，应在保持G自身恢复的同时降低WA_G相关注入或条件输出变异；仅夹角增大不够。
2. 若改动主要改变读出适配，应出现固定头受损而独立重拟合读出可恢复；若源端膜态仍可读而事件观察不足，则继续转入载体与通信分析。
3. 若所谓目标子空间的优势主要来自降维与正则，同维同方差控制及训练样本量变化应能解释相当一部分差异；不能把这个现象直接写成核心新增信息。

每个命题都需要新的seed、等待长度、排列或工作点上的留出预测，以及L4中相应的干预确认。若关键量与误差不对应，或一阶估计无法稳定，允许该解释失败，继续调用非线性/事件路径分析。

这些是候选研究贡献和执行判据。当前尚无正式训练证据支持哪条机制规律普遍成立。

## 7. 工程交付与已实际完成的验收

### 文件怎样对应分析

| 文件 | 职责 |
|---|---|
| bridge.py | 引入按哈希固定的本项目3.3.1/深化复盘父代码 |
| subspace_geometry.py | SVD、秩选择、复测、主夹角、功率捕获、三类空间参照与固定秩Bootstrap |
| readout_geometry.py | 条件协方差正则与选择、协方差度量几何、WA映射、观察限制、probe和成对评价 |
| sampling.py | 六种合法排列及声明条件下的独立场景/重复 |
| run_geometry.py | 连接既有checkpoint，保存每种排列的完整协议、结果与数组 |
| run_calibrations.py | 已知答案的几何反例及图 |
| run_linear_reference.py | 非平凡线性系统的整个分析链路验收 |
| tests/test_geometry.py | 35项数学与数据合同检查 |
| parent_l331/、parent_deep/ | 13个未经修改的父Python文件；来源哈希保留 |

### 已知答案与线性系统验收

35项检查通过，包括空子空间、强制交集、非对称归一化、Haar期望、协方差参照、噪声参照反例、错误符号头、共同可逆变换、收缩不变性边界、近共线、非线性focal余项、独立秩选择、正交基旋转和同秩同方差构造。

另构造12维观察、20维独立Gaussian因子的线性系统，按相同的独立拟合/选择/复测、噪声度量、probe及Bootstrap流程运行。它的F/G均有4个强线性模式，G在额外空间提供抵消参照，U在其他方向编码。

实际结果：

| 校准量 | 数值 |
|---|---:|
| 已知总体最优MSE | .0195126687 |
| 独立4096样本上的已知oracle MSE | .0196493409 |
| 完整观察ridge MSE | .0199351321 |
| 仅目标编码子空间ridge MSE | .5049829293 |
| 理想目标编码子空间总体MSE | .5024875622 |
| 独立F/G子空间重叠 | .498122 |

F/G/U选择秩均为4，复测重叠均高于.999；48次固定秩Bootstrap全部有效。这些是已知线性系统的分析校准，不是SNN架构比较结果。

### 真实SNN流程验收

复用原lif_hard checkpoint 272ae6230942ebeb；它原先只有1次优化更新。本轮不训练SNN。运行C焦点的六种排列、H1焦点的排列0，以及joint_post的C焦点排列0，共八个视图，仍是同一个checkpoint。

每个smoke视图使用128编码拟合、64秩选择、128独立复测、128余项协方差、128probe拟合、64probe选择、128评价；条件面板为32×4拟合、16×4选择、32×4评价。

这些极小模型中，有的角色没有选择非零秩；相应重叠保持未定义。对于具有非零F秩的视图，随手选出的最近Haar方差候选并未全部匹配成功，程序保留失败标记；新构造对照则直接核对同秩与同保留方差。不能挑选某一视图的角度作为成熟模型的结论。

原头在last_post/joint_post中的嵌入与原forward作数值核对；精确角色风险与条件风险继续核对。最终误差、匹配精度和八视图清单以evidence/verification.json为准。

初次控制迭代保留简要历史记录，不计为额外模型或独立seed。正式benchmark仍为0/435拟议配置；本节完成的是协议、实现与校准。

### 图的读法

figures/known_answer_controls.png展示四个已知答案：编码与读出不同；目标编码投影损失噪声参照；可逆变换时欧氏角度与协方差度量的区别；近共线编码对噪声敏感。图上明确标记不是训练SNN结果。

### PowerShell单行命令

```powershell
python -m pip install -r requirements-reference.txt
python -m unittest discover -s tests -v
python run_calibrations.py --output outputs/known_answers.json
python run_linear_reference.py --output outputs/linear_reference.json
python -m pip install -r requirements-runtime.txt
python run_geometry.py evidence/source_smoke_run --output outputs/c_geometry --all-schedules --focal C --smoke --device cpu
python run_geometry.py evidence/source_smoke_run --output outputs/h1_geometry --schedule-index 0 --focal H1 --smoke --device cpu
python run_geometry.py evidence/source_smoke_run --output outputs/joint_geometry --schedule-index 0 --observation joint_post --smoke --device cpu
```

绘图还需requirements-figures.txt。每次使用尚不存在的输出路径。去掉--smoke后运行默认分析样本量；输入若仍是极小checkpoint，增加分析样本不会使其成为正式训练模型。

## 8. 可写入论文的方法段落草案

We analyzed role-specific linear coding subspaces in a fixed terminal observation of each model. Visible content variables were jointly regressed onto the observation, with coordinates labeled as the focal target, other requested targets, or unrequested content. Coding subspaces were estimated on a dedicated fit split, their predictive ranks were selected on a separate split, and their reproducibility was assessed using an independently estimated encoding matrix. Rank zero was permitted and yielded undefined angular comparisons rather than an artificial encoding axis.

We reported principal-angle overlap, retained coding power and reference distributions matched for subspace dimension, alongside covariance-shaped and explicitly variance-controlled comparisons. These descriptive quantities were evaluated separately from the actual task readout. Conditional covariance was estimated using independent repeated scenes with the focal content and presentation schedule fixed. Its regularization was selected without evaluation data. We distinguished this covariance from the remainder after subtracting a fitted linear focal component, which can include nonlinear focal signal.

Readout consequences were assessed through fixed-head mappings, held-out error accounting, conditional prediction variability, and independently fitted probes under affine observation restrictions. All probes retained the original target coordinates, used a common observation scale, and selected regularization on separate data. Projection effects were interpreted as restrictions of accessible observations and finite-sample learning, rather than as increases in population information or causal evidence about the recurrent core. Presentation orders were analyzed separately using paired content and measurement draws. No final audit data were used to choose ranks, covariance regularization, projections, or probe hyperparameters.

该段只陈述本版方法，不预写任何SNN优劣或机制发现；定稿时应按正式实验实际采用的合同调整。
