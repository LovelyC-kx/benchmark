> **执行状态更新（2026-10-29）**：本文保留原研究记录；当前代码入口、训练/验收状态、数据清单以[当前状态](docs/CURRENT_STATUS.md)和[执行手册](docs/SERVER_RUNBOOK.md)为准。旧“完成”不自动表示全量训练完成。

# 第三层3.7：跨层传播、通信子空间与目标到达

2026-10-27，v0.1。沿用原六层及第三层七节编号，承接已交付的3.6。原3.7总协议10.1–10.6与6条计划文献全部逐项对接；新增2025/2026的动态通信、隐输入、乘性交互及结构稀疏化参照。文献取得、选读、源码核验和实际执行分别登记。

**本节要回答：源端读得出的内容，是否经过真实接口到达接收方？接收方为什么可能读不好、反而读得更好，或只有结合历史/情境才能读出？这些判断能否指导修改？**

English abstract: L3.7 links local content recovery to actual inter-layer messages and receiver states. It separates statistical predictability, known structural transport, task-content arrival and specified output effects. We combine rank-selected regression, repeated-trial controls, finite-message/precision analysis, Gaussian information references, explicit I_min FIT/cFIT and matched-world replay through the real SNN edge. Low linear rank does not generally determine the information carried by a finite spike alphabet; receiver history and context can change the interpretation of instantaneous decoding. The package provides executable analyses, selected author-source checks and exact replay verification. Model findings remain development evidence on one old, one-update hard-LIF checkpoint; formal benchmark training remains 0/435.

## 0. 在论文中的责任与贡献重点

3.5问“看哪个载体”，3.6问“在什么时候、用什么读取规则看”，3.7进一步问“系统真正送出了什么、接收方得到什么”。本节承担第三层论证C，为L4的内容选择性干预、恢复和实际任务收益提供对象。

四种证据明确分开：

| 层次 | 具体判断 | 不能替代的后续判断 |
|---|---|---|
| 统计共享 | 源活动能在留出数据上预测接收活动 | 不自动说明目标内容、真实方向或直接路径 |
| 结构传递 | 实际代码将哪个返回值通过哪个算子送给谁 | 不自动说明接收状态保留多少目标或原生头是否有效利用 |
| 目标到达 | 指定目标在消息、接收状态或合法历史窗口中的可恢复性 | 不自动说明原生输出使用了该内容 |
| 指定路径作用 | 在明确图、输入和替换规则下，改变该路径怎样影响接收和输出 | 不自动形成跨模型的内容选择性归因或设计优越性 |

本节不是七个并列指标章节。主文宜围绕三个连续判断组织：**局部可读为何不等于当步发送；当步发送为何不等于接收状态/输出效果；诊断能否预测哪种接口或核心改动有效。**

| 候选贡献的落点 | 沿用的已有方法 | 我们要形成的实质认识 |
|---|---|---|
| 同内容的跨层定位 | 解码、RRR、已知输出映射 | 对齐来源、阶段、真实消息和完整接收状态，区分发送、历史累积与读取问题 |
| 适合SNN的传输分析 | 有限字母表、噪声、精度、信息分解 | 联合消息碰撞、读出限制、时序窗口和接收历史，避免用单一秩/局部几何代替有效传输 |
| 诊断驱动改进 | 配对操作、留出选择和恢复 | 在匹配成本的正式模型中预测应改发放接口、接收更新、精度还是读出，并验证目标收益和非目标代价 |

已有方法和相关工作不会取消上述研究价值；同样，借用工具或在一个旧模型上看到现象，也不自动成为已确认的跨模型规律。

### 本轮实际执行范围

| 项目 | 已完成 | 解释范围 |
|---|---|---|
| 模型 | checkpoint 272ae6230942ebeb，原训练1次optimizer更新 | 同一冻结模型；本轮新增SNN训练0 |
| 真实接口 | 返回output→第二层affine输入逐值一致；current和pre方程核对 | 当前两层前馈MLP，不冒充全部骨架的完整图 |
| 内容读取 | 12观察×5来源×6时刻×p128/512，2个共享draw | 288次联合矩阵求解，分类/连续恢复共1440个相关风险单元 |
| 数据划分 | 512校准、1024训练池、2048选择、8192新评价场景 | 每个独立读者仅取128或512标签；额外无标签状态访问另列 |
| 训练draw | p128合计242独特场景；p512合计763独特场景 | 两draw共享训练池，不是独立SNN训练seed |
| RRR | 47条同刻/时滞统计链接，7个rank候选含0 | rank独立选择；不是47个神经网络 |
| 主选择 | 10个同时配对比较，正置信下界0个 | 数值改善尚未被本族区间确认 |
| 补充比较 | 已冻结读者的全部3960个成对比较 | 事后构造完整比较族并重放；不是预注册或新的独立确认 |
| 条件重复 | 128内容×16噪声试次，按内容64/32/32划分 | 48条均值/残差/配对打乱RRR记录，不把试次当独立内容 |
| 真实边替换 | H1/C两种配对世界，各128场景、4种操作 | donor完整后缀与baseline patchback均精确重现对应接收/输出 |
| 精度 | 4096新场景、2时刻、全精度及2/4/8/12/16bit电流 | 已知权重的逆读者；不是最优译码误差或硬件能耗 |
| 验收 | 42项测试无skip；含12单元真实MLP等式及96未训练单元×骨架观察子案例 | 子案例不计为训练后架构结果 |
| 作者代码 | 16个固定源码文件匹配Git blob；选定Python函数执行及FIT公式转写对照 | 未执行完整作者训练、MATLAB/Octave或原始人脑/动物实验 |
| 正式benchmark | 0/435，audit_test未访问 | 未下载真实实验数据集，未使用用户A40 |

主分析CPU约40.17秒，是本实现的运行记录，不是硬件速度benchmark。下文文件名均相对本代码包根目录。

## 3.7.1 真实接口与阶段：先知道系统实际发送什么

### 属于什么问题，为什么先做

若把“研究者能观察到的膜状态”当作“下层实际接收的信号”，就属于**观察权限与真实通路混淆**。后面的通信子空间、信息转移甚至干预都可能对错对象。

在当前模型中，当步先计算源pre，再从pre产生事件，并更新源post。发放与post是由pre产生的两个分支，不能画成post导致同一步spike。

\[
p^1_t=\beta_1v^1_{t-1}+I^1_t,\quad
e^1_t=\mathbf1[p^1_t\ge\theta],\quad
v^1_t=\operatorname{reset}(p^1_t,e^1_t).
\]

前馈MLP下一层调用实际使用的是Cell返回的output：

\[
o^1_t=\operatorname{CellOutput}(p^1_t,e^1_t),\qquad
I^2_t=2(W_2o^1_t+b_2),\qquad
p^2_t=\beta_2v^2_{t-1}+I^2_t.
\]

当前hard-LIF中output是发出的二值事件；continuous/tanh为模拟输出；analog_hard_ref虽然内部存在事件和hard reset，返回值默认却是pre。这正是必须观察实际返回值的原因。若发出事件被覆盖，natural_event、emit_event、返回output也可能不相同。

### 文献怎样进入

Stavisky等，2017，*Motor Cortical Visuomotor Feedback Activity Is Initially Isolated from Downstream Targets in Output-Null Neural State Space Dimensions*，Neuron，DOI10.1016/j.neuron.2017.05.023。其BMI实验使用已知下游映射，使output-null/potent有明确对象。本轮选读原文关于BMI映射、早期/后期活动和相关控制的方法，未核实独立官方完整分析仓库，未执行其动物任务。

我们采用“先明确真实下游映射”的实验原则；进一步使用已知SNN代码直接核对每一步到达的张量，并将源状态、实际返回消息、接收current、接收旧状态和新状态分别保存。

Liu、Sacks、Golub，ICML2025，*Accurate Identification of Communication Between Multiple Interacting Neural Populations*，还把局部动态、区域间消息与未观察输入分别建模。我们知道当前网络图，因此优先核对真实对象，再判断统计方法能恢复什么；不必先拟合一个潜变量模型来猜已知接口。

### 实现及适配

communication_capture.py的capture_edge同时观察源Cell返回值和接收affine层真正收到的输入，不能只读Cell内部的natural_event日志。接收Cell另导出previous_v、base_current、effective_current、beta以及存在的a/syn等辅助量。

主模型全评价池的实际到达误差为0，current方程最大差约1.19e−7，pre方程逐值差0。前者浮点误差不被隐藏成结构不一致。

两状态细胞的base_current经syn滤波后才成为effective_current；ADLIF还会扣除a。一般式应写为：

\[
p^2_t=\beta_2v^2_{t-1}+I^{2,\mathrm{effective}}_t,
\]

不能把所有单元的base_current都直接代入hard-LIF公式。12注册单元的MLP接口子案例已检查这些等式。这里是接口验收，未训练12种单元并比较能力。

### 骨架变体具体改哪里

operator_observer.py可被动记录全部8参考骨架中的命名模块调用，保留完整空间形状及调用顺序。96个未训练单元×骨架子案例的原输出与加观察后的输出逐值一致。但函数加法、拼接、attention乘积和cache不是独立nn.Module，不能据这些日志自动宣布已经恢复完整DAG。

| 骨架 | 必须识别的真实接口 | 与当前MLP适配相比要改什么 |
|---|---|---|
| mlp | 相邻Cell返回值→affine→接收Cell | 当前已实现完整指定边捕获和替换 |
| recurrent_mlp | 当步前馈消息＋历史cache的循环输入 | 分开两条输入和cache；替换后可能反馈到未来源，不能套“源状态不变”的证明 |
| temporal_conv | 当步嵌入＋前两步cache经过各tap | 标注各tap来源年龄，不能把cache贡献算成单一当步Cell传输 |
| cnn1d | 完整传感器轴卷积消息 | 保留轴位置；改变一处位置会影响卷积邻域 |
| cnn2d | 完整空间特征→卷积/归一化→Cell | 明确空间索引和归一化观察权限，不能无说明平均池化 |
| sew_resnet18 | 主支与skip支合并 | 一条支路的null不等于整个网络null，操作需要覆盖相应cut或单列旁路 |
| spike_attention | Q/K/V乘积、投影及残差 | 有效算子依赖输入/状态；不能将其写成固定W乘单个spike向量 |
| unet | 下采样、上采样与skip拼接 | 同时记录跨尺度通路和空间对应；局部边替换不代表切断全部来源 |

**研究用途：** 为每种骨架定义合法的通信观察和操作，使后续关于“膜中有、下层没有”的判断对应真实代码，而非层名或可视化位置。

## 3.7.2 共享子空间与内容传输：RRR能回答什么

### 识别类别与目的

若结论主要来自源活动预测接收活动的R²，就属于**统计共享问题**。它首先回答活动的可预测结构；目标内容传输需要另做标签关联、噪声分析或真实接口实验。

Semedo等，2019，*Cortical Areas Interact through a Communication Subspace*，Neuron102:249–259，DOI10.1016/j.neuron.2019.01.026。作者在源/接收群体中分析trial-to-trial残差，用RRR刻画跨区共享方向，并与区内主要变异比较。主文结果和STAR Methods相关部分已核读，作者44页PDF已取得；没有声称本轮逐图复现所有补充结果。

官方代码joao-semedo/communication-subspace固定fd0e408ba2244d9369bda59f326480386c770e5c，选读ReducedRankRegress.m、RidgeRegress.m、RegressModelSelect.m。原代码保留rank0、常量通道恢复和1SE选择规则；本项目没有把这些已存在的设计宣称为首次。

### 先把数学目标说清楚

对训练集中心化后的X、Y，本项目声明目标为：

\[
\min_{\operatorname{rank}(W)\le r}
N^{-1}\|Y-XW\|_F^2+\lambda\|W\|_F^2.
\]

令C=XᵀX/N+λI，W_full=C^{-1}XᵀY/N。完成平方后，目标仅差常数地等于：

\[
\|C^{1/2}(W-W_{\rm full})\|_F^2.
\]

因此对C^{1/2}W_full作SVD，取前r个右奇异向量V_r，得到：

\[
W_r=W_{\rm full}V_rV_r^T.
\]

这是相应惩罚目标的秩约束解。另一常用流程是先ridge、再对训练预测XW_full做PCA；本项目保留为prediction_svd选项。它是明确的两阶段算法，一般不等于上述惩罚目标的最优解，除非满足额外条件。

本轮数值例子中rank1的惩罚目标约为26.28583与26.29518。差别不意味着两阶段算法无用；需要正确标注其优化对象，不把“先后两步”不加条件地写成与原目标完全等价。MR-LFADS修订稿Appendix A.2及Semedo代码提供了这个核查入口，我们将数学目标和实际算法分别登记。

### 原作者代码中的具体适配问题

Binish等2026正式论文官方仓库nehabinish/pfc-m1-communication-subspace固定40783c621765ac8ca1b37b51b3764cdbd88867e9。原RRR方法已选读和实际调用：

- 当输入4通道、其中1通道恒定时，选定调用返回含截距4行的系数，而原坐标需要5行；该分支没有恢复到完整输入坐标。
- 显式use_ridge_init=False的选定调用在返回best_alpha时触发UnboundLocalError。
- 默认ridge初始化直接使用X，再计算中心化预测；与本项目统一中心化惩罚目标的约定不同。

这些是固定版本中的最小调用结果，不能据此推断原论文实际数据的预处理、所用版本或人脑结果有误。本项目选择保留常量坐标并使用正则化，显式保存源/接收均值与完整维数，避免SNN大量零发放通道引起坐标错配。

### 本轮RRR与内容指标并列

run_communication_review.py包含47条链接：两个源观察(sender_post、实际sender_output)到current/pre/post的同刻比较，以及真实output→current的前后一步时滞参照。rank候选为0/1/2/4/8/16/32；校准512场景拟合，2048独立场景选rank，8192新场景评价。

R²的均值基准来自训练集，允许负值；零目标变异时返回未定义，不强制得到好看的数值。负/正时滞能预测不意味着真实方向；当前真实边是同一步中的拓扑先后。

在t12：

| 统计链接 | 选定rank | 留出响应R² | 解释 |
|---|---:|---:|---|
| actual output→receiver current | 8 | .99801 | 接近已知实际线性接口 |
| sender post→receiver current | 8 | .56797 | 统计预测，不是实际传输载体 |
| actual output→receiver post | 8 | .22566 | 接收旧状态、重置等使当步关系不同 |

反例另设源x=(目标,nuisance)，接收只取10×nuisance加小噪声：共享预测可以极高，但关于目标的接收Gaussian Fisher严格为0。

**与已有方法的相近与推进：** RRR、rank0与噪声残差均有前人。我们的比较单位是同一内容在真实发送接口、接收阶段和历史窗口中的访问；RRR与真实算子和目标恢复互相核对，不独自承担“传递了目标”的结论。

## 3.7.3 共同输入、接收旧状态与重复噪声

### 这类问题如何识别

源、接收活动相关，可能来自真实消息，也可能来自共同输入、相同任务均值、接收方已有历史或未记录状态。时间滞后和去均值并不会自动消除全部混杂。

Liu等ICML2025的MR-LFADS正面处理局部动态、已记录区域消息和未观察输入的区分；还使用已知多区域网络及拟合时未见的扰动验证。因此不能把“用已知网络检验通信推断”或“留出扰动预测”称为我们的首次贡献。本轮同时查阅了2026修订的arXiv v5相关方法与附录，正式出版身份仍标ICML2025。

MacDowell等，2025，*Multiplexed subspaces route neural activity across brain-wide networks*，Nature Communications16:3359，DOI10.1038/s41467-025-58698-2，提供动态共享维度与网络组织的参照。正式XML中的相关Results/Methods已选读；官方仓库的README、GenerateFigures.m、Figure5.m为绘图与预生成MAT数据流程，不能写成取得了作者RRR完整训练实现。

### 已执行的三种控制

**一，共同输入已知答案。** 构造X=Z+η_x、Y=Z+η_y且没有X→Y边。X可以很好预测Y；加入可见Z后，X的额外预测空间基本消失，而结构do(X)对Y的效应为0。它说明预测强度与路径作用需要不同证据。

条件控制也不能机械地“越多越好”：若控制变量是路径中介，可能移除要测的作用；若是碰撞变量，可能引入关联。真实SNN优先根据已知图决定控制对象，潜变量模型则需要自己的识别假设。

**二，接收旧状态。** 当前pre包含βv_previous与当步effective current。我们比较只给previous_v与同时给previous_v、actual sender output的冻结预测器。对于hard-LIF，这些量具有明确更新关系；对于a/syn/caches，必须扩充控制状态。额外预测改善只称为指定旧状态条件下的增量预测，未冒充非线性条件独立或因果效应估计。

**三，固定内容重复噪声。** 生成128份完整内容与相同schedule，每份16次测量噪声重复。按内容划分64训练、32选择、32评价，分别拟合：原响应、条件样本均值、试次残差、仅在同内容内部打乱配对的残差。

\[
X_{i,r}=\bar X_i+\delta X_{i,r},\qquad
Y_{i,r}=\bar Y_i+\delta Y_{i,r}.
\]

原始和残差中的一行试次不是一个独立内容；评价单位仍是32个内容组。均值/去均值也使用额外重复试次权限，不可称为单次在线观察。

当前t12实际output→current的残差R²约.9999998；同内容内部打乱残差配对后约−.00229。实际output→receiver post的残差R²约.13971，打乱后选择rank0、R²约0。这与真实接口能传递共同测量波动相符，但**传播噪声不等于传播了目标内容**。

### 作者源码核查的具体启发

MR-LFADS官方commit aae0a3b317626312bc3c1a4e9408aaa25b48fb32：

- 原Communicator在CPU的选定函数调用已执行。mask_hn遮住指定源坐标；仿射消息的bias仍可存在。不能把“源输入被遮住”默认解释为“消息本身全零”。
- 原volume函数在隔离后执行，来源/接收索引检查最大差0。它计算消息L2幅度，幅度本身不是目标信息或路径贡献。
- 固定metrics.py在第41行有语法错误；本轮未静默修复后冒充运行原模块，只执行独立抽取的volume函数。官方README明确要求GPU训练，未进行其完整训练。

**设计用途：** 以后选择性操作应分别规定输入坐标、消息、bias、接收旧状态和其他通路的处理。把一个变量置零不等于阻断全部路径；接收状态能恢复内容也不等于内容来自当前消息。

## 3.7.4 SNN消息、有限字母表、精度与时间窗口

### 为什么不能只看秩或欧氏null-space

Kaufman等，2014，*Cortical activity in the null space: permitting preparation without movement*，Nature Neuroscience17:440–448，DOI10.1038/nn.3643，依据输出映射区分output-null/potent。Stavisky等2017又展示输出不可见性怎样随时间改变。这些工作为本节提供已知映射与时间敏感性的分析对象；本轮未核实二者完整官方分析代码，不冒充其数据复现。

SNN还有一个必须补入的条件：**实际发出的消息可能只取有限二值模式，任意欧氏方向未必是合法消息变化。**

对消息集合E和线性映射B，无噪声时真正发生碰撞的条件是：

\[
\exists e\ne e'\in\mathcal E:\quad B(e-e')=0.
\]

存在非零ker(B)本身不够，还要它与合法差集E−E相交。给二值消息作任意正交投影，通常会产生分数值，不再是相同的二值发放接口。代码对二值边替换强制检查合法性；模拟电流注入必须另行命名并登记权限。

### 一个可精确枚举的参照

8个独立二值通道e_j经一个标量通道发送：

\[
m=\sum_{j=0}^{7}2^je_j.
\]

线性算子rank为1，但256个合法消息全部不同；均匀输入下m携带完整8bit。任意目标bit都可由合法的非线性/查表读者恢复。它不是说“一维连续表示免费拥有无限容量”，而是说明rank不等于有限消息的bit数。

若接收只保留m的高4bit，即floor(m/16)，消息数降为16，低4bit目标的Bayes分类错误为.5，高4bit为0。finite_messages.py按明确有限支持与概率精确计算分类/MSE风险，不把有限观察样本中的不同模式数当作真实SNN容量估计。

这些原理与Shannon的离散消息、噪声和精度框架相接；编码理论本身不是本项目的新发明。我们要研究的是：**在相同内容、实际SNN接口、合法消息和资源约束下，这种区别是否改变能力判断与设计选择。**

### 代价来自间隔、噪声和读出

将256个消息压在单位范围，最小间隔为Δ=1/255。均匀等间隔一维消息、加独立Gaussian噪声σ、最近邻恢复完整消息的精确错误为：

\[
P_{\rm msg}=2(1-M^{-1})\Phi\!\left(-\frac{\Delta}{2\sigma}\right).
\]

M=256时，σ=.001得到约4.97%消息错误，σ=.01约84.12%。因此低维打包必须同时核算动态范围、精度、噪声和读取复杂度，不能由无噪声例子直接推导低功耗优势。

### 当前真实SNN的已知算子与精度实验

run_interface_precision.py检查当前32×32层间算子，rank为32，条件数约62.69。已知权重与bias时，全精度current可逆恢复实际sender输出；4096个新场景、t12/t15均恢复全部二值消息。这说明在该具体接口与精度条件下，低任务读取成绩不能简单归为affine映射的不可逆丢失。

精度实验先用独立512校准场景、跨时间固定每通道min/max，然后才生成4096新场景。对32维current进行2/4/8/12/16bit量化，使用同一已知逆映射和.5阈值恢复消息。

| t15的current精度 | 逆读者bit错误比例 | 完整消息错误比例 |
|---|---:|---:|
| 原全精度 | 0 | 0 |
| 2bit/坐标 | .27448 | .99902 |
| 4bit/坐标 | .02544 | .81372 |
| 8bit/坐标 | 0 | 0 |
| 12bit/坐标 | 0 | 0 |
| 16bit/坐标 | 0 | 0 |

这是指定逆读者的误差，不是量化接口的Bayes错误；受约束的消息译码可能更好。8bit下所查场景零错误也不是任意输入、任意噪声的保证。32个8bit current坐标的名义载荷为256bit/步，原32个二值通道为32bit/步，二者不能被当成相同通信预算。

### 可以进一步用于设计的精度充分条件

设精确仿射current为i=Ae+b，无clipping，通道量化范围为r_j，ADC采用b_q位均匀舍入。已知逆读者的逐坐标误差满足：

\[
\|\hat e-e\|_\infty
\le\frac{\big\||A^{-1}|\,r\big\|_\infty}{2(2^{b_q}-1)}.
\]

右侧小于.5时，阈值恢复全部二值坐标。这是三角不等式给出的充分条件；额外数值/物理噪声和clipping要另加误差预算。

当前冻结校准范围给出的理想无clipping上界：4bit约1.9111，8bit约.1124，12bit约.0070。该理论界是在精度结果之后由既有校准矩阵推导/补算，已在记录中注明；未重新选择范围、精度或评价场景。真实评价有极少量坐标clipping，不能把理想界宣称为全部实际输入的硬件保证。

**设计含义：** 传输鲁棒性不仅取决于输出秩，还取决于合法消息的可区分性与算子条件数。正式实验可以检验谱约束、范围校准、发放稀疏度和编码选择怎样改变精度需求，并把理论充分条件与更强译码器、实际功耗比较。

### 当步消息与历史消息不能混为一谈

当前t15、p128的C分类错误：sender post约.20263，实际当步output约.49454，receiver current约.49438，receiver post约.42398。局部post与当步output是不同观察分支；这些数值不是post→output的物理变换因果分解。

过去4步的sender计数约.40193，完整raster约.40874。相对当步output，误差分别降低约9.26和8.58个百分点；在全部3960个冻结成对比较的补充族中，下界分别约3.85和3.17个百分点。

这支持“历史消息窗口比单步观察更有用”的具体判断，**没有证明精细脉冲时序优于计数**。计数是raster的函数，因此最优raster读者不会比最优计数读者差；有限样本、维数、正则和学习规则可以使当前raster读者更差。正式比较需要共享窗口、标签、读者选择与资源预算。

同理，完整接收状态可综合历史消息：

\[
S^2_t=F(S^2_0,o^1_{1:t}).
\]

若初态没有额外目标信息、且不存在其他目标输入，才可比较I(F;S²_t)与I(F;o¹_1:t)；不能改成I(F;S²_t)≤I(F;o¹_t)。当前接收post好于当步current并不违反数据处理。

最后，瞬时输出null不等于永久null。线性参照中，O_H=[C;CA;…;CA^(H−1)]决定有限视野可观察方向。A把当前null方向旋转到下一步potent方向时，当下输出0，下一步可变为1。本轮已验证该参照；SNN真实未来还依赖事件序列与完整状态，不能把一个普通局部Jacobian套到全部跨事件变化。

## 3.7.5 目标方向、相关噪声和接收方的抵消/增强

### 属于什么问题

若已经看到消息到达，却不知道它为何对指定目标有用或无用，就进入**任务方向与噪声、接收状态交互**。Euclidean角度、共享能量和响应R²不足以单独回答。

Weiss与Coen-Cagli，2025，*Measuring Stimulus Information Transfer Between Neural Populations Through the Communication Subspace*，Neural Computation37(9):1600–1647，DOI10.1162/neco.a.17，是本节最直接的理论近邻。正式原文2.1–2.2、相关例子和讨论已选读；本轮仍未核实完整可执行官方仓库。

原文明确区分投影到通信子空间的信息、contributed information、接收噪声后的impactful information及交互项。不能把这些已有区别重新命名后当作我们的独创。

### 源信息为什么不能简单分成两个投影信息相加

对常协方差Gaussian位置族，列向量内容导数为a，噪声协方差Σ>0：

\[
J_x=a^T\Sigma^{-1}a.
\]

设P是通信方向投影，Q=I−P。把a拆成Pa和Qa，可得：

\[
J_x=(Pa)^T\Sigma^{-1}(Pa)+(Qa)^T\Sigma^{-1}(Qa)
+2(Pa)^T\Sigma^{-1}(Qa).
\]

这两个contributed项与“只观察Px或Qx”的信息不同。代码information_partition在正交基中计算投影信息，避免不说明支持空间地逆一个奇异投影协方差。

例如a=(1,0)，Σ的对角为1、相关系数.9：

- 完整源Fisher约5.26316；
- 只看第一坐标为1；
- 第二坐标单独为0；
- 已看第一坐标后，第二坐标的条件增量约4.26316。

第二坐标没有均值调谐，但可以帮助消除相关噪声。它不是额外独立编码了4.26316的信息，而是联合观测的条件作用。

Schur形式把这一点写清楚。分块a=(a_1,a_2)、协方差S，对第二块残差导数和协方差：

\[
\tilde a_2=a_2-S_{21}S_{11}^{-1}a_1,\quad
S_{2|1}=S_{22}-S_{21}S_{11}^{-1}S_{12},
\]

\[
J_{12}=J_1+\tilde a_2^TS_{2|1}^{-1}\tilde a_2.
\]

该恒等式和负交互项的独立控制均已实现。直接相加边缘Fisher既可能少算，也可能重复计算。

### 接收噪声与其他目标路径

若y=Bx+η，η独立、无额外目标调谐，则：

\[
J_y=(Ba)^T(B\Sigma B^T+\Sigma_\eta)^{-1}(Ba)\le J_x.
\]

本实现要求对应协方差正定；未悄悄以伪逆替代不同问题。对低维投影，使用明确基坐标计算。

但接收方若已有目标历史或其他目标输入，应写为y=Bx+cθ+η。在接收噪声Σ_y度量下：

\[
J_y=(Ba)^T\Sigma_y^{-1}(Ba)+c^T\Sigma_y^{-1}c
+2(Ba)^T\Sigma_y^{-1}c.
\]

最后的交互项可能抵消或增强。已实现标量参照：传来导数1、其他路径导数−1、噪声方差1.25，两个单项均为.8，交互−1.6，总信息为0；其他路径改为+1，总信息为3.2。增强例子并不违反单通道数据处理，因为接收方另有目标输入。

这给出正向诊断：**目标已通过current到达，但接收状态可因旧状态、其他内容或更新规则而抵消；反过来，接收状态更可读也可能来自历史协助。** 应分别检查接收输入、旧状态、更新后状态与输出。

### 对SNN如何适配

上述精确参照具有明确Gaussian、常协方差条件。SNN的二值输出、阈值边界和条件均值几何依3.6分别处理：普通前向导数、有限扰动、概率/均值信息不是同一量。本轮没有把真实SNN的RRR、经验敏感性或MSE改名成精确Fisher。

正式模型分析将按来源估计重复噪声与内容变化，比较局部膜、真实消息、接收辅助状态和合法窗口；再用配对操作检查减小噪声耦合或保护旧状态是否真的有益。此处的Gaussian恒等式与控制为这些实验提供可核验参照。

## 3.7.6 情境调制与内容特异信息转移

### 两类问题要分别识别

一类是“第三变量改变源—接收映射”；另一类是“指定目标的新增信息是否与源过去共享”。前者可用条件/乘性回归，后者是FIT的对象。二者不等同于总相关，也不自动等同于因果路径。

### 2026的情境与乘性交互工作

Binish等，2026，*A communication subspace relays context-dependent actions from human prefrontal to motor cortex*，Nature Neuroscience29:1690–1698，DOI10.1038/s41593-026-02290-4。其人类PFC/M1记录将情境相关动作与通信子空间关联。本轮核实原摘要、正式身份、代码可用性声明与官方仓库，选定Python RRR方法已执行；全文人脑分析方法尚未全部重新核读，不能由题名relays推断完成了何种因果干预。

Celotto等，2026预印本v1，*Decomposing the modulation of interactions between neuronal populations*，DOI10.64898/2026.05.14.725145，进一步用低秩三阶张量描述乘性交互通道(MICs)：

\[
A_{\rm eff}(z)=A+\sum_j(w_{Z,j}^Tz)w_{X,j}w_{Y,j}^T.
\]

原文2–3节相关推导、拟合和讨论已选读。其方法先处理Z对源、接收的直接加性作用，再拟合基线及乘性交互，使用交叉验证、置换和约束拟合检查几何成分。数据/代码声明为接受后公开，本轮未取得公开官方实现，未声称复现CP-ALS。

我们已实现独立双线性参照：Y=qX+噪声、q=±1可见。单一不看q的线性映射效果差；加入可见q与X×q项后可恢复。另实现一个外积调制项相对源/接收基线子空间的ss/ns/sn/nn四项权重，作为几何检查。

“情境可以调制通信”和“四部分几何分解”已经有最近邻；我们的推进是把情境落到SNN的阈值、接收旧状态、aux与真实消息上，并核对它究竟改变发送、接收更新还是读取。当前固定W不随情境变化，并不等于整个状态转化的有效关系恒定。

正式扩展中，若Z取自接收旧状态，它是内生变量；不能直接当外生情境来做因果解释。若Z是输入规则或可见时间戳，要登记模型是否实际看到它。若仅分析元数据可见，不把条件读者当免费可部署改进。

### 2026结构稀疏化如何纳入

Zhu与Jia，2026预印本，*State-dependent structured sparsification of cortical signal-transmission networks enhances visual coding*，DOI10.64898/2026.07.19.737501，已核实原始元数据/摘要及v2入口。全文PDF请求受阻或被限流，具体方法和官方代码尚未审计。

摘要给出了状态相关稀疏化、局部性、特征特异性和前馈组织与编码的联系。我们据此保留“边更少未必目标到达更差”的候选实验：在同输入、同负荷、匹配活动/资源条件下，分别改变连接稀疏、局部组织和目标方向，测共享噪声、到达延迟及实际任务收益。取得正文后再复用具体网络推断和扰动参数，不从摘要补造作者算法。

### FIT/cFIT：完整标明定义与代码

Celotto等，NeurIPS2023，*An information-theoretic quantification of the content of communication between brain regions*，提供Feature-specific Information Transfer。定义、讨论及有关SM1公式已选读，官方mcelotto/Feature_Info_Transfer固定fc2f44cb9aebe5e5fe2caf0c389fad253ef9cb23。

本项目discrete_transfer.py采用作者选定实现中的Williams–Beer I_min，而非MMI、BROJA或未经说明的PID。其核心为两个Shared Unique Information原子的最小值：

\[
\mathrm{FIT}=\min\{\mathrm{SUI}(F:X^- ,Y^+\setminus Y^-),
\mathrm{SUI}(Y^+:X^- ,F\setminus Y^-)\}.
\]

SUI由相应I_min冗余之差构造。保留两个原子很重要：一方只持有目标的第1bit，另一方只持有第2bit时，第一个原子可为1，而完整FIT为0。本轮直接转写公式与独立向量化实现最大差约1.07e−15；没有称为执行了原MATLAB程序。

| 明确有限分布 | TE | FIT | 应怎样理解 |
|---|---:|---:|---|
| 直接复制新目标bit | 1 | 1 | 新的目标共享传输参照 |
| 只复制与目标独立的nuisance | 1 | 0 | 总传输不等于目标传输 |
| 目标早已在receiver past | 0 | 0 | 仍有目标信息，但没有该定义下的新增量 |
| 接收payload=目标XOR既有context bit | 1 | 0 | payload单独无目标MI；与保留context联合有1bit |
| 双方各编码目标的不同bit | 0 | 0 | 不能只用一个冗余原子认定传递了相同内容 |

XOR例子没有否定FIT；它说明该度量与“完整接收状态在情境下可用”的问题不同。若接收方合法保留context，则应同时测试联合状态和任务，而非由FIT=0宣布路径无用。

另构造两个结构世界：直接X→Y复制，或共同原因Z同时决定X/Y。它们可有相同观察分布和FIT=1；在指定共同原因参照中，给出Z后cFIT=0。这里的cFIT也不能普遍识别结构边：条件变量若包含全部目标信息，真实直接传递的unique部分也可能被移除。要区分相同观察分布的结构世界，需要额外假设、已知图或操作证据。

### 样本和置换不能省略

32次独立有限样本null模拟中，4值变量、N64的plugin FIT平均约.00875bit，最大约.04003；N1024平均仍约.00063。这些小正数不自动是传输证据。

代码对过大的联合字母表直接拒绝，不默默将32维连续状态或二值历史压成未经说明的几个bin。原文也讨论多变量估计的样本需求、过去时刻选择和更保守surrogate的需求。真实SNN拟采用预先固定的小任务相关符号、训练集确定的离散化、有限样本校准及合适置换；本轮FIT结果为明确有限分布控制，没有宣称完成全维SNN的信息流图。

**研究用途：** 同时保留共享响应、目标新增量、情境协同和已知路径操作。这样既能利用神经科学信息分析，也能看见其度量对象与模型任务之间需要补上的连接。

## 3.7.7 指定路径作用与诊断驱动的设计实验

### 配对世界如何构造

固定128个场景的其他内容、测量噪声、source_present和schedule，仅替换H1或C的潜变量，产生原世界A与donor世界B。make_source_pair重绘合法输入，目标写入之前两世界相同。

随后保留A的输入与源层局部计算，只在真正返回的源output接口送入B的二值消息。源reset和aux不随输出覆盖改变；当前前馈图没有接收方反馈到源，因而源post保持逐值不变。

四种操作为：完整donor消息后缀、仅写入步、仅末步、baseline消息patchback。窗口和规则明确，不作任意分数值投影。

### 为什么完整后缀应当重现donor

在当前无旁路的确定性前馈接收系统中：

\[
R_t=F(R_{t-1},o_t).
\]

若两世界在τ之前接收状态相同，且从τ起接收完全相同的消息，则按时间归纳，所有后续接收状态相同。若原生读出使用的历史前缀也相同，则相应原生输出相同。

这不是由相关性推断的结论。其条件包括完整接收状态、全部有关消息、相同接收噪声及无遗漏旁路；在循环/skip/attention骨架必须重建对应图和操作对象。

### 实际结果与含义

| 配对来源 | 完整donor后缀：接收轨迹对donor最大差 | 原生预测对donor | baseline patchback |
|---|---:|---|---|
| H1 | 0 | 逐值一致 | 接收状态与原生输出逐值回到baseline |
| C | 0 | 逐值一致 | 接收状态与原生输出逐值回到baseline |

两种来源的操作前缀不变，源局部post不变。仅写入步或仅末步一般不能重现donor后续状态，说明不能把整个内容的路径作用压缩成一个当步值。

完整后缀操作使原生预测向量相对baseline的均方变化约.00961(H1)、.01266(C)。这是指定路径在这些配对输入中的输出作用，不是任务性能改善。原目标与donor目标不同，不能把对原标签的损失变化直接解释为改进。

这里已得到具体图、具体消息替换下的精确路径证据；其后还需要L4检验内容选择性：虽然只改了F的输入，消息本身可同时改变F与其他内容的交互组织。不能由完整消息重放推断每个被替换神经元只承载F，或其他目标完全未受影响。

### 读取收益怎样验收

主分析先在2048场景中，从receiver post/output/current/pre/raster4/post+current中为每个来源与p选择一个方案，之后用8192新场景确认。基线为同p、同时刻的receiver post读者，时间平均只覆盖该来源已出现的时刻。

10个主比较都有正的数值差，但同时95%区间全部包含0。本节保留这些结果，不将其包装成已确认的普遍读取收益。

补充的3960族覆盖既有12观察、6时刻、2标签预算、5来源的全部成对固定读者。对每个独立场景先平均固定任务方向与draw，损失落在[0,1]，配对差在[−1,1]。Hoeffding与union bound给出半径：

\[
\epsilon=\sqrt{2\log(2M/\alpha)/N}.
\]

该补充族在主分析之后构造，明确不是预注册；因覆盖既有固定有限候选全集，可对其中选择的比较使用相应同时界。但它仍是已有数据的重放，不是新模型或独立重复；与主10比较族各自95%，不合称全研究95%。连续MSE未套用该有界分类证书。

### 诊断具体怎样改变策略

| 诊断组合 | 优先改动 | 为什么 | 正式验收 |
|---|---|---|---|
| 局部pre/post可读、单步output弱、历史消息好 | 合法窗口读出、发放接口或辅助慢路径 | 先分开发送时机与当步幅度；当前count已有效，不能先认定需要精细时序 | 同历史窗口、标签、状态/通信bit预算；count/raster/模拟对照 |
| actual output→current可逆，但低精度逆恢复不稳 | 范围、条件数、编码与精度联合设计 | 问题在鲁棒性与译码权限，不能只改rank | 理论误差界、clipping、强译码器、原生任务与实际资源 |
| current到达、post/输出变差，旧状态可抵消 | 接收写入门、状态保护、aux或读者调整 | 到达与整合是两个环节 | 固定来流与旧状态控制；F/G/U收益和损伤同时报告 |
| 共享R²高但主要是nuisance或噪声 | 内容相关路由或压缩 | 响应拟合目标可能不符合任务 | 相同rank/活动/监督预算的任务风险与响应风险并列 |
| 只有结合可见情境才可读 | 情境条件读者/接口或状态门控 | 单一映射可平均掉真实条件关系 | 加性/乘性对照、无关情境、标签泄漏与情境访问成本 |
| 当前null、之后变potent | 时间条件保护或迟发读出 | 不把暂时隔离当永久丢失 | 合法前向窗口、完整未来状态及来源身份 |

以上是可直接开展的设计实验，不要求每个改动都成为新神经元。若诊断能在未见模型或条件上预先选对改动，并在相同预算下提升原生分类或重建，就形成对后来架构设计的实质指导。

## 4. 每篇文献的原文、代码与复用身份

下表给出完整题名，避免只留缩写。选读/执行深度和缺口见各小节及references/literature_use_and_plan_coverage.json；“有代码入口”不等于本轮完成作者全部实验。

| 文献 | 原文入口 | 代码与本轮用途 |
|---|---|---|
| Semedo等，2019，Cortical Areas Interact through a Communication Subspace | https://doi.org/10.1016/j.neuron.2019.01.026 | https://github.com/joao-semedo/communication-subspace ；RRR、trial残差、rank0与模型选择参照；选定MATLAB未执行 |
| Weiss与Coen-Cagli，2025，Measuring Stimulus Information Transfer Between Neural Populations Through the Communication Subspace | https://doi.org/10.1162/neco.a.17 | 官方完整可执行仓库尚未核实；信息类型、相关噪声与接收交互的独立参照 |
| Celotto等，2023，An information-theoretic quantification of the content of communication between brain regions | https://papers.nips.cc/paper_files/paper/2023/hash/ca9eaef07eca2a50fc626cb929617b1c-Abstract-Conference.html | https://github.com/mcelotto/Feature_Info_Transfer ；明确I_min FIT/cFIT，原公式转写核对，不改名宣称新度量 |
| Binish等，2026，A communication subspace relays context-dependent actions from human prefrontal to motor cortex | https://doi.org/10.1038/s41593-026-02290-4 | https://github.com/nehabinish/pfc-m1-communication-subspace ；官方示例/模拟数据与选定Python RRR，未运行人脑数据 |
| Kaufman等，2014，Cortical activity in the null space: permitting preparation without movement | https://doi.org/10.1038/nn.3643 | 本轮官方分析仓库未核实；已知输出映射与null/potent参照 |
| Stavisky等，2017，Motor Cortical Visuomotor Feedback Activity Is Initially Isolated from Downstream Targets in Output-Null Neural State Space Dimensions | https://doi.org/10.1016/j.neuron.2017.05.023 | 本轮官方分析仓库未核实；已知BMI输出、时间与接收敏感性参照 |
| MacDowell等，2025，Multiplexed subspaces route neural activity across brain-wide networks | https://doi.org/10.1038/s41467-025-58698-2 | https://github.com/buschman-lab/MultiplexedSubspaces ；三份完整绘图文件，不能当作者RRR训练代码 |
| Liu、Sacks、Golub，ICML2025，Accurate Identification of Communication Between Multiple Interacting Neural Populations | https://proceedings.mlr.press/v267/liu25bh.html ；2026修订v5 https://arxiv.org/html/2506.19094v5 | https://github.com/golub-lab/MR-LFADS ；局部动态/消息/隐输入、已知网络和留出扰动参照；选定原函数执行 |
| Celotto等，2026预印本v1，Decomposing the modulation of interactions between neuronal populations | https://doi.org/10.64898/2026.05.14.725145 | 原声明为接受后公开，本轮未取得公开官方实现；双线性参照及四项几何适配，未复现完整CP-ALS |
| Zhu与Jia，2026预印本，State-dependent structured sparsification of cortical signal-transmission networks enhances visual coding | https://doi.org/10.64898/2026.07.19.737501 | 原始摘要/元数据及v2入口已核实；全文/代码仍缺口，只形成待验证的设计假设 |
| Shannon，1948，A Mathematical Theory of Communication | https://doi.org/10.1002/j.1538-7305.1948.tb01338.x | 离散消息与噪声的基础框架；本轮有限支持例子和误差式单独推导/枚举，不宣称编码理论首次发现 |

原计划6条文献全部路由，新增5条用于补足具体问题。原文没有取得的部分不通过二手概述补造，也不妨碍清楚标注后的独立参照实现。后续取得完整资料时，优先补Binish人脑完整方法、Zhu/Jia v2、MIC正式代码和Weiss可执行实现。

### 固定源码的实际执行身份

references/selected_author_sources.json保存16个文件的完整commit、blob、路径、SHA256和阅读范围。当前检查包括：原MR Communicator、隔离抽取的原volume、原Binish RRR方法，以及明确标注为Python公式转写的FIT对照。原作者完整训练均未执行。

发现实现问题的处理不是缩小研究目标：把常量通道、非零bias、优化目标和观察对象写成明确合同，形成可复用实现和最小反例。公开代码的一个最小调用问题也不扩大成对整篇文章结果的否定。

## 5. 与原计划、两篇核心benchmark及后续层级的关系

### 原3.7协议逐项覆盖

| 原总协议条目 | 本轮落点 |
|---|---|
| 10.1 四种证据层次 | 第0节与3.7.1的真实消息合同 |
| 10.2 Weiss/MacDowell文献与代码 | 3.7.2、3.7.3、3.7.5、3.7.6及完整来源表 |
| 10.3 含rank0的正则RRR | communication_math.py及47条独立选rank链接；另区分两种算法目标 |
| 10.4 Gaussian Fisher与nuisance反例 | 信息投影/贡献/交互、Schur、接收抵消/增强、有限字母表反例 |
| 10.5 真实SNN current等式 | 全评价池及12单元检查；加实际返回值、DC/aux阶段与96被动观察子案例 |
| 10.6 目标到达、接收旧状态、重复控制与路径操作 | 12观察内容读取、48重复面板记录、实际边配对世界重放及设计方案 |

Ma等2025的*Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects*与Du等2026的*Benchmarking spiking neural networks across sensing modalities on edge devices*仍是全篇benchmark的核心比较背景。沿此前已核原文的范围对照，不因本节增加通信分析就断言它们完全没有机制、时序窗口或精度分析。

本节相同之处是比较SNN时序任务、模型接口与资源影响；明确推进的比较单位是**同一冻结模型中的同一来源，实际经何返回消息、何历史窗口和接收更新进入输出**。最终优势必须体现为可复核的新设置、跨模型现象、解释或留出设计收益，不能只表现为多算几个统计量。

### 在论文中怎样写

建议主文在3.5/3.6之后，用一个连贯结果单元组织3.7：

1. 先给一条内容从局部状态、真实消息、接收状态到原生输出的轨迹，明确pre分支和历史访问。
2. 用最有辨别力的反例/理论解释为什么单一R²、rank或瞬时解码不足；完整数学与估计细节放方法/附录。
3. 展示同内容配对的实际路径操作及恢复，再展示诊断预测的改动在未见条件上的收益和代价。

正文不必按七小节各放一套指标排行榜。3.7.4的有限消息/精度/历史区别与3.7.7的路径作用及设计验证最值得成为本节突出内容；其余方法使这些判断可信。

### 与L4分工

本轮已验证指定边在已知图中的精确重放作用。L4接着回答：能否选择性改变F的使用，同时保持G、U和其他工作点；能否用合法恢复排除一般损伤；原生任务是否真正改善。emit-only、reset-only、aux-only与耦合改变需要分别做，不能把当前只覆盖返回消息的操作当成全部生物/神经元机制改变。

跨层hybrid不是自然输入分布中的一个完整世界；但完整donor后缀在此图中使接收方沿自然donor轨迹运行，提供明确接收侧参照。后续复杂骨架仍需检查完整cut、旁路、反馈及恢复对象。

## 6. 正式实验怎样直接接上

### 6.1 内容、目标和访问合同

当前原生模型监督H1与C各4坐标；H2–H4为原训练unrequested U。以C为F时H1是cotarget G，反过来亦然，G不是所有其他来源。每来源另构造16个固定随机二分类方向，同时恢复4连续坐标，共80分类方向和20连续输出。

这些分类是冻结表示上的辅助任务，当前checkpoint的原生目标是回归；连续源坐标恢复也不是图像PSNR实验。原生头对U的误差仅作未监督输出诊断，不当作其训练任务性能。

12观察包括两层pre/post/output、receiver base_current/previous_v、source count4/raster4、receiver raster4及post+current。窗口和联合观察的额外状态、时间与缓存成本明确列出。二值4步raster最多需128个bit；32通道4步计数的保守整数表示最多96bit，而单步消息为32bit。它们不是相同访问预算，也未转换成异步硬件功耗。

### 6.2 模型与因素

优先从原435运行计划的已训练配置中选兼容子集：hard/soft LIF、ALIF或two-state、continuous及analog_hard对照，用来分开发放接口、reset和辅助状态。若某候选不属于原435配置，应独立登记，不能改写已有运行数。

先在MLP与recurrent/temporal-conv中完成完整图合同，再迁移到CNN、ResNet/attention和U-Net。需要匹配K/L、项目年龄、后续写入次数、噪声、状态/观察bit预算与训练充分性；种子是独立SNN训练seed，不是时间点、任务方向、读者draw或不同风险格点。

机制比较不预设SNN必胜。模拟输出对照允许更丰富当步消息，应按精度/带宽计费；辅助状态可能主要通过未来事件输出，应给合法窗口；高维raster要有样本和容量对照，不能默认其有限样本读者应更好。

### 6.3 三组优先实验

| 实验组 | 固定什么，改变什么 | 关键比较与竞争解释 | 可写入论文的结果形式 |
|---|---|---|---|
| 发送与接收瓶颈 | 同内容/来源与模型；改变观察载体、时间窗、接收旧状态条件 | 局部pre/post、actual output/current、完整接收状态；count/raster；同预算读者 | 可重复的瓶颈分型及其与细胞/骨架的关系 |
| 鲁棒传输与精度 | 同任务和来流；改变阈值、噪声、精度、谱条件与活动稀疏 | 真实消息碰撞、强译码器、范围/clipping、原生风险与实际成本 | 有适用条件的精度/噪声预测及设计规则 |
| 情境与选择性作用 | 同感知内容；改变可见规则、旧状态或来源保护策略 | 加性/乘性、情境可见性、F/G/U、配对真实边操作与恢复 | 区分编码、接收整合和使用的变化，并预测改动收益 |

不要把所有分析无差别扩成笛卡尔积。先使用原计划的开发池筛出代表配置，同时保留反例；冻结诊断和规则后，在未见模型/负荷/延迟条件检验P3。audit_test继续封存。

### 6.4 分类和重建怎样适配

分类使用原生类别头与相应任务损失，辅助来源任务仅帮助定位；重建使用合法空间位置/内容控制、连续恢复及原生图像误差，分别报告PSNR/SSIM等。图像卷积接口保存完整空间轴，跳连和多尺度路径单列；不能将通用回归API等同于已完成图像实验。

同一输入的配对改动必须来自合法生成/标注规则。没有明确潜变量的真实数据，应从任务、刺激、行为或受控扰动定义内容，再判断哪些是观察标签、哪些可实际给模型。不能将分析者知道的未来事件或真实目标作为免费接收情境。

### 6.5 可用真实数据与缺口（仅列出，不下载）

| 数据入口 | 可用于本节什么 | 当前限制 |
|---|---|---|
| Semedo相关CRCNS V1–V2：https://doi.org/10.6080/K0B27SHN | paired population、刺激条件、trial残差与任务内容对照 | 生物通路并非完全已知，不替代SNN已知图；需遵守原始数据使用要求 |
| MacDowell原始数据：https://doi.org/10.5061/dryad.gxd2547x8 | 多区域、宽场与电生理的共享/情境分析 | 需按原记录/动物/试次分组，不能把时间bin当独立样本 |
| Binish官方仓库与论文Source Data | 代码示例和情境相关通信分析入口 | 仓库示例为模拟数据；额外真实数据原声明需联系作者，本轮未发送任何请求 |
| MIC2026原文 | 新采集多变量状态/两群体数据的候选适配 | 原声明接受后公开，当前不能列为已取得的可运行数据 |

MR-LFADS及Zhu/Jia相关真实数据按各自原文进一步定位，不猜测具体原始数据版本。此处只补3.7通信验证入口，不替代已有L1真实任务与93条数据候选总计划。

## 7. 代码阅读与复现

### 建议阅读顺序

| 顺序与文件 | 阅读重点 |
|---|---|
| 1 communication_capture.py | 实际返回值、到达hook、base/effective current、previous_v、合法覆盖与窗口 |
| 2 communication_math.py | 两种RRR目标、rank0、信息投影/交互、Schur、有限视野、精度充分条件 |
| 3 finite_messages.py | 合法有限支持、消息碰撞、目标风险、噪声及四项几何 |
| 4 discrete_transfer.py | 明确I_min的FIT/cFIT、变量轴和字母表上限 |
| 5 run_communication_controls.py | 所有人工已知答案与估计偏差控制 |
| 6 run_communication_review.py | 场景划分、读者拟合、rank/访问选择、冻结后新评价 |
| 7 run_communication_extensions.py | 条件重复与同内容打乱；原生实际边配对世界重放 |
| 8 run_interface_precision.py | 先校准范围和冻结，再评价新场景；已知逆与bit成本 |
| 9 certify_chain_contrasts.py | 既有固定读者的全部3960比较；明确重放身份 |
| 10 operator_observer.py | 跨骨架被动模块观察，完整tensor形状及其图覆盖限制 |
| 11 audit_communication_sources.py / fetch_pinned_sources.py | 固定作者文件、原函数与公式转写的不同执行身份 |
| 12 tests/ / verify_l37_artifacts.py | 数学控制、实际路径、无侵入观察及保存结果重放 |
| 13 plot_l37.py | 两组PNG/SVG科学图，仅读取已存证据 |

### 三个核心调用片段

**独立rank选择，而非在评价集选最好rank：**

```python
from communication_math import rank_bank, choose_rank, predict
bank = rank_bank(X_calibration, Y_calibration, alpha=1e-3)
rank, selection_losses = choose_rank(bank, X_selection, Y_selection)
frozen_mapping = bank[rank]
Y_prediction = predict(frozen_mapping, X_evaluation)
```

**精确有限消息参照，不把连续rank当bit数：**

```python
import numpy as np
from finite_messages import binary_words, exact_message_risk
events = binary_words(8)
message = events @ (2 ** np.arange(8))
targets = 2 * events.astype(float) - 1
full = exact_message_risk(message, targets)
quantized = exact_message_risk(message // 16, targets)
```

**通过真实接口做配对世界重放：**

```python
from pathlib import Path
import numpy as np
from bridge35 import load_model, fixed_batch
from load_data import make_source_pair
from communication_capture import capture_edge
model, cfg, metadata = load_model(Path('evidence/source_smoke_run'))
a = fixed_batch(cfg.data, 128, 'development_test', 7750000, 0)
b = make_source_pair(cfg.data, a, 'C', pair_seed=370401)
base = capture_edge(model, a['x'])
donor = capture_edge(model, b['x'][:, :cfg.data.deadline_steps])
onset = int(a['event_times'][0, 4])
hybrid = capture_edge(model, a['x'], donor['sender_output'],
                      range(onset, a['x'].shape[1]))
np.testing.assert_array_equal(hybrid['sender_post'], base['sender_post'])
np.testing.assert_array_equal(hybrid['receiver_post'], donor['receiver_post'])
np.testing.assert_array_equal(hybrid['native_prediction'], donor['native_prediction'])
```

### 运行

```bash
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -v
python run_communication_controls.py --output outputs/controls.json
python run_communication_review.py --output outputs/primary
python run_communication_extensions.py outputs/primary --output outputs/extensions
python run_interface_precision.py outputs/primary --output outputs/precision
python certify_chain_contrasts.py outputs/primary --output outputs/complete_contrasts.json
python plot_l37.py
python verify_l37_artifacts.py
```

输出路径默认必须不存在，避免无意覆盖冻结证据。plot/verifier默认读取包内记录；兼容新checkpoint可用--run-directory，但对应primary/extension/certifier必须匹配checkpoint hash，且实际图适配条件不能跳过。

可选作者核验：

```bash
python fetch_pinned_sources.py --output outputs/author_sources
python audit_communication_sources.py --source-directory outputs/author_sources --output outputs/author_checks.json
```

只取得16个固定公共文本源码，不下载作者实验数据或权重，不运行作者训练。作者环境与本项目环境分开；不把当前可导入的选定函数检查扩大成完整论文复现。

### 数据池和权限

| 用途 | 大小 | split/首ID | 权限与单位 |
|---|---:|---|---|
| 读者训练池 | 1024，单读者取128/512 | probe_train/7700000 | 2个共享嵌套draw，目标标签只用于相应读者 |
| rank/访问选择 | 2048 | probe_val/7710000 | 只选择，之后冻结 |
| 主评价与全族重放 | 8192 | development_test/7720000 | 新场景确认后，补充族仍重放同批场景 |
| 无标签校准 | 512 | geometry_fit/7730000 | 尺度、RRR与精度范围；不是免费访问 |
| 条件重复 | 128×16 | geometry_fit/7740000 | 全内容/时刻固定，测量噪声变化；按内容64/32/32划分 |
| 原生路径操作 | 128及其配对世界 | development_test/7750000 | 只变H1或C潜变量；全部其他内容/噪声/schedule相同 |
| 精度评价 | 4096 | development_test/7760000 | 已冻结原读者、已知权重、额外译码权限 |

## 8. 验收、可复核性和仍需取得的论文证据

42测试通过且无skip。其中96个未训练单元×骨架案例仅检查被动观察不改变输出；12单元案例检查两层MLP实际接口方程。它们与正式训练结果分开。

主记录1440个风险单元、20个保存场景损失数组及冻结模型hash均通过检查。重新生成相同评价池前128个场景，26个捕获数组与20个损失数组逐值一致，最大差0。该重放证明保存/运行一致，不另算统计确认。

代码依赖包含46个从已校验3.6包逐字节继承的Python文件，原checkpoint SHA256保持f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e。包自包含运行依赖，不要求旧3.5压缩包。原作者整仓、完整论文PDF及实验数据不与自有代码混合打包。

正式仍需完成：充分训练后的模型/seed验证，目标特异的接收交互分析，复杂骨架完整通路合同，真实分类/图像重建，诊断预先选择修改的未见条件收益，以及真实资源评估。作者全文/官方代码缺口也继续保留。

本节已使“真实送出—目标到达—接收整合—指定路径作用”成为可直接运行、可逐项审视的研究单元。其在论文中的价值将由跨模型规律、竞争解释排除和可迁移设计收益进一步建立。
