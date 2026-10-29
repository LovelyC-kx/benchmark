# L4.1：观测权限与实际读出——深入分析与执行稿

2026-10-27，v0.1。保持原L4“观察与读出”身份。整体五部分与后续实验规格另见SNN_L4_Framework_and_Research_Plan_2026-10-27.md。本稿将第一部分做透，后面的内容操作与恢复不在此处冒称完成。

**核心问题：最终任务表现来自神经元保留、骨架缓存、读出累计、时间位置权限，还是额外的内部观察？**

English abstract: We audit the observation-to-output route of a frozen SNN by separating carrier access, temporal aggregation and decoding. We distinguish trained native inference, same-head counterfactual access and frozen-core readout refitting. Exact controls show that readout accumulators can preserve history without cell recurrence, symmetric pooling cannot generally recover temporal order, and adaptive weights may introduce additional observation channels. A registered model grid validates native and streaming interfaces. Development results on one previously saved, one-update checkpoint illustrate access-conditioned recovery; they do not establish trained architecture rankings or content-selective causal attribution.

## 0. 本轮实际完成什么

| 项目 | 实际执行 | 不能混称为什么 |
|---|---|---|
| 原计划 | L4.1七条文献逐项说明用途和阅读/代码状态；L4.2—4.5保留规格 | 不是全部L4文献及实验都已复现 |
| 原生接口 | 同一旧模型12个同维G/P组合分解；只有post/last为原训练配置 | 其他11个是相同W的访问反事实，不是训练过的原生模型 |
| 合法读出分析 | 3载体×5摘要，p128/512、3个ridge候选；90个联合矩阵求解 | 不是90次SNN训练 |
| 目标 | H1/C各k维连续原任务；各16个固定方向的辅助二分类 | 二分类是诊断任务，不是新训练的原生分类器 |
| 数据 | 512校准、1024训练池、2048选择、8192新评价 | 时间/方向/共享训练预算不是独立模型seed |
| 风险/比较 | 120个主网格风险单元，另8个训练均值/多数类常量基线；12个同时分类比较 | MSE只报点估计，未套分类有界证书 |
| 全骨架接口 | 1440分类＋288图像重建数值案例；另288个线性轨迹折叠验证 | 全部未训练，测的是G/P/R和streaming一致性 |
| 已知答案 | 9类控制：摘要、尺度、原头null、未来null、NoTD累计、概率/损失、冗余/新增量、顺序、附加门控观测 | 不算发现了9种跨模型规律 |
| 作者代码 | 5个固定Git blob；选定函数/语句执行，MI依赖替换明确记录 | 未跑作者完整训练、SpikingJelly模型、CUDA/AMP或数据 |
| 正式训练 | 0/435；audit_test未访问；真实数据未下载；未用A40 | 原检查点训练仅1次optimizer update |

当前开发模型仍为checkpoint 272ae6230942ebeb，SHA256 f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e。它是验证分析流程和合同的接入对象，不能支撑细胞/骨架优劣结论。

## 4.1.1 先定义系统真正读到了什么

### 属于什么问题、目的

同一个冻结内部系统，更换膜/事件、时间摘要或解码器后性能变化，首先属于观察与读出问题。不能先把它说成神经元记忆变强或变弱。

用实际模型定义：

\[
S_t=(S_t^{cell},S_t^{backbone},S_t^{readout}),\quad
o_t=G(S_t,x_t),\quad a_T=P(o_{1:T}),\quad \hat Y=R(a_T,Q).
\]

只列实际存在的状态。pre可能是计算过程中的瞬时量，post是持久状态，output是Cell真正返回给下游的张量。natural_event还可能与强制emit分支不同。原模型的G由部署接口决定，研究者能导出pre不意味着部署已经允许访问它。

### 文献与我们的适配

Colas-Gattepaille与Muzy，2026-10-14预印本，**Design and Analysis of Temporal Representations in Neural Networks**，arXiv2610.19730，直接讨论编码、激活、学习与解码怎样显式处理时间，是本节重要近邻。本轮核实主记录/摘要；HTML/PDF访问受阻，官方代码未核实，未据题名补写算法。我们不能把“拆开编码—激活—解码”本身当成无人做过的新框架。

我们的具体工作是在现有SNN代码中核对G/P/R，登记训练与部署是否一致、额外状态由谁维护、以及固定同一核心后权限改变对原任务意味着什么；再把这些比较接到后续选择性操作和恢复。

### 实际代码路径

当前LoadModel先用14→64→64编码器并乘事件标志，再调用两层MLP核心。核心BenchmarkModel.forward在每个时刻取backbone.step返回值；若cfg.observation是pre/post，则改取最后Cell.last中的对应量。分类特征若还有空间/传感器轴，再做空间平均；之后才执行时间汇总与head。

因此实际顺序是：**指定载体 → 分类所需空间汇总或重建完整空间map → 时间汇总 → 线性/1×1图像头。** 全时间轨迹若已在前一步丢了空间位置，仍只是池化后的轨迹，不能恢复原先被GAP抹去的位置。

对应代码：parent_l34/.../snnbench/models.py的BenchmarkModel.forward；native_head_audit.py的audit_model/manual_head；access_operators.py的spatial_access/aggregate。

## 4.1.2 五种摘要分别改变了什么

设每步合法观察为o_t∈R^D，T是本次查询前可用步数：

| 摘要 | 公式 | 保留/丢弃的主要条件 | 额外事项 |
|---|---|---|---|
| last | a=o_T | 只保留当前接口；核心自身可能已含历史 | post可别名引用已有状态，不能重复算独立存储 |
| mean | a=T^-1∑o_t | 保留整体统计，一般不保留排列 | 需要累计器及count；不是无历史读出 |
| sum/count | a=∑o_t | 与mean在固定已知T下只差尺度 | 变长时需T才能互相恢复；饱和和精度会改变等价性 |
| exponential | a=∑β^(T-t)o_t / ∑β^(T-t) | 对时间位置加固定相对权重 | 需累计器、分母及β；不自动等于更长记忆 |
| trajectory | a=vec(o_1,...,o_T) | 保留已观察的时间位置 | D升至TD；原固定T头不允许无声padding/外推 |

这里“保留”描述摘要函数，不承诺有限训练读者一定能恢复目标。mean/sum/last/exponential通常没有总的优劣次序。

### 可以证明的关系，而不是凭直觉排序

若O2=f(O1)，且读者族允许任意可测函数，则O1可以模拟O2的任意读者，最优风险满足R*(O1)≤R*(O2)。同载体trajectory能构造上述摘要，因此在该理想前提下不会比其摘要少目标信息。

实际p有限、维数/正则/函数族受限时，新头在trajectory上可能更差。当前开发实验正有连续MSE变差的例子，不违反上述信息关系。

mean与last一般不可比较：两段历史可末项相同、均值不同；也可均值相同、末项不同。末膜与spike历史更没有天然包含关系。只有加入足够的pre、阈值及实际分支规则时，才能推导某个输出；不能由pre/post/spike这些名字直接排列信息大小。

### 固定T下，mean与sum的正则要怎样适配

若sum特征为T·mean，带截距的ridge目标使用平均平方损失加λ‖W‖²，要获得同一预测，应令λ_sum=T²λ_mean，W_sum=W_mean/T，截距适当对应。直接用相同λ可能人为制造差异。

本轮T=7已知控制，λ_mean=.01、λ_sum=.49，预测最大差约8e−15。实际访问扫描用每个观察的校准RMS归一化，使固定T的整体尺度显式受控；不把sum/mean的重复行当成两种新记忆机制。

代码：temporal_weights、aggregate、refinement_relation；run_access_controls.py。既有taxonomy已提出这些合同，本轮把它们落实为可运行控制，不改名宣称新定理。

## 4.1.3 读出自己的历史与NoTD比较

### 为什么必须做

Ma等，2025，**Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects**，arXiv2502.09449，STP区分STBP、SDBP与NoTD，系统比较前向时间计算与反向时间信用。本轮重读STP定义和判据，官方代码入口为https://github.com/liyc5929/neuroseqbench；未重跑其训练。我们沿用这一分离思路，继续明确神经元之外的历史路径。

“cell不跨步”与“整个系统不利用历史”不同。我们的确定控制取x=[1,0]与[0,0]、frame-local notd Cell、相同当前输入及原线性头：最终cell状态相同，但mean输出分别为.5和0。保留快照后清core，仍得到[.5,0]；清readout累计器才得到[0,0]。它直接验证读出拥有历史，而不是靠MI或探针猜测。

### 必须补上顺序条件

这不意味着NoTD+mean能解决任意时序任务。若encoder逐帧共享且无时间/位置cue、无骨架cache，mean/sum对帧排列不变，因此不能区分同一帧集合的不同顺序标签。exponential、位置头或显式时间cue会改变这个条件。

所以正式benchmark要分别审计：是否需要过去的某份内容；是否需要内容顺序；是否需要经过时间；以及哪一个系统部件提供这种能力。我们提供了读出累计与排列控制，未据此宣称原STP全部无效。

### 现有流式实现与内存

训练用BenchmarkModel.forward收集fs并stack完整features。已交付的StreamSession是另一条推理接口：mean/sum维护accumulator，exponential还维护denominator；普通模式不存完整features列表，trajectory才保留它。

每个接口同时记录batch前向的特征栈、stream的cell/cache/readout逻辑行，以及这些操作状态张量中的去重storage字节。去重范围不包含cell.last诊断、参数、autograd、allocator开销或硬件峰值；不能把它称实测设备显存。scalar count/cursor和分母另列。

因此可以对比“理论上可流式”“当前推理代码确实流式”“训练/捕获保留轨迹”三种执行情况。数据类型、别名与额外内部载体都要计入，不能把实际float32 spike默认为1bit硬件存储。

## 4.1.4 线性头的null与未来作用

### 文献对象先固定

Kaufman等，2014，**Cortical activity in the null space: permitting preparation without movement**，Nature Neuroscience，DOI10.1038/nn.3643；Stavisky等，2017，**Motor Cortical Visuomotor Feedback Activity Is Initially Isolated from Downstream Targets in Output-Null Neural State Space Dimensions**，Neuron，DOI10.1016/j.neuron.2017.05.023。它们使“相对哪个下游映射的null”成为明确问题。后者沿3.7选定方法审阅；本轮未复跑动物实验/完整作者代码。

对当前仿射头ŷ=Wa+b，Δa若位于ker(W)，则当前输出变化严格为0。这里W的行必须对应声明的查询：全20个输出、原任务请求的H1/C八坐标、或仅C四坐标，零空间都可能不同。

实际旧头全20输出rank20、null维数12；仅请求H1/C的八输出rank8、null维数24。单位特征扰动造成对应输出变化约1e−16。这是**离线头输入的解析控制**，未把它称概念删除、膜状态干预或未来系统null。

### 为什么当前null不等于未来null

在线性参照S_(t+1)=AS_t、y_t=CS_t中，未来H步的不可见方向应满足所有CA^kδ=0；只检查Cδ=0不够。取C=(1,0)，A把第二坐标移至第一坐标，δ=(0,1)：当前变化0，下一步变化1；可观测矩阵从一步rank1变为两步rank2。

SNN中更要保留有限前向操作，因为阈值和重置会使局部导数与真实变化不同。把某个时刻的线性头null方向写回膜状态，后续事件可能改变；这由L4.3—L4.4验证，不能用当前SVD代替。

**研究用途：** 既能提出“当前内容可读但原头未使用”的候选，又能防止把瞬时沉默误写成整条未来路径无作用。真正有用的诊断需预测后续目标作用和可恢复条件。

## 4.1.5 logits、概率和训练损失分别登记

同一共享线性头、归一化固定权重α时，∑α_t(Wo_t+b)=W∑α_to_t+b。sum权重和为T时，bias项是T·b，不能不加修正地套同一个等式。

softmax非线性使“平均概率”与“平均logits后softmax”不等；逐步平均CE也不是汇总logits后的CE。实际控制的三个时刻logits为(20,0)、(0,2)、(0,2)：

| 操作 | 结果 |
|---|---|
| 平均logits后softmax | class0概率约.99520，预测class0 |
| softmax后平均概率 | class0概率约.41280，预测class1 |
| 目标class0的逐步平均CE | 约1.41795 |
| 目标class0的平均logits后CE | 约.004816 |

因此L4记录三个不同对象：P_train参与训练损失，P_deploy产生部署预测，运行时状态支持后续访问。训练时利用某步信息可以改变学到的参数，却不说明部署预测访问了该步；学习来源由L6分析，部署访问由本节核对。

## 4.1.6 2025 MI-TRQR怎样真正用于本研究

Xue等，NeurIPS2025，**MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks**。原文：https://papers.nips.cc/paper_files/paper/2025/file/dfb89b72be941eeeaeb5ee53bd3e7fc1-Paper-Conference.pdf 。官方代码https://github.com/dfxue/MI-TRQR，固定commit bc47d8a8240d9c3bad74fb5968281821f8031641。

本轮选读§3.1—3.3、实验入口及相关对照段；5个源码文件Git blob匹配，执行选定MI/mask、TET_loss与重标定赋值。不是只看题名判断：原文已有局部/全局冗余、训练掩蔽和最终时间权重重标定，且已有随机mask等对照。不能把这些已做内容当成我们补出的新想法。

### 原文/代码复用与具体适配

| 已有内容/本次读到的代码 | 我们怎样利用 | 对应改进/新检验 |
|---|---|---|
| 高层特征的时间掩蔽及权重重标定 | 把汇总P作为可改动部件 | 固定核心、固定读出、联合训练三种比较分开 |
| CIFAR MI函数把batch×channels作为局部采样轴；全局再展平空间 | 精确登记MI随机变量与样本轴 | 不把它说成单场景高维完整模式的信息量 |
| mask模块eval直接返回输入，掩蔽只在训练运行 | 区分训练规整与部署操作 | 不能用训练mask存在推断推理正在删除历史 |
| CIFAR选定train/test使用outputs.mean(1)，TET可改变训练loss | 区分P_train与P_deploy | 逐步CE、平均logits、平均概率各自核对 |
| ImageNet选定AMP训练分支按batch汇总firing数量加权；选定eval条件分支取output[0] | 按任务/配置/分支逐项建合同 | 不能概括成所有数据集训练和推理都使用同一时间汇总 |

代码分支观察不等于判定论文实验错误。我们没有跑完整作者模型、criterion、AMP或数据流；已有权重/分支的实际复现需另做配置级验证。这里的收获是找到了必须进入L4协议的分析轴。

原MI函数在恒定输入的零熵位置会给出非有限归一化值；本轮人工输入中有12个这样的元素。我们记录数量而不把NaN写入结果JSON，也不据此宣称作者真实数据遇到了该情形。执行时MI依赖替换为明确的有限列联表MI，原TorchMetrics运行环境未执行；其余选定定义/语句保留。TET_loss λ=0与逐步CE参照逐值一致；加权赋值的人工结果为(.25,.75)。

### 参数为零，不等于观察权限为零

若时间权重α_t还读取其他层的firing、整个batch或辅助状态，则观察应包含这些变量，不只是最后层o_(1:T)。有一个确定反例：最后层轨迹对两个目标恒为(1,−1)，仅看轨迹无法区分；若α读取另一个携带目标的变量，就能输出不同结果。信息来自附加变量，没有违反数据处理。这是我们的观察合同控制，不是说MI-TRQR读取了标签。

同样，I(o_early;o_late)高不说明后者没有新目标内容。精确四状态控制中，两步共享1bit nuisance，同时后来观察新增1bit目标信息；两个量可共存。若要把冗余诊断转为我们的设计，要在合法F/G查询、独立选择、同资源条件下确认删掉的是任务不需部分、保护的是仍有效部分，而非只优化跨步相关性。

这些适配使已有门控/重标定工作能够进入后续内容选择性与风险恢复分析。争取形成的增量是可迁移诊断与设计收益，尚不是当前旧模型已经证实的新架构。

## 4.1.7 实际访问网格与本轮结果

### 怎样拟合、冻结和评价

载体固定为末层实际output/pre/post，摘要为上述五种，合计15视图。每头使用128或512个独立训练场景，嵌套索引共享于全部视图；λ∈{1e−4,1e−3,1e−2}由独立2048场景选择。连续目标为原任务H1/C各4坐标；二分类标签由相同坐标的固定16方向/来源产生，不是32份新的独立测量。

512个无标签校准场景只估每视图RMS。90个联合矩阵解批量拟合连续8输出及二分类32输出；连续头、辅助分类探针和完整批量拟合的参数成本在freeze中分别列出。分类与MSE各自选择λ，不让分类改进代替重建收益。

全部头、候选及选择在保存freeze_contract.json后固定，随后首次生成8192个评价场景。全部15×2预算×2来源×2目标=120风险单元与120个场景损失数组保存。12个预定分类配对比较使用同时有界区间；MSE不使用这套分类证书。本族条件于旧表示和固定选择，不与3.6/3.7各自置信族合称整个研究95%。

### 512标签下的部分完整对照

| 合法观察 | H1分类错误 | C分类错误 | H1连续MSE | C连续MSE |
|---|---:|---:|---:|---:|
| 末时刻实际output | .49248 | .48444 | 1.01879 | 1.00241 |
| output的mean | .46841 | .45829 | 1.00320 | .98758 |
| output完整trajectory | .44325 | .45772 | 1.07619 | 1.12126 |
| 末时刻内部post | .48859 | .41772 | 1.02252 | .91064 |
| 末时刻内部pre | .48862 | .42010 | 1.02182 | .90588 |

训练均值/多数类基线在512标签下的H1/C MSE约1.00734/1.00450，分类约.50108/.50111，完整8个基线风险另存。这进一步显示当前旧模型及部分高维新头仍很弱，不用“超过原头”掩盖未超过常量参照的情况。

12比较中2个同时区间下界为正：H1的output trajectory相对output last减少约4.92个百分点，区间[1.04,8.80]个百分点；C的post last相对output last减少约6.67个百分点，区间[2.79,10.56]个百分点。都是对一个旧表示和固定任务方向的条件结果。

第一项增加历史访问和参数；第二项增加内部模拟观察。两者都不能叫作神经元本体记忆延长。trajectory的连续MSE反而更差，说明在512维、有限标签和所测正则下，理想权限包含关系不会自动变成更好的有限样本学习。不能只保留有利的分类结果。

原生post/last头的H1/C MSE约1.30049/1.57767，同权限新线性头可更好，但原模型仅训练1步、新头数据/求解不同，因此尚不能排除原头未训练充分。原头重放的最大数值差约4.3e−7，来自NumPy/PyTorch浮点路径；没有虚报逐位完全一致。

### 成本具体是多少

当前D=32、T=16。末时刻连续8输出线性头为32×8+8=264参数；全轨迹连续头为512×8+8=4104参数。联合40目标的批量求解和辅助分类头另计，不把它们全算成原生8输出任务的部署头。

mean/sum/exp在已有stream实现可用D维累计器，而原StreamSession的trajectory实现保留TD个观察值；这只是该实现，不是线性轨迹读者的必要存储下界。实际额外字节还取决于dtype、别名、是否保留诊断、当前特征与cell state是否共享storage；本包同时给registry而不只给大O。

### 线性trajectory头的等价流式改进

若固定T、G和线性头在序列开始前已经确定，把轨迹权重分为W_1,...,W_T，则：

\[
\hat Y=\sum_{t=1}^{T}W_t o_t+b.
\]

推理可以维护K维输出累计器，逐步加W_t o_t，而不保留TD维轨迹。folded_trajectory.py已实现该等价路径，并在相同288个trajectory配置上重新检查；不是288个新训练模型。参数仍是TDK+K，分时权重访问与时钟仍需计费。当前特征及core/cache也仍存在，不能把K维输出累计器写成整系统仅K维。

**等价的是已确定R的输出，不是任意未来查询都可访问的完整轨迹。** 若query直到序列结束才到达并决定一个此前未知的W_t(Q)，不能无条件提前折叠；需保留足够的查询无关摘要/基，或另定合法查询集合。任意非线性轨迹头也不由本等式保证可折叠。模型是否真正存完整轨迹、是否仅计算某个固定任务的充分输出，正是L4应分开的两种系统。

旧模型四场景post/last的注册操作张量去重为1024字节，post/mean为1536字节，均不计诊断/参数/allocator；per-scene分别256/384字节。它显示实际累计器额外状态，但不是设备峰值内存或功耗。不能用一种未经优化的实现直接断言某种观察权限必然昂贵。

## 4.1.8 骨架、任务变体与验收

| 变体 | G/P/R中具体改哪里 | 关键适配 |
|---|---|---|
| mlp/recurrent_mlp | 末Cell观察与相同汇总；递归cache另属核心 | 同一读出不代表相同总状态预算 |
| temporal_conv | 时延tap/cache先进入核心，再产生被读特征 | 不把cache的历史归给Cell |
| cnn1d | 分类时传感器轴/空间轴可被平均 | 完整轨迹不能补回已丢轴位置 |
| cnn2d/SEWResNet | 分类的G含空间平均；skip影响上游核心 | 输出null只相对当前下游对象 |
| spike_attention | 空间交互属于核心，最后G/P仍明确 | 不将内部attention写成固定W；本轮只核对原生头 |
| U-Net重建 | G保留C×H×W；R为像素1×1头 | 不用分类GAP；本版图像trajectory头明确不支持 |
| ALIF/ADLIF/two_state | pre/post受a/syn影响，output仍取实际返回值 | 完整核心状态与读出观察分开，不遗漏aux |
| analog_hard_ref | 自然阈值事件仍存在，但实际output为pre | 不把事件日志当实际部署输入 |

全网格：12细胞×8骨架×3观察×5摘要=1440个分类案例；12细胞×2空间重建骨架×3观察×4摘要=288个图像重建案例。所有案例未训练；网格固定width2、T3、normalization=none和默认full模式，不能冒充所有超参数都已覆盖。载体→原生feature和手工G/P/R预测的最大绝对差约7.63e−6，stream/batch最大差约4.77e−7，均在明确2e−5数值验收界内。原参数哈希保持不变。

测试覆盖固定T、变长不可逆、mean/sum ridge尺度、偏置、概率/CE差别、空间池化损失、readout状态快照、NoTD累计、原头/未来null及门控附加观察。31项测试无skip；三阶段一键流水线已实际通过，增加常量基线后单独重跑访问阶段并确认原120风险与12比较完全不变；冻结读者前128个场景另行重放验证。测试和网格不是训练后架构证据。

## 5. 七条原计划文献的用途与当前阅读状态

| 原ID/完整文献 | 怎样进入L4.1 | 代码与阅读状态 |
|---|---|---|
| REP-TEMPORALDESIGN，Design and Analysis of Temporal Representations in Neural Networks，2026预印本 | 编码/激活/学习/解码显式时间的近邻 | arXiv2610.19730主记录/摘要；全文请求受阻，代码未核实 |
| BEN-STP，Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects，2025 | 前向/反向/NoTD分离，继续登记读出及cache | 重读STP原文；官方neuroseqbench入口核实，未重跑作者模型 |
| BEN-SOUL，Benchmarking spiking neural networks across sensing modalities on edge devices / PDF题名A modality-aware benchmark reveals when spiking neural networks benefit edge sensing，2026预印本 | 模态、骨架、原生分类/连续目标与系统成本背景 | 原文题名/范围及代码声明重新核验；https://github.com/yudi-mars/Soul-NeuSim；本轮未逐个模型审计其部署汇总 |
| READ-NULL，Cortical activity in the null space: permitting preparation without movement，2014 | 明确null相对的下游/查询 | 重新核实身份及摘要；完整作者分析代码未核实，本轮自己的线性参照 |
| BEN-RETHINK，Rethinking the performance comparison between SNNS and ANNS，2020 | 比较条件与工作负载背景 | 重新核实正式2020身份；不是2025/2026新文献，未新执行作者代码 |
| REP-TRQR，MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks，NeurIPS2025 | 汇总/训练loss/掩蔽/自适应权重的实际区别 | 选定原文方法；5个固定源码与选定执行，详见manifest/audit |
| READ-FEEDBACK，Motor Cortical Visuomotor Feedback Activity Is Initially Isolated from Downstream Targets in Output-Null Neural State Space Dimensions，2017 | 瞬时output-null与后续作用的区分 | 沿3.7相关方法审阅；本轮不虚增动物/作者完整复现 |

L4.3后续另对照2025 Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE、2025 How Reliable are Causal Probing Interventions?以及2023 LEACE。前两篇身份/选定内容已核，代码深审在相应小节继续；不在此处假装完成全部干预实验。

## 6. 与两篇核心benchmark相近和推进之处

相近处：都研究SNN时序行为与可比模型/任务，保留输入、神经元、骨架、训练和部署条件。STP已有机制/梯度分离，Soul已有多模态与系统分析，不把它们概括成“只报准确率”。

本节具体增加的比较单位是**同一个冻结核心和同一内容，分别改变G、P、R及它们的运行状态/训练权限**。输出是访问偏序、实际部署头、原头与受控重训差别、训练/推理汇总、外部读出记忆、以及带成本的风险变化。它为L4.2—4.4的内容作用/恢复提出可以被否定和改进的候选解释。

这些已实现的合同/控制是研究基础；是否形成突出的论文结果，取决于充分训练模型上是否发现稳定条件规律、是否预测后续干预，以及是否能在原生分类和重建中指导收益。这个目标不因有工具近邻而缩小，也不由本轮旧模型数值提前宣布完成。

## 7. 代码与逐项复现

| 文件 | 功能 |
|---|---|
| access_operators.py | G/P/R中的空间/时间算子、权限偏序、头null与可观测参照 |
| native_head_audit.py | 实际头、手工分解、batch/stream、操作状态去重计量 |
| run_access_controls.py | 9类已知答案与反例 |
| run_native_access_audit.py | 原模型实际配置及11个同W访问反事实 |
| run_interface_grid.py | 1728个注册骨架/细胞/头数值案例 |
| folded_trajectory.py、run_folded_grid.py | 已知线性轨迹头的等价有界累计；288个配置验证 |
| run_access_review.py | 冻结核心、15视图、90联合拟合、独立选择与评价 |
| audit_author_access.py | 固定作者源码、选定函数/语句及MI依赖适配 |
| l41_suite.py | 控制、访问分析、骨架接口三阶段流水线 |
| verify_l41_artifacts.py | 只读一致性、冻结头哈希和场景重放 |
| plot_l41.py | 实际证据科学图 |
| tests/test_access.py | 数学、状态、接口和合同验收 |

~~~bash
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -v
python l41_suite.py --output outputs/l41
python run_native_access_audit.py --output outputs/native_audit.json
python run_folded_grid.py --output outputs/folded_grid.json
python verify_l41_artifacts.py --output outputs/verification.json
python plot_l41.py
~~~

输出目录须不存在。一键pipeline不训练SNN、不下载真实数据。run_access_review的原任务适配器限all-present physical_k、H1/C regression；其他原生任务需要对应目标/观察适配，不能把辅助分类探针当作已适配的真实分类任务。generic原生接口网格支持的分类/图像重建组合另有明确表和验收。

## 8. 接下来L4.2要收束的对象

带着当前已经固定的观察和原任务，接着问：同权限重训的改善能否在原头充分训练后保留？非线性读者是否增加同预算利用？哪一项候选已足够值得进入原头内容操作？使用独立验证probe与合法标签，逐项排除优化、额外权限和容量的竞争解释。

后续再进入内容命中/非目标保护、损伤—恢复和区域重建。L4.1不把这些提前全部打成一个总分，也不据当前头null或新探针优胜直接宣布原模型已经/没有使用某份内容。
