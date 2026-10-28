# SNN第三层3.2 v0.3：文献/代码复盘与内容操作分析

先读`SNN_L3_2_Review_v0_3_2026-10-25.md`；实验合同见`PROTOCOL.md`。本包继承原六层计划与v0.2，历史协议留在history；正式拟合仍为0。

## 研究目的

在内容复杂度、必须恢复的历史份数、完整状态和观察权限明确后，测量SNN各份内容的恢复怎样变化，区分发放输出、反馈复位、年龄、后续写入、读出和学习因素。等总目标量控制用于检验单一“总维数”是否足以描述需求；它不先验保证出现新容量定律。

## 文件入口

| 文件 | 做什么 |
|---|---|
| SNN_L3_2_Review_v0_3_2026-10-25.md | 九篇文献与作者代码的复用、修改、缺口和创新复盘 |
| PROTOCOL.md | 任务变量、模型、实验矩阵和本轮执行补充 |
| LITERATURE_MAP.md | 9篇具体文献的使用范围、读取范围、代码核查和不得冒称之处 |
| load_data.py | 内容/需求/写入/年龄生成、权限、单来源配对、Gaussian观察oracle |
| linear_reference.py | 线性维数与噪声参照、年龄展开、独立拟合与含交叉项风险恒等式 |
| experiment_plan.py/json | 15个screen、135主配置、最多435个分阶段回归配置；不自动启动训练 |
| torch_adapter.py | 共同编码器、L2模型注入、原头/内容特征、冻结操作 |
| run_train.py | 单配置训练/验证选模、特征导出、可选冻结编码器和分类 |
| analyze_features.py | 独立ridge probe、每来源恢复、原头对比和线性风险记账 |
| intervention_features.py | 五种操作下的匹配特征与原头预测 |
| analyze_interventions.py | 固定自然探针、单来源配对、原头变化、即时编码诊断 |
| parent_l2/ | 4份原样复制的L2必要代码；来源哈希在parent_provenance.json |
| tests/、run_reference.py | 数据、解析、实际Torch功能验收与校准图 |
| evidence/ | 本轮实际检查、回归/分类/冻结编码器三个极小训练流程smoke及其身份，不能当正式基准结果 |

## Windows / PowerShell快速入口

在本文件目录执行；不使用Linux反斜杠换行：

```powershell
python run_reference.py
python experiment_plan.py
python -c "import json; p=json.load(open('experiment_plan.json')); print([(r['run_id'],r['config']['cell'],r['config']['data']['required_history']) for r in p['runs'] if 'screen15' in r['stages']])"
python run_train.py --run-id <上一步的ID> --device cpu --output runs --smoke
python analyze_features.py runs/<实际输出目录> --probe-n 64
python analyze_interventions.py runs/<实际输出目录> --probe-n 64 --sample-n 32
```

正式开发screen去掉`--smoke`，设备可设为已核验环境中的`cuda:0`。不在本脚本默认同时占用两张GPU；每个进程一个明确run_id/输出目录。可使用现有兼容PyTorch/NumPy环境；不要无差别覆盖用户服务器环境。

独立audit只在协议与选择规则固定后用`--export-audit`导出；该访问被记录。smoke明确拒绝访问audit。正式probe默认6144例；smoke须显式设`--probe-n 64`，其结果仅验收程序连通。

冻结共同编码器的条件控制用`--frozen-encoder <本脚本生成的best.pt>`，输出ID包含该文件哈希，不覆盖原run；记录总参数与训练参数。分类确认加`--objective classification`。它们不自动计入主回归矩阵。

## 运行范围与解读

当前CPU环境可以运行PyTorch；此前L3.1交付的“仅语法检查”是当时记录，不自动算成本包验收。以本包evidence的当次输出为准。分析图明确标注解析/数据校准，不含伪造的SNN能力曲线；极小smoke的MSE/准确率不进入论文模型比较。

需要正式训练、更多种子、编码器/坐标对称控制及后续层验证，才能将本节的协议设计升级为机制发现或设计规则。

## 本轮新增分析验收

复制的三个极小训练checkpoint均为lif_hard，分别用于回归、分类、冻结编码器流程验收；五种单元的前后向/回放另由自动检查覆盖，不能把三个极小训练说成五模型正式结果。

```powershell
python analyze_interventions.py evidence/smoke_runs/272ae6230942ebeb --probe-n 64 --sample-n 32
```

正式开发分析使用实际run目录并省略两个样本量参数。脚本只访问development_test；每个探针在自然probe_train拟合、自然probe_val选择，之后固定评价操作。输出包括来源恢复、配对误差、原头、单来源输入响应及出现当刻的编码器诊断。bootstrap区间仅描述同checkpoint的序列波动，不能替代模型训练种子。新输出重跑会替换该run的派生分析文件，不修改checkpoint。

参见evidence/verification.json与references/code_audit.json。28项检查通过；本轮没有新增正式训练或作者网络训练。MAD只运行两个生成器行为检查，其语义记录不用于否定论文已发表数值。
