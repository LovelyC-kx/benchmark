# L3.3.1：分析对象、表征分解与条件变异

先读SNN_L3_3_1_Decomposition_v0_1_2026-10-25.md。本包承接L3.2，完成focal/共同目标/未请求内容分组、联合线性编码、未解释部分与二阶诊断、三种条件重复面板。没有新训练SNN，没有执行最终audit。

## 从已有验收样例开始

在本目录运行；命令适用于PowerShell，不使用Linux反斜杠换行：

```powershell
python -m unittest discover -s tests -v
python run_decomposition.py evidence/source_smoke_run --output runs/my_check_c --smoke
python run_decomposition.py evidence/source_smoke_run --output runs/my_check_h1 --focal H1 --smoke
```

已附3.2原有的一个lif_hard极小checkpoint及必要特征，原训练只有1次优化更新，供复查流程；两个focal视图是同一模型，不是两个模型实验。

分析自己的完整run时，将run目录替换为3.2 runner的实际输出，省略--smoke；需要run.json、best.pt、geometry_fit.npz和development_test.npz。默认观察last_post；--observation还支持joint_post、last_pre、last_output。原头和模型权重不变。

初版只接入连续坐标回归、全部来源存在的主条件。分类、图像重建与部分来源缺失需先补目标/来源合同；程序对当前未覆盖的分类或presence条件明确停止。

选择新的--output目录；程序不覆盖已有派生分析。device默认cpu。当前只验证CPU；GPU执行如与原导出不符会停止，不能把不一致的特征与新面板混合。

## 依赖

纯数学和数据检查只需NumPy。读取checkpoint需要与原run兼容的PyTorch。复用现有环境；不要为运行本包覆盖已工作的Windows GPU环境。

## 文件

| 文件 | 作用 |
|---|---|
| roles.py | 任务相对的逐坐标角色与全来源存在性合同 |
| decomposition.py | 联合编码、评价成分、协方差交叉项、二阶预测诊断 |
| repeat_panels.py | 完全同输入/只变测量/只固定focal的面板与条件协方差 |
| run_decomposition.py | checkpoint身份、潜变量行身份和再生成特征核查；分析输出 |
| inherited.py、parent_l32/ | 复用并保留原样的本项目父代码；哈希在parent_provenance.json |
| tests/ | 14项已知答案与合法性检查 |
| references/ | 原计划覆盖、文献与作者代码读取范围、父checkpoint身份 |
| evidence/ | 实际检查与两个视图的极小流程验收；没有论文模型比较结果 |

## 输出怎么读

report.json区分编码R²和目标恢复R²；本程序的编码R²用于“内容解释表征”。协方差和各角色成分保存在decomposition_arrays.npz。联合线性未解释项不叫物理噪声；只减去线性focal项的余项也可能包含focal非线性函数。

三种重复面板分别登记条件。其他共同目标在focal-only分析中属于非focal内容，但不自动是有害干扰。不同接口与观察维数分别报告；本程序不输出Wakhloo四量、不输出干扰因果中介份额。

正式阶段的14项工具检查与已知例子，不替代充分训练、多种子和留出任务实验。
