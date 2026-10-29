# L4.3：内容、载体与通路干预

2026-10-27，v0.1。研究规格、理论、文献/作者源码对照与CPU参考执行包。保留原六层及L4五部分；不替代正式训练和真实数据实验。

优先阅读：

1. `SNN_L4_3_Deep_Review_and_Execution_2026-10-27.md`：按4.3.1—4.3.8讲对象、目的、代码、文献/增量、实际结果和后续输入。
2. `SNN_L4_3_Mathematical_Specification_2026-10-27.md`：完整假设、推导、反例和代码对照。
3. `SNN_L4_3_Literature_and_Code_Map_2026-10-27.md`：Master原6条文献、固定作者文件、复用/适配/独创归属。
4. `module_status.json`：机器可读的本轮状态；`evidence/`是本轮新执行记录。

## 本轮执行范围

原冻结模型是一次优化更新的hard-LIF、两层MLP、width32、post观察、last读出，原生任务是连续H1/C回归。新SNN训练0，正式0/435，audit_test未访问；数据与预训练模型未下载，未用A40。符号分类是原回归输出的辅助评分，不是已训练分类模型。

- 原状态实验238条件、真实二值边24条件，共用2048个独立场景对。
- 看到无正则算子放大后，单独追加112条件的稳定性诊断，使用新1024场景对；不称预注册确认。
- 1496条逐场景损失数组；40项测试；10类已知答案控制；240个未训练no-op接口。
- 两个冻结拟合bank；原238状态与24边条件各前32场景独立重放，保持batch128，输出/状态逐值一致。
- 186次ridge候选矩阵求解、14次ΔF映射求解、68次离线MLP拟合/27200更新；均不更新原SNN。

图`figures/L4_3_intervention_evidence.png`：A是2048拟合样本的算子范数；B/C是新1024开发配对的读取/输出差异；D是原开发池64个场景的最后一步阶段控制。图中无误差条的MSE为经验点估计。整个图只作方法与开发证据。

## 代码组织

| 文件 | 职责 |
|---|---|
| bridge43.py | 调用未改动的旧模型/生成器；自然F配对；23维独立验证标签 |
| intervention_math43.py | LEACE/orth/MP、随机/范数匹配、内容转移、误差分解、事件包络 |
| regularized_intervention43.py | 固定协方差度量及范数界 |
| runtime_interventions43.py | 真实v/a/syn/联合状态，阶段控制、完整供体快照 |
| fit_intervention_bank.py | 冻结方向、独立自然probe和操作后probe |
| run_state_interventions.py | 238原生状态条件及原任务风险 |
| run_edge_interventions.py | 实际二值边及计数/翻转匹配 |
| run_stability_followup.py | 作者默认与固定收缩的适应性追加诊断 |
| run_intervention_controls.py | 10类已知答案控制 |
| run_intervention_interfaces.py | 240个未训练分类/图像重建接口 |
| audit_author_sources.py | 固定作者原类/AST定义的受限执行审计 |
| fetch_pinned_sources_l43.py | 下载并逐文件核对固定公开源码 |
| verify_l43_artifacts.py | 继承hash、bank、场景损失、比较族及全部原条件前缀重放 |
| plot_l43.py | 用实际记录生成科学图 |

`parent_l42/`为未改动的依赖代码及完整parent_l41，另含两份L3.7实际边模块。104个继承文件有hash。它不是L4.2全部主实验的重打包；部分旧入口缺少其原L4.2证据/refs，不能据此重复运行整个旧包。旧L4.1报告只作历史证据，不计本轮新结果。

## 运行

在包根目录配置Python和依赖后：

```bash
python -m unittest discover -s tests -v
python verify_l43_artifacts.py --output /absolute/path/new_verification.json
```

所有生成入口使用新目录，避免覆盖已有科学记录。若从头重跑本轮：

```bash
python fit_intervention_bank.py --output /absolute/path/replay/fitted
python run_state_interventions.py --fit-directory /absolute/path/replay/fitted \
  --output /absolute/path/replay/state
python run_edge_interventions.py --output /absolute/path/replay/edge
python run_intervention_controls.py --output /absolute/path/replay/controls.json
python run_intervention_interfaces.py --output /absolute/path/replay/interfaces.json
```

边入口使用包内固定实验合同及相同样本，不读取新拟合目录的参数。精确前缀重放对运行库、CPU和batch保持要求比统计复现严格；本轮已在记录的环境中验证。

作者默认分支的重新拟合需要固定作者源码；已有追加bank的结果分析和原生重新评估不需要下载作者模型：

```bash
python fetch_pinned_sources_l43.py --directory /absolute/path/author_sources
python audit_author_sources.py --source-root /absolute/path/author_sources \
  --output /absolute/path/replay/author_audit.json
python run_stability_followup.py fit --source-root /absolute/path/author_sources \
  --output /absolute/path/replay/stability_fitted
python run_stability_followup.py evaluate \
  --fit-directory /absolute/path/replay/stability_fitted \
  --output /absolute/path/replay/stability
```

`evidence/metadata_errata.json`修正继承probe选择目标的旧文字；实际拟合始终用全部23维标签。`numeric_recheck_receipt.json`解释实际β、FP32余量及no-op张量布局修复，952条初始场景损失未改变。追加稳定性方法单列，不通过删去失败分支营造结果。

## 尚需正式研究

充分训练、多seed、相关因素、非线性操作族、aux/cache/读出内容语义、反馈/旁路真实边，以及原生分类/图像或语音重建。240接口只验收no-op，不能算这些模型已完成内容归因。L4.4将接收冻结内容、阶段、参考运行和损失，做对应的损伤—恢复四组实验。
