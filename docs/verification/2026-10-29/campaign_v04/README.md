# v0.4四阶段管线工程验收

日期：2026-10-29。这里保存实际执行的CPU工程验收、全量设计计数和原六层索引。研究执行方案见[四阶段方案](../../../FOUR_PHASE_EXECUTION_PLAN.md)。

## 实际执行结果

```text
PYTHONPATH=/workspace/scratch/0086de0e9beb/runtime_deps:. \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m unittest discover -s tests -v

Ran 60 tests in 124.611s
OK
```

完整输出见[unit_tests.log](unit_tests.log)。包含实际小规模优化更新、检查点续跑、独立分析拟合和后处理子进程；临时训练数据在各测试的临时目录中，日志保留其实际执行过程。

新增验收覆盖：

- 原精度字节恢复、二值压缩、signed zero、bfloat16、标量、空数组和非有限标量。
- 超过旧128坐标预算的完整采集，以及MLP、Transformer KV前缀、Mamba-style状态、ConvLIF空间状态的后缀预测重放。
- 失败采集的部分档案、失败训练的实际batch/优化器/权重/RNG现场。
- 逐更新场景损失与梯度统计、已提交检查点对应的日志前缀、未提交尾部保留。
- 六层来源索引和实际8项smoke配置入队；独立后处理子进程实际执行分析、同一run的作业串行领取。
- 当前全部14项公共模块、拟合器文件保存及原生读出权重/标准化/投影的复算。

既有模型、真实格式fixture、预测目标、SG/信用、空间/频率、角色、全状态和精确续跑检查同时通过。

## 全量设计计算

[dry_campaign.json](dry_campaign.json)来自实际执行：

```bash
python -m snnbench campaign-plan \
  --root /workspace/scratch/0086de0e9beb/campaign_v04_design
```

默认profiles为full/paired/learning，扩展模型名单，五种子。实际计算得到315,270、125,210、84,290份合成配置，共524,770；尚未跨profile去重，未加入真实canonical目录。本次没有物化这个大清单或启动其训练。

[plan_inventory.json](plan_inventory.json)保留40个Master主模块、29个编号细分节、33份来源的SHA与代码候选。907标题的完整上下文索引由campaign-plan重新生成，此处保存计数而不重复所有标题。索引记录逐项待执行的专用分析，没有把14个公共模块的运行等同于原计划全部完成。

[local_hardware.json](local_hardware.json)记录本次实际CPU环境；GPU列表为空。用户服务器按声明84GB规划，实际设备和吞吐在服务器运行同一入口时检测。

本次证据证明工程接口、数值回放和数据保留行为；全量研究结果将在服务器队列运行后生成。各项来源和执行文件的哈希见[summary.json](summary.json)。
