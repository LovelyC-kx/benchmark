# L4框架与L4.1：观察权限、实际读出和状态归属

2026-10-27，v0.1。本包完成L4.1深入分析、参考代码和开发验收；L4.2—L4.5保留详细规格，未计作实验完成。

先读SNN_L4_Framework_and_Research_Plan_2026-10-27.md，再读SNN_L4_1_Deep_Review_and_Execution_2026-10-27.md。沿原五部分推进，不重定义六层。

## 已执行

- 31项测试无skip。
- 1440分类与288图像重建G/P/R、batch/stream接口案例，均未训练；另在相同288个trajectory配置上核对已知线性头的折叠推理，未新增训练模型。
- 原旧检查点12个同W访问配置核查；只有post/last为实际训练配置。
- 冻结核心15观察、p128/512、3个ridge候选，90次联合矩阵求解；120主网格风险单元与8常量基线，8192评价场景。
- 12个预定同时分类比较中2个下界为正；这些是增加历史/内部权限的访问比较。MSE负结果与常量基线完整保留。
- 5个固定作者代码blob；选定函数/语句实际核对，MI依赖替换及未执行原模型范围明确登记。
- 三阶段流水线通过；新增基线后重跑访问阶段，原120风险与12比较完全相同；128个保存场景的分类及MSE损失重放最大差0。

主要模型仍是原训练1次optimizer update的checkpoint。新SNN训练0，formal=0/435，audit_test未访问，真实数据未下载，未用用户A40。

## 复现

需要Python及requirements-runtime.txt依赖。常规pip安装Torch会安装其测试依赖；如使用人为--no-deps安装，应补足Torch声明的依赖。

~~~bash
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -v
python l41_suite.py --output outputs/l41
python run_native_access_audit.py --output outputs/native_audit.json
python run_folded_grid.py --output outputs/folded_grid.json
python verify_l41_artifacts.py --output outputs/verification.json
python plot_l41.py
~~~

输出路径须不存在。一键入口运行controls、access、interfaces并记录各阶段退出码；未训练SNN。默认验证/绘图读取包内证据。自定义兼容检查点可加--run-directory，旧LoadModel加载器限原项目两层MLP。当前原任务评分只适配all-present physical_k H1/C回归；辅助分类不等于已适配真实原生分类任务。

可选作者源码核查：

~~~bash
python retrieve_pinned_author_sources.py --output outputs/pinned_sources
python audit_author_access.py --source-directory outputs/pinned_sources --output outputs/author_checks.json
~~~

下载器仅获取manifest中的固定代码文件并核验Git blob，不下载数据。选定作者源文件不随本包重新分发；出处与阅读/执行范围保留，不能冒充完整作者复现。

## 代码入口

- access_operators.py：G/P/R、空间/时间汇总、权限关系、头null/可观测参照。
- native_head_audit.py：实际head分解与stream操作张量状态账。
- run_access_controls.py：9类已知答案控制。
- run_access_review.py：冻结、同权限重训和额外访问的风险比较。
- run_native_access_audit.py、run_interface_grid.py：原生配置与注册骨架接口。
- folded_trajectory.py、run_folded_grid.py：固定且已知线性trajectory头的等价有界累计；不是任意晚到查询的完整轨迹缓存。
- audit_author_access.py：选定作者函数与语句的可核查执行。
- verify_l41_artifacts.py：只读一致性及保存场景重放。
- references/：计划覆盖、固定源码、项目依赖来源。

parent_l34/为41个原项目继承模块；桥接/数组保存等工具沿用3.5/3.6。新L4实现和继承代码分别记录。内存数据为指定操作张量范围，排除诊断/参数/allocator与峰值能耗；别名去重与逻辑大小分别报告。
