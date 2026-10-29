# L4.2：数学理论、潜空间与受控读者

2026-10-27，v0.1。先读三份正文：

1. SNN_L4_Mathematical_and_Latent_Framework_2026-10-27.md：覆盖L4.1—4.5的理论主线与条件。
2. SNN_L4_2_Deep_Review_and_Execution_2026-10-27.md：4.2八个内部部分、实际代码、结果与后续输入。
3. SNN_L4_2_Literature_and_Code_Map_2026-10-27.md：原七条计划＋七篇新增；oral身份和阅读/执行深度逐项登记。

## 实际完成

- 12类解析/穷举控制，25项测试无skip。
- 同一个原训练仅一次更新的冻结SNN，6观察、两拟合预算、两共享draw、4种主/控制读者，共96个被选读者，384相关风险单元。
- 拟合/选择监督预算分别128/32及512/128，总160/640；512无标签校准另列。独立验证probe另512拟合＋128选择，未参与主选择。
- 150次ridge联合求解、96个MLP优化拟合、38400次离线头更新；核心参数不变，新SNN训练0。
- 原头输出合同下row/null及各5个同秩随机方向，24投影、48新读者、144矩阵求解、192相关风险单元，另8常量基线。
- 主分析8192、子空间4096各自独立评价池；不跨池构造成对比较。
- 8份固定作者代码blob核对；选定原函数、人工依赖替换及未执行范围逐项声明。

这些不等于正式多架构发现；formal=0/435，audit_test未访问，真实数据未下载，原作者预训练模型未下载，未用A40。parent_l41/是不变的继承依赖与旧证据，旧1728接口案例不重复计算。

## 复现

~~~bash
python -m pip install -r requirements-runtime.txt
mkdir -p outputs
python -m unittest discover -s tests -v
python run_latent_controls.py --output outputs/controls.json
python run_reader_ladder.py --output outputs/reader_ladder
python run_task_subspaces.py --output outputs/task_subspaces
python verify_l42_artifacts.py --output outputs/verification.json
python plot_l42.py
~~~

输出路径须不存在；verify/plot默认读取包内已保存证据。作者源码可选获取：

~~~bash
python retrieve_pinned_author_sources.py --output outputs/pinned_sources
python audit_selected_authors.py --source-directory outputs/pinned_sources --output outputs/author_checks.json
~~~

下载器只取8份固定代码并核Git blob，不下载任何数据或模型。第三方完整源文件不随包分发。

## 代码组织

- latent_theory42.py：Gaussian仿射风险、嵌套分解、查询秩、任务度量、range白化、谱定义。
- readers42.py：仿射、参数受限tanh、残差读者和同目标优化控制。
- run_reader_ladder.py：冻结、监督/选择预算、标签打乱、独立验证和实际原任务。
- run_task_subspaces.py：真实头的内容子空间、同秩随机方向及新评价池。
- run_latent_controls.py：已知答案。
- audit_selected_authors.py：固定作者定义的选定执行。
- verify_l42_artifacts.py：只读哈希、合同与保存场景重放。
- plot_l42.py：实际证据科学图。

当前原任务adapter为all-present physical_k=4的H1/C连续目标。辅助分类由预测潜变量的固定方向派生，不是原生分类模型；4像素图像是数学控制，未冒称真实重建实验。L4.3的内容选择性/动态干预和L4.4恢复尚未执行。
