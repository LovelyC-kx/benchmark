# v0.3扩展验收

52项完整unittest通过；原模型合同测试现遍历14骨架×12单元=168组合，检查原生损失、有限梯度、采集一致性及完整状态重放。18种连续参照也覆盖兼容目标与采集路径。

新增格式测试检查IDX扫描/固定置换/重复、ESC-50父录音隔离、WISDM的subject/活动边界、OpenBMI的subject与块平均时间、预测的未来目标形状与无跨角色原始索引、train-only尺度、minimum_first长预测配置。另测位置编码在NoTD下仍保留外生stream clock、专用SSM初始化、作者源码commit/改动守卫与可选作者数据包默认跳过。

4个短CPU实验分别训练Mamba-style回归、位置Transformer分类、ConvLIF-RNN重建、完整H×F预测fixture。每个2轮、32训练场景、8次参数更新；4个参数hash均改变，全部通过14模块分析入口，数学上不适用的分支继续按合同记录。最后在最终Python源代码下重新分析四份固定checkpoint；见cpu_acceptance.json及各自receipt。

所有原始语料测试都是人工格式fixture。预测CSV fixture是合成正弦，不是ETT真实结果；没有GPU、完整真实语料、充分收敛、官方模型训练复现或优越性证据。19种新参考实现也不能称为19篇官方方法复现。

profile_counts.json是配置数量：expanded每seed 4954；full搭配expanded名单每seed 63054、5seed 315270。数量不是完成训练数，也不表示对每份配置都进行了前向预检。

summary.json记录Python/PyTorch环境、训练/分析源码hash；unit_tests.log保存完整52测试日志。upstream_pins.json固定21个上游版本；9个原生suite只完成运行入口与本地人工脚本守卫验收，真实作者环境尚待服务器部署。

父目录的42测试/96组合/8短运行证据属于v0.2历史验收，原样保留；不要用它覆盖v0.3状态，也不要把两轮小实验相加当作论文训练结果。
