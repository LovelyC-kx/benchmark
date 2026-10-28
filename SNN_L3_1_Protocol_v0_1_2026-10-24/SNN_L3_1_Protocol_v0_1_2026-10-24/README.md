# L3第一单元：可审议协议与参考代码

先读 PROTOCOL.md，再核对 protocol.json。

已实现：NumPy数据/配对/标签/可辨识参照，以及配对响应几何。
已核对：从L2父包恢复的Cell与ModelConfig源码接口。
尚未执行：新SNN训练、PyTorch适配验收、分类训练、重建和正式几何预测。

## 文件

- PROTOCOL.md：完整执行协议、分析判据与论文用途。
- protocol.json：明确默认值；不是已执行的实验记录。
- factor_data.py：同源输入、项目标签、单因素/噪声配对、合法标签与输入参照。
- paired_geometry.py：配对变化矩阵、独立拟合的线性方向、目标方向投影。
- l2_adapter_example.py：依赖上一份L2包与PyTorch的适配例；未运行。
- tests/test_reference.py：数据、权限、可辨识与几何控制。
- verification.json、test_log.txt：本轮实际检查范围和结果。
- parent_provenance.json：父包字节与源码校验。

## 运行参考检查

~~~bash
python -m pip install -r requirements-reference.txt
python -m unittest discover -s tests -v
~~~

## 生成与配对

~~~python
from factor_data import FactorConfig, generate, make_pair, regression_target, model_payload
cfg = FactorConfig()
base = generate(cfg, 24, "audit_test")
pair = make_pair(cfg, base, "B")
x = model_payload(base)["x"]  # only this field goes into a model
y = regression_target(base, "joint")
y_pair = regression_target(pair, "joint")  # always recomputed
~~~

合成数据无需下载。不要把这个带标记保持任务当成已证明需要相对顺序的任务。
父L2包不在本压缩包内，需将其snnbench放入Python导入路径后才能使用适配例。
本包没有修改父包任务、已有checkpoint或GitHub仓库。

