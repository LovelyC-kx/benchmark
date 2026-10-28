# L4.5数据下载与适配交接单

2026-10-27。继承原时空容量计划和2026-10-21数据总表，不自动下载数据。下表是可执行优先级，不表示所有候选都要同时跑。原93条来源候选总表随代码包保留，未把候选数量变为完成任务数。

## 1. 第一优先：EarthNet2021＋U-TILISE受控缺失

**研究作用。** 在真实四波段图像上，把有干净目标的人工缺失、同一目标来源和真实云下未知目标分开；先检验H调用和C保真，再检验真实变化与不规则时相。

**官方入口。** [EarthNet](https://www.earthnet.tech/)，[U-TILISE代码/数据说明](https://github.com/prs-eth/U-TILISE)。

需要保存：

- 官方训练来源和既定iid/ood评价来源；从训练来源建立独立development validation，不能用iid test选择超参数。
- B2/B3/B4/B8、云/阴影mask与可用云概率、原日期/position_days、tile/ROI/source标识及所有切块/窗口的父来源。
- 干净目标帧和人工缺失mask；保持原辐射尺度/预处理版本，并记录无缺失筛选带来的分布变化。
- 原数据manifest、实际文件大小/数量/校验及缺帧记录；下载后才确认完整性。

作者约14GB的iid演示包是测试演示，不能代替训练集。原EarthNet名义五天网格，在删除缺失帧后可能变为不规则有效间隔；直接把剩余帧索引当物理时间不成立。

两条协议分别登记：

| 协议 | 模型可见 | 目标与比较 |
|---|---|---|
| 原生U-TILISE seq-to-seq | 允许规定完整序列及时间信息 | 复现/参考；不改称严格在线因果模型 |
| 本文因果当前目标 | 过去/当前且在deadline前真正可用 | ANN与SNN相同时间/输入权限、mask及目标 |

跨日期图像不自动是同刻无误差真值；只有确认干净的当前目标与人工遮挡才能直接计算该受控问题的误差。

## 2. 第二优先：真实观测/地域扩展

| 数据 | 官方入口 | 最小取用范围 | 主要缺口与约束 |
|---|---|---|---|
| SEN12MS-CR-TS | [作者仓库](https://github.com/PatrickTUM/SEN12MS-CR-TS) | 先多时相S2；SAR作为另一个实验因素 | 13波段映射明确；光学/SAR时间接近不等于同步；真实变化与旧历史分开 |
| AllClear | [作者项目](https://allclear.cs.cornell.edu/)、[代码](https://github.com/Zhou-Hangyu/allclear) | 固定ROI子集和官方来源划分，先S2后多模态 | 跨日期参考、地理相邻/重叠、尺度和传感器配准；不先下载全量 |
| SEN12MS-CR | 同上作者说明 | 单时相光学/SAR对照 | 它本身不是多时相保持任务，不作为长程历史主证据 |

ROI标识不同仍可能地理邻近或切块重叠。代码的source组集合检查只能排除相同ID跨split，实际坐标缓冲与空间重叠必须等数据到位后另审计。

## 3. 事件与spike camera分支

| 数据/套件 | 官方入口 | 取用要求 | 不能默认的事情 |
|---|---|---|---|
| E2VID公开序列/IJRR等 | [作者仓库](https://github.com/uzh-rpg/rpg_e2vid) | 只选有合法同步参考的序列，保留原时间戳与曝光 | 参考APS可能有运动模糊/动态范围限制，不是天然完美真值 |
| HQF | [作者项目](https://www.cedricscheerlinck.com/20ecnn) | 保留原序列边界、参考与预处理 | 小评价套件不当训练数据；所有滑窗跟随原序列split |
| MVSEC | [官方页面](https://daniilidis-group.github.io/mvsec/) | 同相机/同视角匹配、事件与帧时间及曝光 | depth/pose标注不能代替干净强度图 |
| DSEC | [官方页面](https://dsec.ifi.uzh.ch/) | event/RGB标定、曝光时间、rectification与有效裁剪 | 原生event与RGB并非逐像素同坐标；空白对齐区域不能计入重建真值 |
| SSMT对应spike camera数据 | [正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/37364/41326) | 从作者明示的数据版本/划分开始，保存窗口和片段reset规则 | spike camera不等于DVS，也不等于SNN模型；当前官方代码URL404，未伪造加载器 |

EVSNN和EvDiff的官方说明分别作为模型输入/对齐参照。不得只把event voxel输入代入当前五通道toy接口就宣称完成真实适配；应明确bins、目标通道、物理窗口、模拟步和状态reset。需要预训练模型时，外部训练权限/存储/计算另列。

## 4. 原数据总表中其余连续输出/视频候选

下面保留原D66—D80的角色，避免只剩本轮新找的数据。

| 原ID | 数据 | 本节角色 |
|---|---|---|
| D66 | SLoClas | 定位/分类扩展，不是像素或波形重建 |
| D67 | N-DNS | 后续因果连续输出/跨模态；官方仓库当前404，不把fork称官方 |
| D68 | AISHELL-1 | 语音识别扩展，不是恢复连续信号 |
| D69 | Vimeo-90K septuplet | [原作者](https://github.com/anchen1011/toflow)；视频恢复，中心帧未来权限与因果分支分开 |
| D70 | REDS | [官方](https://seungjunnah.github.io/Datasets/reds.html)；连续运动与更新，固定官方/REDS4划分之一 |
| D71 | DAVIS | [官方](https://davischallenge.org/davis2017/code.html)；遮挡/对象素材，分割mask不是被遮挡RGB真值 |
| D72 | Set8 | [原协议代码](https://github.com/m-tassano/fastdvdnet)；外部视频去噪测试，不当训练集 |
| D73 | GoPro | [官方](https://seungjunnah.github.io/Datasets/gopro.html)；多帧分析需要连续序列，单图对不等于时序任务 |
| D74 | BSD真实视频去模糊 | [作者代码](https://github.com/zzh-tech/ESTRNN)；不是Berkeley静态BSD |
| D75 | CRVD | [作者代码](https://github.com/cao-cong/RViDeNet)；RAW域、参考生成、ISP和序列split固定 |
| D76 | E2VID套件 | 见上，按合法同步参考选择 |
| D77 | MVSEC/DSEC | 见上，同传感器/目标/对齐逐项核对 |
| D78 | SEN12MS-CR-TS | 见上，多时相主扩展 |
| D79 | SEN12MS-CR | 见上，单时相对照 |
| D80 | AllClear | 见上，地域/观测外部检验 |

Diving48 V2保留为原分类与设计迁移路线，不挪成图像重建数据。现有自建合成序列直接生成，不需要下载；它不是任何真实传感器的物理仿真。

## 5. 已交付的输入适配边界

`real_reconstruction_contract45.py`可接收经过转换的U-TILISE式局部样本：

```python
sample = {
    'x': masked_frames,           # T,C,H,W
    'y': clean_frames,            # T,C,H,W
    'masks': cloud_masks,         # T,1,H,W；1表示遮挡
    'position_days': dates,       # T
    'availability_days': ready,   # T；包括派生预处理的真实可用时间
    'source_id': roi_id,
    'target_is_verified_clean': True,
}
item = from_utilise_sample(sample, target_index=target_index, mode='causal')
```

它输出四波段＋visible mask，目标独立；拒绝未验证干净目标、未来支持预处理和非法维度/波段。position_days返回供后续timed模型使用，当前参考SNN没有偷偷获得该信息。

本适配不是原生HDF5/GeoTIFF/ROS/event-h5全格式读取器。数据到位后仍需按官方加载器转换，固定radiometry/target/配准并核验内容；未伪造不存在的真实实验。

## 6. 下载后实际执行顺序

1. 核验来源、文件清单、日期/曝光、band、坏帧和目标质量；保留官方划分。
2. 按父ROI/原视频/原传感序列划split，所有裁剪与窗口随父来源；另检查地理相邻和内容重复。
3. 用当前帧、最近合法观测及同权限ANN参考检验任务可识别性。
4. 先复现原数据/模型协议，再建立明确标记的因果当前目标分支。
5. 冻结候选与指标后，再在独立序列/地域执行来源、载体、恢复和诊断指定的改动。

当前只需优先准备第一部分EarthNet/U-TILISE来源；其余分支按研究问题逐步加入。这份清单不要求现在下载全部候选。
