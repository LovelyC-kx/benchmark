# 服务器数据下载与转换交接

完整机器清单：[JSON](../snnbench/resources/datasets.json)、[CSV](../snnbench/resources/datasets.csv)。完整可读目录：[121项总表](DATASETS_ALL.md)。本轮新增真实任务、适配合同与模型配套见 [扩展说明](REAL_DATA_AND_MODEL_EXPANSION.md)。

原93项是来源、生成器、任务族和参照协议。加EarthNet、Diving48及本轮17项真实预测/生物信号来源后，共112项来源/任务/素材；另有无需下载的G01—G09。不能计为121个真实数据集或完成实验。

注册直接文件共24个，其中5个NSA作者预处理包默认跳过。`--include-native`才获取这些可选包；它们可能没有subject/session身份，不自动满足本文隔离协议。新增真实来源优先下载脚本是`bash scripts/download_real_priority.sh /data/snn/raw`；CSV/IDX/WAV/WISDM/subject-MAT命令详见扩展说明。

## 优先下载的文件

| ID | 文件/入口 | 取用内容 | 适配 |
|---|---|---|---|
| D31 | [SHD train](https://zenkelab.org/datasets/shd/shd_train.h5.gz)、[test](https://zenkelab.org/datasets/shd/shd_test.h5.gz) | 时间、unit、类别及原划分 | HDF5原始适配 |
| D32 | [SSC train](https://zenkelab.org/datasets/ssc/ssc_train.h5.gz)、[valid](https://zenkelab.org/datasets/ssc/ssc_valid.h5.gz)、[test](https://zenkelab.org/datasets/ssc/ssc_test.h5.gz) | 原训练/验证/测试 | HDF5原始适配 |
| D33 | [Speech Commands v0.02](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) | WAV及原validation/testing列表 | 35词版本，因果FFT特征；与TFDS12分类协议分开 |
| D35 | [Tonic Gesture train](https://figshare.com/ndownloader/files/38022171)、[test](https://figshare.com/ndownloader/files/38020584) | 已切片NPY，保留user目录 | 维护者预处理版，毫秒时间精度；不是原始IBM AEDAT |
| D46 | [UCI HAR压缩包](https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip) | `Inertial Signals`、y、subject | 9通道、128步原时序；不用561维汇总特征 |
| D94 | [EarthNet来源](https://www.earthnet.tech/)、[U-TILISE作者处理说明](https://github.com/prs-eth/U-TILISE) | 正式train与iid/ood，B2/B3/B4/B8、mask、日期与ROI | 先作者转换，再因果清单；约14GB演示包只有iid测试，不能代替训练集 |

SHD/SSC的地址已更新为 `datasets/shd/` 和 `datasets/ssc/` 子目录。校验来自[官方MD5表](https://zenkelab.org/datasets/md5sums.txt)。Gesture的URL与MD5来自[Tonic维护者源码](https://tonic.readthedocs.io/en/latest/_modules/tonic/datasets/dvsgesture.html)。HAR下载按钮与[UCI官方页面](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)一致。当前核对日期2026-10-29，未在本机下载完整真实语料。

```bash
python -m snnbench.download --list --tier all
python -m snnbench.download --ids D31 D32 D33 D35 D46 --out /data/snn/raw --extract
python -m snnbench.download --ids D31 D32 D35 --out /data/snn/raw --verify-only
```

保留 `.part` 和 `.part.json` 可续传；服务器忽略Range时从头重写，不追加损坏文件。提供方无公开校验的文件记录本地SHA256，不能把它称作官方校验。压缩包解压拒绝越界路径、链接和设备文件。

## 实际转换命令

```bash
python -m snnbench prepare har --root /data/snn/raw/D46/extracted --out /data/snn/canonical/har
python -m snnbench prepare spikes --dataset SHD \
  --train /data/snn/raw/D31/extracted/shd_train.h5 \
  --test /data/snn/raw/D31/extracted/shd_test.h5 \
  --steps 100 --window 1.4 --out /data/snn/canonical/shd
python -m snnbench prepare spikes --dataset SSC \
  --train /data/snn/raw/D32/extracted/ssc_train.h5 \
  --val /data/snn/raw/D32/extracted/ssc_valid.h5 \
  --test /data/snn/raw/D32/extracted/ssc_test.h5 \
  --steps 100 --window 1.4 --out /data/snn/canonical/ssc
python -m snnbench prepare gsc --root /data/snn/raw/D33/extracted --out /data/snn/canonical/gsc35
python -m snnbench prepare gesture --root /data/snn/raw/D35/extracted \
  --steps 100 --window 3 --size 32 --out /data/snn/canonical/gesture
python -m snnbench verify-data --root /data/snn/canonical/har
```

其余每个转换结果也运行 `verify-data`。HAR的UCI外层ZIP内可能另含 `UCI HAR Dataset.zip`：若如此，先将内层包再解压到同一个 `extracted`，直到找到 `UCI HAR Dataset/train/Inertial Signals`。转换器不猜测汇总特征或隐藏解包失败。

事件数按固定物理窗口分箱，输入在每个箱子的右边界才可用；SHD/SSC窗口之外的事件被裁掉，窗口长度是实验参数。Gesture默认相对首事件取3秒前缀并作空间计数分箱，必须与完整手势或其他窗口协议区分。

## 真实重建、视频和无线扩展

优先保留原计划的[Vimeo-90K七帧](https://github.com/anchen1011/toflow)、[REDS](https://seungjunnah.github.io/Datasets/reds.html)、[SEN12MS-CR-TS](https://github.com/PatrickTUM/SEN12MS-CR-TS)、[AllClear](https://github.com/Zhou-Hangyu/allclear)、[MVSEC](https://daniilidis-group.github.io/mvsec/)、[DSEC](https://dsec.ifi.uzh.ch/)、[Diving48 V2](https://mmaction2.readthedocs.io/en/latest/dataset_zoo/diving48.html)。完整候选逐项见总表，未把访问失效项标成可自动下载。

作者原始格式经其加载器读出后，按 `examples/tensor_manifest.schema-example.json` 导出 `TF` 或 `TCHW` 的NPY/NPZ。每行必须给父来源、官方划分、观测时间、真正可用时间与目标deadline；重建还必须明确干净真值已验证。转换命令：

```bash
python -m snnbench prepare manifest --manifest exported/manifest.json \
  --objective reconstruction --out /data/snn/canonical/earthnet_causal
python -m snnbench verify-data --root /data/snn/canonical/earthnet_causal
```

已另增EarthNet作者转换HDF5的专用入口。它读取 `highresdynamic` 与作者 `idx_good_frames`，取合法过去/当前帧，构造本文人工矩形缺失和当前干净目标，保留原帧索引对应日期。默认tile分组，建议先用OOD测试来源：

```bash
python -m snnbench prepare earthnet --train /data/earthnet/train.hdf5 \
  --test /data/earthnet/ood_test_split.hdf5 --steps 6 --size 32 \
  --group-unit tile --out /data/snn/canonical/earthnet_causal
python -m snnbench verify-data --root /data/snn/canonical/earthnet_causal
```

schema按作者commit `1ea50fe1e4f1840a14b3f23062334077fc5f943b`的转换器和加载器核对。本文人工矩形不是作者真实云mask仿真；短于T的干净序列被登记排除，异常真值会报错。IID若同tile跨官方训练/测试，默认分组会拒绝；另用cube协议前必须明确地理重叠与原生IID权限。该适配不支持把约14GB预处理演示HDF5冒充训练用的原始转换HDF5。

这些接口不会自动解决event/RGB标定、云下未知真值、GeoTIFF波段/辐射尺度、SAR不同步、CSI复数编码、地理邻近与裁剪重叠。它们会拒绝未声明的时间和目标权限；这些领域审计必须在实际数据到位后完成。

## 统一数据划分

训练来源划为train、val、calibration、probe_train、probe_val、verification_train、verification_val、development；原官方val存在时用于val/calibration；官方test来源进一步分为confirmation与audit。父来源不跨角色，训练从不读取测试。每个角色样本数量写入 `dataset.json`。

HAR按subject、Gesture按user、GSC按speaker隔离；SHD/SSC当前按原recording隔离，**没有声称speaker完全隔离**。数据不足不会循环复制以补足请求数量。确认/审计合起来保留官方test来源，但不再与未经重划分的全test单一分数混报。
