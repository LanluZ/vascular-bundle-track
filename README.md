# Bamboo Cell Microscopy Video Pipeline

本项目用于竹细胞/维管束显微视频中的目标检测、ByteTrack 跟踪、轨迹导出、热力图分析和实验指标汇总。当前检测模型基于 YOLOv10，跟踪管线使用 ByteTrack。

## 目录结构

```text
.
├── assets/                  # 归档的原始文件，不作为默认运行入口
│   ├── archives/             # 压缩包
│   ├── pickles/              # 历史 pkl 文件
│   └── raw_videos/           # 根目录迁移来的旧视频
├── data/                     # YOLO 数据集
│   ├── train/images
│   ├── train/labels
│   ├── val/images
│   └── val/labels
├── reports/                  # 指标报告和导出结果
│   └── figures/              # 热力图、colorbar 等图像产物
├── runs/                     # YOLO 和视频管线运行输出
├── tests/                    # 统计工具单元测试
├── video/                    # 默认视频输入目录
├── weights/                  # 模型权重
├── data.yaml                 # YOLO 数据配置
├── train.py                  # YOLOv10 训练入口
├── evaluate_detection.py     # 检测指标导出
├── detect_cam.py             # YOLO + ByteTrack 视频管线
├── summarize_pipeline.py     # 追踪启发式指标和效率汇总
├── metrics_utils.py          # 指标计算工具
├── screen.py                 # 轨迹过滤和中心点/面积补充
├── select_object.py          # 按轨迹裁剪目标图像
├── heatmap_draw.py           # 运动热图绘制
├── splicing.py               # 帧序列合成视频
└── vascular.py               # 轨迹数据结构和 CSV 导出
```

## 环境

建议使用独立 Python 环境，避免和全局 Anaconda 包冲突。

```powershell
pip install -r requirements.txt
```

已验证的关键版本：

- Python 3.11
- PyTorch 2.6.0 + CUDA 12.6
- Ultralytics 8.3.87
- NumPy 1.26.4
- OpenCV 4.11.0
- boxmot 19.0.0
- motmetrics 1.4.0

注意：Ultralytics 8.3.87 在当前环境下需要 `numpy<2`，否则验证 mAP 时会因为 `np.trapz` 缺失报错。

## 数据集

`data.yaml` 指向本项目内的 YOLO 数据集：

```yaml
# data.yaml — 路径相对本文件所在目录解析（train.py / evaluate_detection.py 会自动转为绝对路径，
# 因此不含用户名硬编码，任何机器 clone 后可直接运行）
train: "data/train/images"
val: "data/val/images"
names:
  0: Vascular
```

当前验证集较小：`data/val` 中有 2 张图、224 个标注实例。报告或论文中引用指标时，需要说明该验证集规模。

## 训练

默认从 `weights/yolov10m.pt` 开始训练：

```powershell
python train.py --model weights/yolov10m.pt --data data.yaml --epochs 50 --batch 8 --patience 30 --imgsz 640 --degrees 0 --device cuda:0 --name bamboo_yolov10_20260515
```

训练输出位于：

```text
runs/detect/<实验名>/
```

本次训练权重：

```text
runs/detect/bamboo_yolov10_20260515/weights/best.pt
```

## 模型权重（Hugging Face）

模型权重已托管到 Hugging Face，可直接下载或远程加载，无需从仓库 LFS 下载大文件：

- 模型页面：[LanluZ/vascular-bundle-yolov10](https://huggingface.co/LanluZ/vascular-bundle-yolov10)

下载到本地后使用：

```bash
hf download LanluZ/vascular-bundle-yolov10 weights/best.pt --local-dir .
```

或用 Ultralytics 远程加载：

```python
from ultralytics import YOLO
model = YOLO("https://huggingface.co/LanluZ/vascular-bundle-yolov10/resolve/main/weights/best.pt")
```

仓库内文件：

| 文件 | 说明 |
| --- | --- |
| `weights/best.pt` | 最终模型（mAP50 0.9931） |
| `weights/best_previous.pt` | 上一版模型（mAP50 0.9830） |

## 检测指标

对验证集导出 Precision、Recall、mAP50、mAP50-95 和 F1：

```powershell
python evaluate_detection.py --model runs/detect/bamboo_yolov10_20260515/weights/best.pt --data data.yaml --device cuda:0 --name val_bamboo_yolov10_20260515 --output reports/detection_metrics_bamboo_yolov10_20260515.json
```

当前结果见：

```text
reports/detection_metrics_bamboo_yolov10_20260515.json
reports/detection_metrics_bamboo_yolov10_20260515.md
reports/metrics_summary.md
```

本次新模型验证结果：

| Precision | Recall | mAP50 | mAP50-95 | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.9722 | 0.9777 | 0.9931 | 0.9765 | 0.9749 |

## 视频数据（Hugging Face）

源视频与检测/追踪演示成片已托管到 Hugging Face 数据集：[LanluZ/vascular-bundle-media](https://huggingface.co/datasets/LanluZ/vascular-bundle-media)。

下载演示源视频与成片：

```bash
hf download LanluZ/vascular-bundle-media videos/56-fire.mp4 --local-dir video
hf download LanluZ/vascular-bundle-media videos/56-fire_tracked.mp4 --local-dir .
```

仓库内视频（`videos/`）：`56-fire.mp4`（演示源）、`56-fire_tracked.mp4`（追踪成片），以及其它测试视频（`21-air`、`22-microwave`、`24-oil`、`25-water`、`51-vapour`、`52-control` 等）。

## 视频检测与跟踪

运行 YOLO + ByteTrack 管线：

```powershell
python detect_cam.py --source video/56-fire.mp4 --model runs/detect/bamboo_yolov10_20260515/weights/best.pt --conf 0.58 --device cuda:0 --rotate ccw90 --fps 5 --name 56-fire_bamboo_yolov10_20260515 --clean
```

输出目录：

```text
runs/pipeline/56-fire_bamboo_yolov10_20260515/
├── frames/          # 绘制检测框和轨迹后的帧
├── csv/             # 每个 track 一个 CSV
├── tracks_mot.txt   # MOT 格式跟踪结果
├── timing.json      # 管线耗时
└── *_tracked.mp4    # 合成视频
```

## 追踪指标说明

项目目前没有“逐帧带 ID 的人工真值轨迹标注”，因此不能严谨计算正式的 IDF1、MOTA 和真实 ID Switch 次数。

当前报告采用可复现的启发式连续性指标：

- `track_count`：ByteTrack 输出轨迹数
- `short_track_count`：短轨迹数
- `avg_track_length`：平均轨迹长度
- `fragmentation_proxy`：同一 track 内时间帧断裂次数的代理指标
- “过滤前后”对比：过滤小于 40 帧的短轨迹前后变化

生成汇总：

```powershell
python summarize_pipeline.py --csv-dir runs/pipeline/56-fire_bamboo_yolov10_20260515/csv --min-track-length 40 --frame-count 61 --manual-basis objects --manual-seconds-per-object 1 --timing-json runs/pipeline/56-fire_bamboo_yolov10_20260515/timing.json --output reports/pipeline_summary_56-fire_bamboo_yolov10_20260515.json
```

当前 `56-fire.mp4` 启发式结果：

| Stage | Tracks | Detections | Short tracks | Avg length | Fragmentation proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before filtering | 320 | 5908 | 267 | 18.46 | 838 |
| After filtering | 53 | 2764 | 0 | 52.15 | 128 |

如需正式追踪指标，需要补充人工真值文件，推荐 MOTChallenge 格式：

```text
frame,id,x,y,w,h,conf,class,visibility
```

## 效率指标

自动管线会写出 `timing.json`。本次 `56-fire.mp4` 的自动处理时间：

| Item | Time |
| --- | ---: |
| Total automatic pipeline | 115.42 s |
| YOLO predict | 6.40 s |
| ByteTrack | 2.07 s |
| Frame writing | 58.51 s |

按人工逐个维管束标记 `1.0 s/object` 估算，61 帧视频内有 `5908` 个未过滤轨迹观测点，人工耗时为 `5908.00 s`。因此可以写成：

> 单段视频处理时间从 5908.00 s 降到 115.42 s，约 51.19 倍提速。

如果你有真实人工计时，替换 `--manual-seconds-per-object` 即可重新生成报告。也可以用 `--manual-basis frames` 切回按帧估算。

## 热力图和后处理

轨迹 CSV 可以继续用于原有后处理：

```powershell
python screen.py
python select_object.py
python heatmap_draw.py
```

热力图输出建议放在：

```text
reports/figures/
```

## 测试

运行统计工具测试：

```powershell
python -m pytest tests/test_metrics_utils.py -q
```

运行脚本语法检查：

```powershell
python -m py_compile metrics_utils.py evaluate_detection.py summarize_pipeline.py train.py detect_cam.py splicing.py vascular.py
```

## 常用命令速查

```powershell
# 训练
python train.py --model weights/yolov10m.pt --data data.yaml --epochs 50 --batch 8 --patience 30 --imgsz 640 --degrees 0 --device cuda:0 --name bamboo_yolov10_20260515

# 验证检测指标
python evaluate_detection.py --model runs/detect/bamboo_yolov10_20260515/weights/best.pt --data data.yaml --device cuda:0 --output reports/detection_metrics_bamboo_yolov10_20260515.json

# 跑视频管线
python detect_cam.py --source video/56-fire.mp4 --model runs/detect/bamboo_yolov10_20260515/weights/best.pt --conf 0.58 --device cuda:0 --rotate ccw90 --fps 5 --name 56-fire_bamboo_yolov10_20260515 --clean

# 汇总追踪和效率指标
python summarize_pipeline.py --csv-dir runs/pipeline/56-fire_bamboo_yolov10_20260515/csv --min-track-length 40 --frame-count 61 --manual-basis objects --manual-seconds-per-object 1 --timing-json runs/pipeline/56-fire_bamboo_yolov10_20260515/timing.json --output reports/pipeline_summary_56-fire_bamboo_yolov10_20260515.json
```
