# Bamboo Cell Microscopy Video Pipeline

本项目用于竹细胞/维管束显微视频中的目标检测、跟踪、轨迹导出、热力图分析和实验指标汇总。当前检测模型基于 YOLO26，跟踪管线默认使用 OCSORT（boxmot，`--tracker bytetrack` 可切回 ByteTrack）。

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
├── train.py                  # YOLO 训练入口（YOLO26/YOLOv10 权重均可）
├── evaluate_detection.py     # 检测指标导出
├── detect_cam.py             # YOLO + OCSORT 视频管线（可切 ByteTrack）
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

本分支实测的关键版本：

- Python 3.12.9
- PyTorch 2.12.0+cu132
- Ultralytics 8.4.165
- NumPy 1.26.4
- OpenCV 4.10.0
- boxmot 12.0.1（dist 12.0.2）

`requirements.txt` 固定值（本分支未在此组合下实测）：Python 3.11、PyTorch 2.6.0 + CUDA 12.6、OpenCV 4.11.0、motmetrics 1.4.0。

注意：`requirements.txt` 固定 `numpy==1.26.4`。NumPy 2.x 移除了 `np.trapz`，Ultralytics 8.3.87 验证 mAP 时会因此报错；8.4.165 已按 NumPy 版本改用 `np.trapezoid`，不再触发。

## 数据与模型（Hugging Face）

大文件（模型权重、源视频）托管在 Hugging Face，**clone 后先下载到仓库对应路径**，再跑后续命令即可直接运行：

| 资源 | Hugging Face | 内容 |
| --- | --- | --- |
| 模型权重 | [LanluZ/vascular-bundle-yolov10](https://huggingface.co/LanluZ/vascular-bundle-yolov10) | `weights/best.pt`（最终，mAP50 0.9931）、`weights/best_previous.pt`（上一版，mAP50 0.9830） |
| 视频数据 | [LanluZ/vascular-bundle-media](https://huggingface.co/datasets/LanluZ/vascular-bundle-media) | `videos/56-fire.mp4`（演示源）、`videos/56-fire_tracked.mp4`（追踪成片），及 `21-air`/`22-microwave`/`24-oil`/`25-water`/`51-vapour`/`52-control` 等测试视频 |

下载 v10 存档模型权重（对照用；当前默认模型 YOLO26 由下文训练命令本地生成）：

```bash
hf download LanluZ/vascular-bundle-yolov10 weights/best.pt --local-dir runs/detect/bamboo_yolov10_20260515
```

下载演示源视频（对应 detect 命令的 `--source`）：

```bash
hf download LanluZ/vascular-bundle-media videos/56-fire.mp4 --local-dir .
```

也可用 Ultralytics 直接远程加载模型：

```python
from ultralytics import YOLO
model = YOLO("https://huggingface.co/LanluZ/vascular-bundle-yolov10/resolve/main/weights/best.pt")
```

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

默认从 `weights/yolo26m.pt` 开始训练：

```powershell
python train.py --model weights/yolo26m.pt --data data.yaml --epochs 50 --batch 8 --patience 30 --imgsz 640 --degrees 0 --device cuda:0 --name bamboo_yolo26_20260929
```

训练输出位于：

```text
runs/detect/<实验名>/
```

本次训练权重：

```text
runs/detect/bamboo_yolo26_20260929/weights/best.pt
```

本次（2026-09-29）实际训练用 `--epochs 150 --batch 16 --patience 30 --optimizer AdamW --lr0 0.001`（AMP），到第 144 轮触发 patience 早停，best mAP50-95 0.970（0.970 为独立 `val()` 复验值；`results.csv` best 行为 0.96936）。

## 检测指标

对验证集导出 Precision、Recall、mAP50、mAP50-95 和 F1：

```powershell
python evaluate_detection.py --model runs/detect/bamboo_yolo26_20260929/weights/best.pt --data data.yaml --device cuda:0 --name val_bamboo_yolo26_20260929 --output reports/detection_metrics_bamboo_yolo26_20260929.json
```

当前结果见：

```text
reports/detection_metrics_bamboo_yolo26_20260929.json
reports/detection_metrics_bamboo_yolo26_20260929.md
```

YOLO26 验证结果：

| Precision | Recall | mAP50 | mAP50-95 | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.9909 | 0.9758 | 0.9939 | 0.9695 | 0.9833 |

同一环境（Ultralytics 8.4.165）下的同源 v10 对照为 0.9576 / 0.9777 / 0.9910 / 0.9563 / 0.9676（`reports/detection_metrics_v10bamboo_84.json`）。两者都在同一验证集（2 张图、224 个实例）上评估，样本小、指标噪声大；8.3.87 环境下的历史存档值不可与本表直接比较。

v10 存档检测指标：P 0.9722 / R 0.9777 / mAP50 0.9931 / mAP50-95 0.9765 / F1 0.9749（`reports/detection_metrics_bamboo_yolov10_20260515.json`，8.3.87 存档，仅供历史参考）。

## 视频检测与跟踪

运行 YOLO + OCSORT 管线（默认追踪器 `ocsort`，`--tracker bytetrack` 可切回 ByteTrack）：

```powershell
python detect_cam.py --source videos/56-fire.mp4 --model runs/detect/bamboo_yolo26_20260929/weights/best.pt --conf 0.25 --device cuda:0 --rotate ccw90 --fps 5 --name 56-fire_bamboo_yolo26_20260929_c025 --tracker ocsort --clean
```

YOLO26 的置信度集中在 0.25–0.4（`--conf 0.58` 时 61 帧只有 334 个检出），因此本管线用 `--conf 0.25`。运行名后缀 `_c025` 表示该 conf 的那次运行；同名不带后缀的目录是 conf=0.58 的存档（该参数下管线几乎为空，仅作留档）。

输出目录：

```text
runs/pipeline/56-fire_bamboo_yolo26_20260929_c025/
├── frames/          # 绘制检测框和轨迹后的帧
├── csv/             # 每个 track 一个 CSV
├── tracks_mot.txt   # MOT 格式跟踪结果
├── timing.json      # 管线耗时
└── *_tracked.mp4    # 合成视频
```

## 追踪指标说明

项目目前没有“逐帧带 ID 的人工真值轨迹标注”，因此不能严谨计算正式的 IDF1、MOTA 和真实 ID Switch 次数。

当前报告采用可复现的启发式连续性指标：

- `track_count`：追踪器（OCSORT 或 ByteTrack）输出轨迹数
- `short_track_count`：短轨迹数
- `avg_track_length`：平均轨迹长度
- `fragmentation_proxy`：同一 track 内时间帧断裂次数的代理指标
- “过滤前后”对比：过滤小于 40 帧的短轨迹前后变化

生成汇总：

```powershell
python summarize_pipeline.py --csv-dir runs/pipeline/56-fire_bamboo_yolo26_20260929_c025/csv --min-track-length 40 --frame-count 61 --manual-basis objects --manual-seconds-per-object 1 --timing-json runs/pipeline/56-fire_bamboo_yolo26_20260929_c025/timing.json --output reports/pipeline_summary_56-fire_bamboo_yolo26_20260929_c025.json
```

当前 `56-fire.mp4` 启发式结果（yolo26 + OCSORT，conf 0.25）：

| Stage | Tracks | Detections | Short tracks | Avg length | Fragmentation proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before filtering | 565 | 8583 | 502 | 15.19 | 588 |
| After filtering | 63 | 2982 | 0 | 47.33 | 66 |

对比基线是 yolov10 存档（Ultralytics 8.3.87、v10 + ByteTrack、conf 0.58），两者环境与 conf 都不同，四行对比见「与 v10 存档及同环境对照」一节。

如需正式追踪指标，需要补充人工真值文件，推荐 MOTChallenge 格式：

```text
frame,id,x,y,w,h,conf,class,visibility
```

## 效率指标

自动管线会写出 `timing.json`（含 `tracker`、`conf` 字段）。本次 `56-fire.mp4`（yolo26 + OCSORT，conf 0.25）的自动处理时间：

| Item | Time |
| --- | ---: |
| Total automatic pipeline | 148.65 s |
| YOLO predict | 5.55 s |
| Tracker (OCSORT) | 16.36 s |
| Frame writing | 48.84 s |

按人工逐个维管束标记 `1.0 s/object` 估算，61 帧视频内有 `8583` 个未过滤轨迹观测点，人工耗时为 `8583.00 s`。因此可以写成：

> 单段视频处理时间从 8583.00 s 降到 148.65 s，约 57.74 倍提速。

对比基线是 yolov10 存档（total 115.42 s，Ultralytics 8.3.87 环境），见「与 v10 存档及同环境对照」一节。

如果你有真实人工计时，替换 `--manual-seconds-per-object` 即可重新生成报告。也可以用 `--manual-basis frames` 切回按帧估算。

## 与 v10 存档及同环境对照

同一段 `56-fire.mp4`（61 帧，`min_track_length=40`）在四种配置下的实测：

| 配置 | Ultralytics | conf | 过滤前 tracks / frag | 过滤后 tracks / frag | total |
| --- | --- | ---: | ---: | ---: | ---: |
| yolo26 + OCSORT（当前默认） | 8.4.165 | 0.25 | 565 / 588 | 63 / 66 | 148.65 s |
| v10 + OCSORT | 8.4.165 | 0.58 | 608 / 709 | 138 / 111 | 169.67 s |
| v10 + ByteTrack | 8.4.165 | 0.58 | 632 / 1800 | 195 / 413 | 144.99 s |
| v10 + ByteTrack（存档） | 8.3.87 | 0.58 | 320 / 838 | 53 / 128 | 115.42 s |

- 数据来自 `reports/pipeline_summary_*.json`（存档行为 `pipeline_summary_56-fire_bamboo_yolov10_20260515.json`）。
- 当前默认行 conf=0.25、其余行 conf=0.58：YOLO26 的置信度集中在 0.25–0.4（0.58 处 61 帧仅 334 个检出），v10 到 0.7 处仍有 17745 个检出，operating point 不同，跨这两组的差异同时包含 conf 差异。
- 存档行是 8.3.87 环境的历史记录；同环境复刻（8.4.165 的 v10 + ByteTrack 行）与存档不一致，存档行不作为同环境基线。
- 与存档相比，当前默认配置过滤后的可用轨迹量级相近（63 vs 53 条、2982 vs 2764 个观测）、过滤后 fragmentation 更少（66 vs 128）、总耗时更长（148.65 s vs 115.42 s）；两行之间模型、追踪器、conf、软件版本四个变量同时不同，只描述差异、不作归因。
- 同环境、同模型、同 conf 下只换追踪器（中间两行，即 v10 + OCSORT 与 v10 + ByteTrack 两行）：过滤后 fragmentation 413 → 111（-73%），通过 40 帧过滤的轨迹 195 → 138，track 段耗时 7.10 s → 20.30 s。

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

运行测试：

```powershell
python -m pytest tests/ -q
```

运行脚本语法检查：

```powershell
python -m py_compile metrics_utils.py evaluate_detection.py summarize_pipeline.py train.py detect_cam.py splicing.py vascular.py
```

## 常用命令速查

```powershell
# 训练
python train.py --model weights/yolo26m.pt --data data.yaml --epochs 50 --batch 8 --patience 30 --imgsz 640 --degrees 0 --device cuda:0 --name bamboo_yolo26_20260929

# 验证检测指标
python evaluate_detection.py --model runs/detect/bamboo_yolo26_20260929/weights/best.pt --data data.yaml --device cuda:0 --output reports/detection_metrics_bamboo_yolo26_20260929.json

# 跑视频管线（默认追踪器 ocsort，--tracker bytetrack 可切换）
python detect_cam.py --source videos/56-fire.mp4 --model runs/detect/bamboo_yolo26_20260929/weights/best.pt --conf 0.25 --device cuda:0 --rotate ccw90 --fps 5 --name 56-fire_bamboo_yolo26_20260929_c025 --tracker ocsort --clean

# 汇总追踪和效率指标
python summarize_pipeline.py --csv-dir runs/pipeline/56-fire_bamboo_yolo26_20260929_c025/csv --min-track-length 40 --frame-count 61 --manual-basis objects --manual-seconds-per-object 1 --timing-json runs/pipeline/56-fire_bamboo_yolo26_20260929_c025/timing.json --output reports/pipeline_summary_56-fire_bamboo_yolo26_20260929_c025.json
```
