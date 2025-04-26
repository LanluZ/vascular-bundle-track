# Vascular Bundle Tracking System

## 项目概述
这是一个基于YOLOv10的维管束(vascular bundle)检测与跟踪系统，主要用于植物生理学研究中的维管束运动分析。

## 主要功能
- 实时维管束检测与跟踪
- 运动轨迹数据记录与分析
- 视频帧处理与切割
- 数据过滤与清洗

## 文件说明

### 核心脚本
| 文件 | 功能 |
|------|------|
| `detect_cam.py` | 主检测程序，实现目标检测与跟踪 |
| `screen.py` | 数据过滤，去除短时跟踪数据 |
| `select_object.py` | 视频帧切割工具 |
| `heatmap_draw.py` | 运动热图生成 |
| `train.py` | 模型训练脚本 |

## 安装与使用

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+

### 快速开始
1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 运行检测程序:
```bash
python detect_cam.py --source video.mp4
```

3. 生成热图:
```bash
python heatmap_draw.py --input csv/tracks.csv
```

## 致谢

本项目基于以下优秀开源项目构建，特此致谢：

- [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) - YOLOv10目标检测模型
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) - YOLO框架实现
- [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot) - 目标跟踪算法