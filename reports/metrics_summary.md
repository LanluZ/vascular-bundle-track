# Bamboo Microscopy Pipeline Metrics

## Detection

Validation set: `data/val` with 2 images and 224 labeled instances.

| Model | Precision | Recall | mAP50 | mAP50-95 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `weights/best.pt` | 0.9270 | 0.9550 | 0.9830 | 0.7630 | 0.9408 |
| `runs/detect/bamboo_yolov10_20260515/weights/best.pt` | 0.9722 | 0.9777 | 0.9931 | 0.9765 | 0.9749 |

The new run was trained with:

```powershell
python train.py --model weights/yolov10m.pt --data data.yaml --epochs 50 --batch 8 --patience 30 --imgsz 640 --degrees 0 --device cuda:0 --name bamboo_yolov10_20260515
```

The independent validation export was generated with:

```powershell
python evaluate_detection.py --model runs/detect/bamboo_yolov10_20260515/weights/best.pt --data data.yaml --device cuda:0 --name val_bamboo_yolov10_20260515 --output reports/detection_metrics_bamboo_yolov10_20260515.json
```

## Tracking

No frame-level identity ground truth is available, so formal IDF1, MOTA, and true ID Switch counts cannot be computed honestly. The table below is a reproducible heuristic continuity comparison before and after filtering tracks shorter than 40 frames.

Source video: `video/56-fire.mp4`

| Stage | Tracks | Detections | Short tracks | Avg track length | Fragmentation proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before heuristic filtering | 320 | 5908 | 267 | 18.46 | 838 |
| After heuristic filtering | 53 | 2764 | 0 | 52.15 | 128 |

Heuristic comparison:

- Removed short tracks: 267
- Removed detections: 3144
- Fragmentation proxy change: -710

## Efficiency

Automatic pipeline command:

```powershell
python detect_cam.py --source video/56-fire.mp4 --model runs/detect/bamboo_yolov10_20260515/weights/best.pt --conf 0.58 --device cuda:0 --rotate ccw90 --fps 5 --name 56-fire_bamboo_yolov10_20260515 --clean
```

Measured automatic pipeline time:

| Item | Value |
| --- | ---: |
| Frames | 61 |
| Total automatic time | 115.42 s |
| YOLO predict time | 6.40 s |
| ByteTrack time | 2.07 s |
| Frame writing time | 58.51 s |

Manual extraction estimate uses the unfiltered vascular bundle annotation count: 5908 track observations at 1.0 second per vascular bundle mark. With that assumption, single-video processing time goes from 5908.00 s manually to 115.42 s automatically, a 51.19x speedup.

Change the manual estimate with:

```powershell
python summarize_pipeline.py --csv-dir runs/pipeline/56-fire_bamboo_yolov10_20260515/csv --min-track-length 40 --frame-count 61 --manual-basis objects --manual-seconds-per-object 1.5 --timing-json runs/pipeline/56-fire_bamboo_yolov10_20260515/timing.json --output reports/pipeline_summary_56-fire_bamboo_yolov10_20260515_manual1p5s.json
```
