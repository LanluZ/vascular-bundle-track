# Pipeline Summary

## Tracking Continuity

No frame-level identity ground truth was provided, so IDF1, MOTA, and true ID switches cannot be computed. These are heuristic continuity metrics for before/after comparison.

| Stage | Tracks | Detections | Short tracks | Avg length | Fragmentation proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before heuristic filtering | 565 | 8583 | 502 | 15.19 | 588 |
| After heuristic filtering | 63 | 2982 | 0 | 47.33 | 66 |

## Efficiency

- Frame count basis: `61`
- Vascular bundle annotation count basis: `8583`
- Manual basis: `objects`
- Manual estimate: `8583.00` seconds
- Automatic pipeline: `148.6523229999948` seconds
- Speedup: `57.74x`
- Sentence: 单段视频处理时间从 8583.00s 降到 148.65s。
