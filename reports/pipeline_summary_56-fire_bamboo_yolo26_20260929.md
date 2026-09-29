# Pipeline Summary

## Tracking Continuity

No frame-level identity ground truth was provided, so IDF1, MOTA, and true ID switches cannot be computed. These are heuristic continuity metrics for before/after comparison.

| Stage | Tracks | Detections | Short tracks | Avg length | Fragmentation proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before heuristic filtering | 24 | 93 | 24 | 3.88 | 13 |
| After heuristic filtering | 0 | 0 | 0 | 0.00 | 0 |

## Efficiency

- Frame count basis: `61`
- Vascular bundle annotation count basis: `93`
- Manual basis: `objects`
- Manual estimate: `93.00` seconds
- Automatic pipeline: `120.96423250000225` seconds
- Speedup: `0.77x`
- Sentence: 单段视频处理时间从 93.00s 降到 120.96s。
