# Pipeline Summary

## Tracking Continuity

No frame-level identity ground truth was provided, so IDF1, MOTA, and true ID switches cannot be computed. These are heuristic continuity metrics for before/after comparison.

| Stage | Tracks | Detections | Short tracks | Avg length | Fragmentation proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before heuristic filtering | 608 | 12718 | 470 | 20.92 | 709 |
| After heuristic filtering | 138 | 7001 | 0 | 50.73 | 111 |

## Efficiency

- Frame count basis: `61`
- Vascular bundle annotation count basis: `12718`
- Manual basis: `objects`
- Manual estimate: `12718.00` seconds
- Automatic pipeline: `169.6698397` seconds
- Speedup: `74.96x`
- Sentence: 单段视频处理时间从 12718.00s 降到 169.67s。
