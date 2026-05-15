# Pipeline Summary

## Tracking Continuity

No frame-level identity ground truth was provided, so IDF1, MOTA, and true ID switches cannot be computed. These are heuristic continuity metrics for before/after comparison.

| Stage | Tracks | Detections | Short tracks | Avg length | Fragmentation proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before heuristic filtering | 320 | 5908 | 267 | 18.46 | 838 |
| After heuristic filtering | 53 | 2764 | 0 | 52.15 | 128 |

## Efficiency

- Frame count basis: `61`
- Vascular bundle annotation count basis: `5908`
- Manual basis: `objects`
- Manual estimate: `5908.00` seconds
- Automatic pipeline: `115.42362260000027` seconds
- Speedup: `51.19x`
- Sentence: 单段视频处理时间从 5908.00s 降到 115.42s。
