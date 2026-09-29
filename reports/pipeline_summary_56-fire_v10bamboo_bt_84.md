# Pipeline Summary

## Tracking Continuity

No frame-level identity ground truth was provided, so IDF1, MOTA, and true ID switches cannot be computed. These are heuristic continuity metrics for before/after comparison.

| Stage | Tracks | Detections | Short tracks | Avg length | Fragmentation proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before heuristic filtering | 632 | 16641 | 437 | 26.33 | 1800 |
| After heuristic filtering | 195 | 10247 | 0 | 52.55 | 413 |

## Efficiency

- Frame count basis: `61`
- Vascular bundle annotation count basis: `16641`
- Manual basis: `objects`
- Manual estimate: `16641.00` seconds
- Automatic pipeline: `144.99181099999987` seconds
- Speedup: `114.77x`
- Sentence: 单段视频处理时间从 16641.00s 降到 144.99s。
