import argparse
from pathlib import Path

from metrics_utils import estimate_manual_object_time, estimate_manual_time, save_json, summarize_track_csv_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize heuristic tracking and processing efficiency.")
    parser.add_argument("--csv-dir", default="csv", help="Directory containing one track CSV per object.")
    parser.add_argument("--min-track-length", type=int, default=40)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--manual-seconds-per-frame", type=float, default=3.0)
    parser.add_argument("--manual-seconds-per-object", type=float, default=1.0)
    parser.add_argument(
        "--manual-basis",
        choices=["objects", "frames"],
        default="objects",
        help="Estimate manual work per detected vascular bundle instance or per video frame.",
    )
    parser.add_argument("--auto-seconds", type=float, default=None)
    parser.add_argument("--timing-json", default=None, help="Optional timing JSON from detect_cam.py.")
    parser.add_argument("--output", default="reports/pipeline_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize_track_csv_dir(args.csv_dir, min_track_length=args.min_track_length)
    frame_count = args.frame_count or _infer_frame_count(summary)
    object_count = int(summary["before"]["detection_count"])
    if args.manual_basis == "objects":
        manual = estimate_manual_object_time(object_count, args.manual_seconds_per_object)
    else:
        manual = estimate_manual_time(frame_count, args.manual_seconds_per_frame)
    auto_seconds = args.auto_seconds if args.auto_seconds is not None else _read_auto_seconds(args.timing_json)
    efficiency = {
        "frame_count": frame_count,
        "object_count": object_count,
        "manual_basis": args.manual_basis,
        "manual_seconds_per_frame": args.manual_seconds_per_frame,
        "manual_seconds_per_object": args.manual_seconds_per_object,
        "manual": manual,
        "auto_seconds": auto_seconds,
        "speedup": manual["seconds"] / auto_seconds if auto_seconds and auto_seconds > 0 else None,
    }
    report = {
        "tracking": summary,
        "efficiency": efficiency,
    }
    output = Path(args.output)
    save_json(report, output)
    _write_markdown(report, output.with_suffix(".md"))
    print(f"Saved pipeline summary to {output}")


def _infer_frame_count(summary: dict) -> int:
    return int(summary["before"]["detection_count"])


def _read_auto_seconds(timing_json: str | None) -> float | None:
    if not timing_json:
        return None
    import json

    path = Path(timing_json)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("total_seconds", 0.0))


def _write_markdown(report: dict, path: Path) -> None:
    tracking = report["tracking"]
    efficiency = report["efficiency"]
    before = tracking["before"]
    after = tracking["after"]
    lines = [
        "# Pipeline Summary",
        "",
        "## Tracking Continuity",
        "",
        tracking["note"],
        "",
        "| Stage | Tracks | Detections | Short tracks | Avg length | Fragmentation proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Before heuristic filtering | {before['track_count']} | {before['detection_count']} | "
            f"{before['short_track_count']} | {before['avg_track_length']:.2f} | {before['fragmentation_proxy']} |"
        ),
        (
            f"| After heuristic filtering | {after['track_count']} | {after['detection_count']} | "
            f"{after['short_track_count']} | {after['avg_track_length']:.2f} | {after['fragmentation_proxy']} |"
        ),
        "",
        "## Efficiency",
        "",
        f"- Frame count basis: `{efficiency['frame_count']}`",
        f"- Vascular bundle annotation count basis: `{efficiency['object_count']}`",
        f"- Manual basis: `{efficiency['manual_basis']}`",
        f"- Manual estimate: `{efficiency['manual']['seconds']:.2f}` seconds",
        f"- Automatic pipeline: `{efficiency['auto_seconds']}` seconds",
    ]
    if efficiency["speedup"] is not None:
        lines.append(f"- Speedup: `{efficiency['speedup']:.2f}x`")
        lines.append(
            f"- Sentence: 单段视频处理时间从 {efficiency['manual']['seconds']:.2f}s 降到 "
            f"{efficiency['auto_seconds']:.2f}s。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
