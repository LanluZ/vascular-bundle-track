import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def compute_f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return 2 * precision * recall / denominator


def estimate_manual_time(frame_count: int, seconds_per_frame: float) -> dict[str, float]:
    seconds = float(frame_count) * float(seconds_per_frame)
    return format_duration(seconds)


def estimate_manual_object_time(object_count: int, seconds_per_object: float) -> dict[str, float]:
    seconds = float(object_count) * float(seconds_per_object)
    return format_duration(seconds)


def format_duration(seconds: float) -> dict[str, float]:
    return {
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
    }


def summarize_track_csv_dir(csv_dir: str | Path, min_track_length: int = 40) -> dict[str, Any]:
    tracks = _load_tracks(Path(csv_dir))
    before = _summarize_tracks(tracks, min_track_length=min_track_length)
    filtered = [track for track in tracks if track["length"] >= min_track_length]
    after = _summarize_tracks(filtered, min_track_length=min_track_length)
    return {
        "min_track_length": min_track_length,
        "before": before,
        "after": after,
        "comparison": {
            "removed_short_tracks": before["track_count"] - after["track_count"],
            "removed_detections": before["detection_count"] - after["detection_count"],
            "fragmentation_proxy_delta": after["fragmentation_proxy"] - before["fragmentation_proxy"],
        },
        "note": (
            "No frame-level identity ground truth was provided, so IDF1, MOTA, and true ID switches "
            "cannot be computed. These are heuristic continuity metrics for before/after comparison."
        ),
    }


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_tracks(csv_dir: Path) -> list[dict[str, Any]]:
    tracks = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        rows = _read_track_rows(csv_path)
        if not rows:
            continue
        times = sorted(int(float(row["time"])) for row in rows if row.get("time") not in (None, ""))
        confs = [float(row["conf"]) for row in rows if row.get("conf") not in (None, "")]
        tracks.append(
            {
                "id": csv_path.stem,
                "length": len(rows),
                "times": times,
                "mean_conf": mean(confs) if confs else 0.0,
            }
        )
    return tracks


def _read_track_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _summarize_tracks(tracks: list[dict[str, Any]], min_track_length: int) -> dict[str, Any]:
    lengths = [track["length"] for track in tracks]
    detection_count = sum(lengths)
    return {
        "track_count": len(tracks),
        "detection_count": detection_count,
        "short_track_count": sum(1 for length in lengths if length < min_track_length),
        "avg_track_length": mean(lengths) if lengths else 0.0,
        "median_track_length": median(lengths) if lengths else 0.0,
        "avg_conf": mean(track["mean_conf"] for track in tracks) if tracks else 0.0,
        "fragmentation_proxy": sum(_gap_count(track["times"]) for track in tracks),
    }


def _gap_count(times: list[int]) -> int:
    if len(times) < 2:
        return 0
    return sum(1 for previous, current in zip(times, times[1:]) if current - previous > 1)
