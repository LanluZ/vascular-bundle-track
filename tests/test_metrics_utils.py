from pathlib import Path

from metrics_utils import (
    compute_f1,
    estimate_manual_object_time,
    estimate_manual_time,
    summarize_track_csv_dir,
)


def _write_track(path: Path, rows: list[dict]) -> None:
    fields = ["xmin", "ymin", "xmax", "ymax", "conf", "cls", "time"]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[field]) for field in fields))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_compute_f1_returns_zero_when_precision_and_recall_are_zero():
    assert compute_f1(0.0, 0.0) == 0.0


def test_compute_f1_uses_harmonic_mean():
    assert round(compute_f1(0.8, 0.5), 4) == 0.6154


def test_summarize_track_csv_dir_compares_before_and_after_filter(tmp_path):
    _write_track(
        tmp_path / "1.csv",
        [
            {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "conf": 0.9, "cls": 0, "time": 0},
            {"xmin": 1, "ymin": 0, "xmax": 11, "ymax": 10, "conf": 0.8, "cls": 0, "time": 1},
            {"xmin": 2, "ymin": 0, "xmax": 12, "ymax": 10, "conf": 0.7, "cls": 0, "time": 2},
        ],
    )
    _write_track(
        tmp_path / "2.csv",
        [{"xmin": 20, "ymin": 20, "xmax": 30, "ymax": 30, "conf": 0.6, "cls": 0, "time": 0}],
    )

    summary = summarize_track_csv_dir(tmp_path, min_track_length=2)

    assert summary["before"]["track_count"] == 2
    assert summary["before"]["short_track_count"] == 1
    assert summary["after"]["track_count"] == 1
    assert summary["after"]["fragmentation_proxy"] == 0
    assert summary["comparison"]["removed_short_tracks"] == 1


def test_estimate_manual_time_formats_seconds_from_frame_count():
    result = estimate_manual_time(frame_count=120, seconds_per_frame=3.5)

    assert result["seconds"] == 420.0
    assert result["minutes"] == 7.0


def test_estimate_manual_object_time_uses_annotation_count():
    result = estimate_manual_object_time(object_count=5908, seconds_per_object=1.0)

    assert result["seconds"] == 5908.0
    assert round(result["minutes"], 2) == 98.47
