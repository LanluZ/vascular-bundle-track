import argparse
import csv
import os
from pathlib import Path

from metrics_utils import compute_f1, save_json


def _absolute_data_path(data: str) -> str:
    """Resolve a relative --data value against the script directory (repo root)."""
    if not data or os.path.isabs(data):
        return data
    return os.path.normpath(os.path.join(Path(__file__).parent, data))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO detection metrics on the validation split.")
    parser.add_argument("--model", default="weights/best.pt", help="Weights to evaluate.")
    parser.add_argument("--data", default="data.yaml", help="YOLO data yaml.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="val_metrics")
    parser.add_argument("--output", default="reports/detection_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.val(
        data=_absolute_data_path(args.data),
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        split="val",
        plots=True,
    )
    metrics = _extract_metrics(results)
    output = Path(args.output)
    save_json(metrics, output)
    _write_csv(metrics, output.with_suffix(".csv"))
    _write_markdown(metrics, output.with_suffix(".md"), args)
    print(f"Saved detection metrics to {output}")


def _extract_metrics(results) -> dict:
    box = results.box
    precision = float(getattr(box, "mp", 0.0))
    recall = float(getattr(box, "mr", 0.0))
    map50 = float(getattr(box, "map50", 0.0))
    map50_95 = float(getattr(box, "map", 0.0))
    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "f1": compute_f1(precision, recall),
    }


def _write_csv(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def _write_markdown(metrics: dict, path: Path, args) -> None:
    lines = [
        "# Detection Metrics",
        "",
        f"- Model: `{args.model}`",
        f"- Data: `{args.data}`",
        f"- Image size: `{args.imgsz}`",
        "",
        "| Precision | Recall | mAP50 | mAP50-95 | F1 |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['map50']:.4f} | {metrics['map50_95']:.4f} | {metrics['f1']:.4f} |"
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
