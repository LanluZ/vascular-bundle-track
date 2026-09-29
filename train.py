import os
import argparse
from ultralytics import YOLO

project_path = os.path.dirname(__file__)


def _absolute_data_path(data: str) -> str:
    """Resolve a relative --data value against the script directory (repo root),
    so a portable data.yaml resolves correctly regardless of cwd."""
    if not data or os.path.isabs(data):
        return data
    return os.path.normpath(os.path.join(project_path, data))


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv10 for bamboo vascular bundle detection.")
    parser.add_argument("--model", default="weights/yolov10m.pt", help="Initial model weights.")
    parser.add_argument("--data", default="data.yaml", help="YOLO data yaml.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--degrees", type=float, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--name", default="bamboo_yolov10")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--optimizer", default=None, help="Optimizer name (passed to model.train).")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (passed to model.train).")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (passes amp=False to model.train).")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    train_kwargs = dict(
        data=_absolute_data_path(args.data),
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        imgsz=args.imgsz,
        degrees=args.degrees,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
    )
    if args.optimizer is not None:
        train_kwargs["optimizer"] = args.optimizer
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.no_amp:
        train_kwargs["amp"] = False

    model.train(**train_kwargs)


if __name__ == '__main__':
    main()
