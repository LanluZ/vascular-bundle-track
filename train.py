import os
import argparse
from ultralytics import YOLO

project_path = os.path.dirname(__file__)


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
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    model.train(
        data=args.data,
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


if __name__ == '__main__':
    main()
