import os
from ultralytics import YOLO

project_path = os.path.dirname(__file__)


def main():
    model = YOLO("yolov10m.pt")

    model.train(
        data="data.yaml",
        epochs=300,
        imgsz=640,
        degrees=90,
        device='cuda:0',
        pretrained=True,
    )


if __name__ == '__main__':
    main()
