import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from metrics_utils import save_json
from splicing import splicing_video
from vascular import Frame, Vascular


ROTATE_OPTIONS = {
    "none": None,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detection + ByteTrack tracking on a video.")
    parser.add_argument("--source", default="video/56-fire.mp4", help="Video file or image directory.")
    parser.add_argument("--model", default="weights/best.pt", help="YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.58)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rotate", choices=ROTATE_OPTIONS.keys(), default="ccw90")
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--name", default=None, help="Experiment name under runs/pipeline.")
    parser.add_argument("--project", default="runs/pipeline")
    parser.add_argument("--clean", action="store_true", help="Replace an existing experiment directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = _prepare_output_dir(args)
    frame_dir = output_dir / "frames"
    csv_dir = output_dir / "csv"
    frame_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    total_start = time.perf_counter()
    model = YOLO(args.model)
    tracker = _create_tracker()
    video = _open_source(args.source)
    output_video_name = _output_video_name(args.source)

    vascular_list: dict[int, Vascular] = {}
    mot_rows = []
    frame_index = 0
    predict_seconds = 0.0
    track_seconds = 0.0
    write_seconds = 0.0

    while True:
        ret, image = video.read()
        if not ret:
            break
        if ROTATE_OPTIONS[args.rotate] is not None:
            image = cv2.rotate(image, ROTATE_OPTIONS[args.rotate])

        start = time.perf_counter()
        detect = model.predict(image, save=False, conf=args.conf, device=args.device, verbose=False)
        predict_seconds += time.perf_counter() - start

        boxes = _detections_to_numpy(detect[0])
        start = time.perf_counter()
        tracker.update(boxes, image)
        track_seconds += time.perf_counter() - start

        for trk in tracker.active_tracks:
            if not getattr(trk, "is_activated", True):
                continue
            track_id = int(_first_attr(trk, ["id", "track_id"]))
            xyxy = _first_attr(trk, ["xyxy", "tlbr"])
            conf = _first_attr(trk, ["conf", "score"], default=0.0)
            cls = _first_attr(trk, ["cls", "cls_id"], default=0)
            frame = Frame(xyxy, conf, cls, frame_index)
            vascular_list.setdefault(track_id, Vascular()).add(frame)
            mot_rows.append(_mot_row(frame_index, track_id, xyxy, conf))

        tracker.plot_results(image, show_trajectories=True, thickness=8, fontscale=2)
        start = time.perf_counter()
        cv2.imwrite(str(frame_dir / f"image-{frame_index:04d}.png"), image)
        write_seconds += time.perf_counter() - start
        frame_index += 1

    video.release()
    cv2.destroyAllWindows()

    for track_id, vascular in vascular_list.items():
        vascular.to_csv(csv_dir / f"{track_id}.csv")
    _write_mot(output_dir / "tracks_mot.txt", mot_rows)
    if frame_index:
        splicing_video(str(frame_dir), str(output_dir / output_video_name), args.fps)

    timings.update(
        {
            "total_seconds": time.perf_counter() - total_start,
            "predict_seconds": predict_seconds,
            "track_seconds": track_seconds,
            "write_frame_seconds": write_seconds,
            "frame_count": frame_index,
            "track_count": len(vascular_list),
            "source": args.source,
            "model": args.model,
            "conf": args.conf,
        }
    )
    save_json(timings, output_dir / "timing.json")
    print(f"Saved pipeline outputs to {output_dir}")


def _prepare_output_dir(args) -> Path:
    source_name = Path(args.source).stem if args.name is None else args.name
    output_dir = Path(args.project) / source_name
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _create_tracker():
    try:
        from boxmot import ByteTrack
    except ImportError:
        try:
            from boxmot.trackers.bytetrack.bytetrack import ByteTrack
        except ImportError as exc:
            raise RuntimeError("ByteTrack requires boxmot. Install it before running detect_cam.py.") from exc
    try:
        return ByteTrack()
    except TypeError as exc:
        raise RuntimeError("ByteTrack requires boxmot. Install it before running detect_cam.py.") from exc


def _open_source(source: str):
    source_path = Path(source)
    if source_path.is_dir():
        return DirVideo(source_path)
    video = cv2.VideoCapture(str(source_path))
    if not video.isOpened():
        raise ValueError(f"Cannot open source: {source}")
    return video


def _output_video_name(source: str) -> str:
    source_path = Path(source)
    return f"{source_path.stem}_tracked.mp4"


def _detections_to_numpy(result) -> np.ndarray:
    box_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    box_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    box_cls = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
    if box_xyxy.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.concatenate([box_xyxy, box_conf.reshape(-1, 1), box_cls.reshape(-1, 1)], axis=1)


def _first_attr(obj, names: list[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not None:
        return default
    raise AttributeError(f"{type(obj).__name__} has none of these attributes: {names}")


def _mot_row(frame_index: int, track_id: int, xyxy, conf) -> str:
    xmin, ymin, xmax, ymax = [float(value) for value in xyxy]
    width = xmax - xmin
    height = ymax - ymin
    return f"{frame_index + 1},{track_id},{xmin:.2f},{ymin:.2f},{width:.2f},{height:.2f},{float(conf):.6f},-1,-1,-1"


def _write_mot(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


class DirVideo:
    def __init__(self, dirpath: Path):
        self.dirpath = dirpath
        self.image_files = sorted(
            path for path in dirpath.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        self.count = 0

    def read(self):
        if self.count >= len(self.image_files):
            return False, None
        image = cv2.imread(str(self.image_files[self.count]))
        self.count += 1
        return image is not None, image

    def release(self):
        pass


if __name__ == "__main__":
    main()
