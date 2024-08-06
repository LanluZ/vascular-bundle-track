import cv2
import shutil
import pickle

import numpy as np

from ultralytics import YOLO
from boxmot import BYTETracker

from splicing import *
from vascular import *


def main():
    # 目标检测
    model = YOLO("best.pt")
    # 目标跟踪
    tracker = BYTETracker()
    # 视频对象
    video = cv2.VideoCapture('3.mp4')

    # 清除文件夹
    if os.path.exists('./output'):
        shutil.rmtree('./output')
    os.mkdir('./output')

    i = 0
    vascular_list = dict()  # 维管束对象字典
    while True:
        ret, im = video.read()  # 帧
        if not ret:
            break

        detect = model.predict(im, save=False, conf=0.6, device='cuda:0')
        box_xyxy = detect[0].boxes.xyxy  # 坐标
        box_conf = detect[0].boxes.conf  # 置信度
        box_cls = detect[0].boxes.cls  # 类别
        # 格式转换
        box_xyxy = box_xyxy.detach().cpu().numpy().astype(np.int32)
        box_conf = box_conf.detach().cpu().numpy().astype(np.float32)
        box_cls = box_cls.detach().cpu().numpy().astype(np.int32)
        # 拼接
        box = np.concatenate([box_xyxy, box_conf.reshape(-1, 1), box_cls.reshape(-1, 1)], axis=1)
        tracker_result = tracker.update(box, im)
        # 跟踪结果记录
        for trk in tracker.active_tracks:
            if trk.is_activated and trk.time_since_update < 1:
                # 帧对象
                frame = Frame(trk.xyxy, trk.conf, trk.cls, i)
                # 加入字典
                if trk.id in vascular_list:
                    vascular_list[trk.id].add(frame)
                else:
                    vascular_list[trk.id] = Vascular()
                    vascular_list[trk.id].add(frame)

        # 轨迹绘制
        tracker.plot_results(im, show_trajectories=True)
        cv2.imwrite("./output/image-{}.png".format(str(i).zfill(4)), im)
        i += 1

    # 保存pkl
    with open('3.pkl', 'wb') as f:
        pickle.dump(vascular_list, f)

    video.release()
    cv2.destroyAllWindows()

    # 合成视频
    # splicing_video('output', 'video/1.mp4', 15)


if __name__ == '__main__':
    main()
