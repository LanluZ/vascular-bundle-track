import os

import cv2
import shutil
import pickle

import numpy as np
from thop.rnn_hooks import count_lstm

from ultralytics import YOLO
from boxmot import ByteTrack

from splicing import *
from vascular import *

detect_filename = r'D:\Users\LanluZ\Downloads\51-vapour'  # 检测视频文件或图片文件夹
rotate_angle = cv2.ROTATE_90_COUNTERCLOCKWISE  # 视频旋转角度
conf = 0.55  # 置信度


def main():
    # 目标检测
    model = YOLO("best.pt")
    # 目标跟踪
    tracker = ByteTrack()
    # 载入视频对象
    global detect_filename # 全局化
    video = cv2.VideoCapture(detect_filename) if not os.path.isdir(detect_filename) else DirVideo(detect_filename)
    # 如果是文件夹重命名detect_filename
    if os.path.isdir(detect_filename):
        detect_filename = os.path.normpath(detect_filename)
        detect_filename = detect_filename.replace('\\', '/')
        detect_filename = detect_filename.split('/')[-1]

    # 清除文件夹
    if os.path.exists('./output'):
        shutil.rmtree('./output')
    os.mkdir('./output')

    if os.path.exists('./csv'):
        shutil.rmtree('./csv')
    os.mkdir('./csv')

    i = 0
    vascular_list = dict()  # 维管束对象字典
    while True:
        ret, im = video.read()  # 帧
        if not ret:
            break
        im = cv2.rotate(im, rotate_angle)  # 旋转

        detect = model.predict(im, save=False, conf=conf, device='cuda:0')
        box_xyxy = detect[0].boxes.xyxy  # 坐标
        box_conf = detect[0].boxes.conf  # 置信度
        box_cls = detect[0].boxes.cls  # 类别
        # 格式转换
        box_xyxy = box_xyxy.detach().cpu().numpy().astype(np.int32)
        box_conf = box_conf.detach().cpu().numpy().astype(np.float32)
        box_cls = box_cls.detach().cpu().numpy().astype(np.int32)
        # 拼接
        box = np.concatenate([box_xyxy, box_conf.reshape(-1, 1), box_cls.reshape(-1, 1)], axis=1)
        tracker.update(box, im)
        # 跟踪结果记录
        for trk in tracker.active_tracks:
            if trk.is_activated:
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

    # 保存csv
    for k, v in vascular_list.items():
        v.to_csv('./csv/{}.csv'.format(k))

    video.release()
    cv2.destroyAllWindows()

    # 合成视频
    splicing_video('output', 'video/' + detect_filename, 15)


# 文件夹视频类(目的兼容cv2.VideoCapture)
class DirVideo:
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.im_filename = os.listdir(dirpath)
        self.im_len = len(self.im_filename)
        self.count = 0  # 伪帧数计数器

    # 读取帧
    def read(self):
        # 检测是否无效帧
        if self.count >= self.im_len:
            return False, None
        # 载入帧
        im = cv2.imread(os.path.join(self.dirpath, self.im_filename[self.count]))
        self.count += 1
        return True, im

    def release(self):
        pass


if __name__ == '__main__':
    main()
