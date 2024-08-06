# 帧类
class Frame:
    xyxy = []  # 坐标信息
    conf = 0  # 置信度
    cls = 0  # 类别
    time = 0  # 时间信息

    def __init__(self, xyxy, conf, cls, time):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.time = time


# 维管束类
class Vascular:
    frames = []  # 连续帧

    def __init__(self):
        pass

    # 添加帧
    def add(self, frame: Frame):
        self.frames.append(frame)
