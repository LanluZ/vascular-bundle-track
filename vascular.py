import csv


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
        self.frames = []  # 分配新指针

    # 添加帧
    def add(self, frame: Frame):
        self.frames.append(frame)

    # 保存为csv文件
    def to_csv(self, csv_path):
        columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'cls', 'time']
        with open(csv_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            for frame in self.frames:
                writer.writerow([
                    int(frame.xyxy[0]),
                    int(frame.xyxy[1]),
                    int(frame.xyxy[2]),
                    int(frame.xyxy[3]),
                    float(frame.conf),
                    int(frame.cls),
                    int(frame.time),
                ])
