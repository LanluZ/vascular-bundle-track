import pandas as pd


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

    # 保存为csv文件
    def to_csv(self, csv_path):
        columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'cls', 'time']
        df = pd.DataFrame(columns=columns)

        df['xmin'] = df['xmin'].astype(int)
        df['ymin'] = df['ymin'].astype(int)
        df['xmax'] = df['xmax'].astype(int)
        df['ymax'] = df['ymax'].astype(int)
        df['conf'] = df['conf'].astype(float)
        df['cls'] = df['cls'].astype(int)
        df['time'] = df['time'].astype(int)

        for i, frame in enumerate(self.frames):
            df.loc[i,'xmin'] = frame.xyxy[0]
            df.loc[i,'ymin'] = frame.xyxy[1]
            df.loc[i,'xmax'] = frame.xyxy[2]
            df.loc[i,'ymax'] = frame.xyxy[3]
            df.loc[i,'conf'] = frame.conf
            df.loc[i,'cls'] = frame.cls
            df.loc[i,'time'] = frame.time

        df.to_csv(csv_path)
