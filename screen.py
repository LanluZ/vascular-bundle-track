import os
import shutil

import pandas as pd


def screen_out():
    csv_dir_path = 'csv'
    files = os.listdir(csv_dir_path)

    # 创建文件夹
    if not os.path.exists(os.path.join(csv_dir_path, 'selected')):
        os.mkdir(os.path.join(csv_dir_path, 'selected'))

    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_dir_path, file), header=0)

            # 排除过短数据
            if df.shape[0] < 130:
                shutil.move(os.path.join(csv_dir_path, file), os.path.join(csv_dir_path, 'selected', file))
                continue

            # 计算矩形面积
            w = df.loc[:, 'xmax'] - df.loc[:, 'xmin']
            h = df.loc[:, 'ymax'] - df.loc[:, 'ymin']
            df.loc[:, 'area'] = w * h

            # 计算中心点
            x_center = (df.loc[:, 'xmin'] + df.loc[:, 'xmax']) / 2
            y_center = (df.loc[:, 'ymin'] + df.loc[:, 'ymax']) / 2
            df.loc[:, 'x_center'] = x_center
            df.loc[:, 'y_center'] = y_center

            # 保存
            df.to_csv(os.path.join(csv_dir_path, file), index=False)


if __name__ == '__main__':
    screen_out()
