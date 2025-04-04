import os
import cv2

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


def main():
    # 读取图像
    image = cv2.imread('./temp/0.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    csv_dir_path = './csv/'

    # 参数设定
    move_threshold = 10  # 运动阈值单位像素
    cmap = sns.color_palette("rocket_r", as_cmap=True)  # 色彩映射
    image_resize_score = 3 / 5  # 图像缩放比例

    # 总体数据存储
    x_delta_list = []
    y_delta_list = []
    time_start_list = []
    time_end_list = []
    time_duration_list = []
    rectangle_position_list = []
    id_list = []

    # 批量读取csv
    for csv_path in os.listdir(csv_dir_path):
        if not csv_path.endswith('.csv'):
            continue

        # 读取csv
        df = pd.read_csv(os.path.join(csv_dir_path, csv_path), header=0)

        # 获取总位移数据
        x_delta = abs(df.iloc[-1]['x_center'] - df.iloc[0]['x_center'])
        y_delta = abs(df.iloc[-1]['y_center'] - df.iloc[0]['y_center'])

        # 获取运动时间
        time_start = 0
        time_end = 0
        move_threshold = y_delta * 0.1  # 动态阈值
        for i in range(1, df.shape[0]):
            # 运动开始
            if abs(df.iloc[i]['y_center'] - df.iloc[0]['y_center']) >= move_threshold:
                time_start = df.iloc[i]['time']
                break
        for i in range(1, df.shape[0]):
            # 运动结束
            if abs(df.iloc[i]['y_center'] - df.iloc[-1]['y_center']) <= move_threshold:
                time_end = df.iloc[i]['time']
                break

        time_duration = time_end - time_start

        # 获取绘制矩形坐标
        xmin = int(df.iloc[0]['xmin'])
        ymin = int(df.iloc[0]['ymin'])
        xmax = int(df.iloc[0]['xmax'])
        ymax = int(df.iloc[0]['ymax'])
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        rectangle_position = (pt1, pt2)

        # 保存数据
        id_list.append(csv_path.split('.')[0])
        rectangle_position_list.append(rectangle_position)
        x_delta_list.append(x_delta)
        y_delta_list.append(y_delta)
        time_start_list.append(time_start)
        time_end_list.append(time_end)
        time_duration_list.append(time_duration)

    # 维度转换
    x_delta_list = np.array(x_delta_list).reshape(-1, 1)
    y_delta_list = np.array(y_delta_list).reshape(-1, 1)
    time_start_list = np.array(time_start_list).reshape(-1, 1)
    time_end_list = np.array(time_end_list).reshape(-1, 1)
    time_duration_list = np.array(time_duration_list).reshape(-1, 1)

    # 绘制热力矩形
    image = draw_heatmap_rect_to_image(image, time_duration_list, cmap, rectangle_position_list)

    # 绘制热力条
    plt.figure()
    sns.heatmap(time_duration_list, cmap=cmap)
    plt.savefig('./colorbar.png')

    # 读取热力条并且绘制白色
    colorbar = cv2.imread('./colorbar.png', cv2.IMREAD_UNCHANGED)
    # 绘制清除左侧区域49/64是计算得出区域
    colorbar = cv2.rectangle(colorbar,
                             (0, 0), (colorbar.shape[1] * 49 // 64, colorbar.shape[0]),
                             (255, 255, 255, 255), -1)
    # 缩放叠图
    resize_image = cv2.resize(
        image,
        dsize=(int(colorbar.shape[1] * image_resize_score), int(colorbar.shape[0] * image_resize_score))
    )
    x, y = (colorbar.shape[1] * 15 // 64) // 2, colorbar.shape[0] // 2 - resize_image.shape[0] // 2
    colorbar[y:y + resize_image.shape[0], x:x + resize_image.shape[1]] = resize_image

    # 保存图像
    cv2.imwrite('./heatmap.png', image)
    cv2.imwrite('./heatmap_with_colorbar.png', colorbar)


# 绘制热力矩形
def draw_heatmap_rect_to_image(image, data_list, cmap, rectangle_position_list):
    # 色彩映射
    alpha = 255
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_list = scaler.fit_transform(data_list)
    colors = cmap(data_list).squeeze(1)
    colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255), alpha) for c in colors]  # 转16进制和BGR

    # 绘制矩形
    for i in range(len(data_list)):
        color = colors[i]

        pt1 = rectangle_position_list[i][0]
        pt2 = rectangle_position_list[i][1]

        cv2.rectangle(image, pt1, pt2, color, -1)  # 绘制矩形

    return image


if __name__ == '__main__':
    main()
