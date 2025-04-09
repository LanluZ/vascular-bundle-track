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
    x_move_threshold = 10  # 运动阈值单位像素
    cmap = sns.color_palette("rocket_r", as_cmap=True)  # 色彩映射
    image_resize_score = 3 / 5  # 图像缩放比例

    # 总体数据存储
    x_delta_list = []
    y_delta_list = []
    x_time_start_list = []
    x_time_end_list = []
    x_time_duration_list = []
    y_time_start_list = []
    y_time_end_list = []
    y_time_duration_list = []
    x_speed_list = []
    y_speed_list = []
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
        x_time_start, y_time_start = 0, 0
        x_time_end, y_time_end = 0, 0
        x_move_threshold, y_move_threshold = x_delta * 0.1, y_delta * 0.1  # 动态阈值
        for i in range(1, df.shape[0]):
            if abs(df.iloc[i]['x_center'] - df.iloc[0]['x_center']) >= x_move_threshold:
                x_time_start = df.iloc[i]['time']
                break
        for i in range(1, df.shape[0]):  # 运动结束
            if abs(df.iloc[i]['x_center'] - df.iloc[-1]['x_center']) <= x_move_threshold:
                x_time_end = df.iloc[i]['time']
                break
        for i in range(1, df.shape[0]):  # 运动开始
            if abs(df.iloc[i]['y_center'] - df.iloc[0]['y_center']) >= y_move_threshold:
                y_time_start = df.iloc[i]['time']
                break
        for i in range(1, df.shape[0]):  # 运动结束
            if abs(df.iloc[i]['y_center'] - df.iloc[-1]['y_center']) <= y_move_threshold:
                y_time_end = df.iloc[i]['time']
                break

        x_time_duration = x_time_end - x_time_start
        y_time_duration = y_time_end - y_time_start
        x_speed = x_delta / x_time_duration if x_time_duration != 0 else 0
        y_speed = y_delta / y_time_duration if y_time_duration != 0 else 0

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
        x_time_start_list.append(x_time_start)
        x_time_end_list.append(x_time_end)
        x_time_duration_list.append(x_time_duration)
        x_speed_list.append(x_speed)
        y_time_start_list.append(y_time_start)
        y_time_end_list.append(y_time_end)
        y_time_duration_list.append(y_time_duration)
        y_speed_list.append(y_speed)

    # 维度转换
    x_delta_list = np.array(x_delta_list).reshape(-1, 1)
    y_delta_list = np.array(y_delta_list).reshape(-1, 1)
    x_time_start_list = np.array(x_time_start_list).reshape(-1, 1)
    x_time_end_list = np.array(x_time_end_list).reshape(-1, 1)
    x_time_duration_list = np.array(x_time_duration_list).reshape(-1, 1)
    x_speed_list = np.array(x_speed_list).reshape(-1, 1)
    y_time_start_list = np.array(y_time_start_list).reshape(-1, 1)
    y_time_end_list = np.array(y_time_end_list).reshape(-1, 1)
    y_time_duration_list = np.array(y_time_duration_list).reshape(-1, 1)
    y_speed_list = np.array(y_speed_list).reshape(-1, 1)

    # 绘制热力矩形
    image = draw_heatmap_rect_to_image(image, y_time_end_list, cmap, rectangle_position_list) # 绘制其他图请修改此处

    # 绘制热力条
    plt.figure(dpi=300)
    sns.heatmap(y_time_end_list, cmap=cmap) # 绘制其他图请修改此处
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
    data_norm_list = scaler.fit_transform(data_list)
    colors = cmap(data_norm_list).squeeze(1)
    colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255), alpha) for c in colors]  # 转16进制和BGR

    # 绘制矩形
    for i in range(len(data_norm_list)):
        color = colors[i]

        pt1 = rectangle_position_list[i][0]
        pt2 = rectangle_position_list[i][1]

        # 绘制矩形
        cv2.rectangle(image, pt1, pt2, color, -1)
        # 标记数值 如果需要取消注释
        # cv2.putText(image, str(data_list[i]), (pt1[0] + 5, pt1[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return image


if __name__ == '__main__':
    main()
