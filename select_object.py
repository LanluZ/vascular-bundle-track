import os
import cv2
import pandas as pd


def main():
    path = 'csv'
    img_path = 'temp'
    selected_path = 'selected'

    # 切割视频
    split_video('4_new.mp4', 'temp')

    files = os.listdir(path)
    for file in files:
        if file.endswith('.csv'):
            # 创建文件夹
            if not os.path.exists(os.path.join(selected_path, file)):
                os.mkdir(os.path.join(selected_path, file))

            df = pd.read_csv(os.path.join(path, file), header=0)

            # 遍历时间帧
            for i in range(df.shape[0]):
                time = int(df.loc[i, 'time'])
                xmin = int(df.loc[i, 'xmin'])
                ymin = int(df.loc[i, 'ymin'])
                xmax = int(df.loc[i, 'xmax'])
                ymax = int(df.loc[i, 'ymax'])

                # 裁剪图片
                img = cv2.imread(os.path.join(img_path, str(time) + '.png'))
                img = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(selected_path, file, str(time) + '.png'), img)

    # 文件夹重命名
    dirs = os.listdir(selected_path)

    for dir in dirs:
        if dir.endswith('.csv'):
            os.rename(os.path.join(selected_path, dir), os.path.join(selected_path, dir.split('.')[0]))


# 按帧切割视频
def split_video(video_path, frames_save_dir_path):
    video = cv2.VideoCapture(video_path)

    if not os.path.exists(frames_save_dir_path):
        os.mkdir(frames_save_dir_path)

    i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            video.release()
            break
        cv2.imwrite(os.path.join(frames_save_dir_path, str(i) + '.png'), frame)
        i += 1


if __name__ == '__main__':
    main()
