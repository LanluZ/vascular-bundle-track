import cv2
import os


# 读取视频帧拼接视频
def splicing_video(frames_dir_path, output_file_path, fps):
    """
    :param frames_dir_path: 视频帧所在文件夹路径
    :param output_file_path: 拼接视频文件保存路径
    :param fps: 合成视频帧数
    :return:
    """
    frame_files = os.listdir(frames_dir_path)

    # 按帧合成视频
    fps = 15
    image = cv2.imread(os.path.join(frames_dir_path, frame_files[0]))
    height, width, _ = image.shape

    # 创建目标视频
    output = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (width, height))
    # 写入帧
    for frame_file in frame_files:
        image = cv2.imread(os.path.join(frames_dir_path, frame_file))
        output.write(image)

    output.release()

    print('视频拼接完成')


if __name__ == '__main__':
    splicing_video('output', 'video/3_new.mp4', 15)
