import os
import cv2
import shutil
import pandas as pd
from tqdm import tqdm


def main():
    path = 'csv'
    img_path = 'temp'
    selected_path = 'selected'

    # 清空文件夹
    for folder in [selected_path, img_path]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # 切割视频
    split_video('video/52-control.mp4', 'temp')

    # 预先创建所有需要的目录
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    for file in files:
        os.makedirs(os.path.join(selected_path, file), exist_ok=True)

    # 创建图像缓存字典，避免重复读取相同的图像
    image_cache = {}
    
    # 处理CSV文件
    with tqdm(total=len(files), desc='裁剪中') as pbar:
        for file in files:
            pbar.update(1)
            if not file.endswith('.csv'):
                continue
                
            # 读取CSV文件
            df = pd.read_csv(os.path.join(path, file), header=0)
            
            # 批量处理图像裁剪
            process_crops(df, img_path, selected_path, file, image_cache)
    
    # 清理图像缓存
    image_cache.clear()

    # 文件夹重命名
    dirs = [d for d in os.listdir(selected_path) if d.endswith('.csv')]
    for dir_name in dirs:
        old_path = os.path.join(selected_path, dir_name)
        new_path = os.path.join(selected_path, dir_name.split('.')[0])
        os.rename(old_path, new_path)


def process_crops(df, img_path, selected_path, file, image_cache):
    """批量处理图像裁剪"""
    # 按时间帧分组，减少重复读取
    time_groups = {}
    for i in range(df.shape[0]):
        time = int(df.loc[i, 'time'])
        xmin = int(df.loc[i, 'xmin'])
        ymin = int(df.loc[i, 'ymin'])
        xmax = int(df.loc[i, 'xmax'])
        ymax = int(df.loc[i, 'ymax'])
        
        if time not in time_groups:
            time_groups[time] = []
        time_groups[time].append((xmin, ymin, xmax, ymax, i))
    
    # 处理每个时间帧
    for time, crops in time_groups.items():
        # 检查缓存中是否有图像，没有则读取
        if time not in image_cache:
            img_path_full = os.path.join(img_path, f"{time}.png")
            image_cache[time] = cv2.imread(img_path_full)
            
            # 如果图像读取失败，跳过此时间帧
            if image_cache[time] is None:
                continue
        
        img = image_cache[time]
        
        # 处理此时间帧的所有裁剪
        for xmin, ymin, xmax, ymax, _ in crops:
            crop = img[ymin:ymax, xmin:xmax].copy()
            output_path = os.path.join(selected_path, file, f"{time}.png")
            cv2.imwrite(output_path, crop)


# 按帧切割视频
def split_video(video_path, frames_save_dir_path):
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    os.makedirs(frames_save_dir_path, exist_ok=True)
    
    # 获取视频总帧数用于进度条
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 批量处理帧
    batch_size = 100  # 每批处理的帧数
    i = 0
    
    with tqdm(total=total_frames, desc='提取视频帧') as pbar:
        while True:
            frames_batch = []
            frame_indices = []
            
            # 读取一批帧
            for _ in range(batch_size):
                ret, frame = video.read()
                if not ret:
                    break
                frames_batch.append(frame)
                frame_indices.append(i)
                i += 1
            
            if not frames_batch:
                break
            
            # 批量保存帧
            for idx, frame in zip(frame_indices, frames_batch):
                cv2.imwrite(os.path.join(frames_save_dir_path, f"{idx}.png"), frame)
                pbar.update(1)
    
    video.release()


if __name__ == '__main__':
    main()