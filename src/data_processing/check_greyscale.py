import subprocess
from pathlib import Path
import cv2
import pandas as pd
import os
import numpy as np

def convert_to_mp4(input_path, output_path, data_name):
    os.environ['PATH'] += os.pathsep + r"C:\Users\um20242\ffmpeg\bin"
    if data_name == 'live_qualcomm':
        command = ['ffmpeg.exe', '-hide_banner', '-loglevel', 'error',
                   '-s', f'1920x1080', '-pix_fmt', 'yuv420p', '-framerate', '25.0',
                   '-i', str(input_path), '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', '-y',
                   str(output_path)]
    else:
        command = ['ffmpeg.exe', '-i', str(input_path), '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', '-y', str(output_path)]
    try:
        subprocess.run(command, check=True)
        print(f"success: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"error: {e}")
        return False

def is_greyscale_image(image):
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3 and image.shape[2] == 3:
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        color_threshold = 3
        if (cv2.norm(b - g, cv2.NORM_INF) <= color_threshold and
                cv2.norm(b - r, cv2.NORM_INF) <= color_threshold and
                cv2.norm(g - r, cv2.NORM_INF) <= color_threshold):
            return True
    return False

def check_video_greyscale(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return False, False

    frame_read = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_read = True
        if not is_greyscale_image(frame):
            cap.release()
            return False, True

    cap.release()
    # 如果循环结束后没有发现彩色帧，且至少读取到一帧，认为视频是灰度的
    return True if frame_read else False, frame_read

def process_videos_from_csv(metadata_path, output_csv, data_name):
    df = pd.read_csv(metadata_path)
    results = []
    for i, row in df.iterrows():
        frame_read = True
        if data_name == 'youtube_ugc':
            video_path = Path("D:/video_dataset/ugc-dataset/youtube_ugc/original_videos") / f"{row['resolution']}P" / f"{row['vid']}.mkv"
        elif data_name == 'konvid_1k':
            video_path = Path("D:/video_dataset/KoNViD_1k/KoNViD_1k_videos") / f"{row['vid']}.mp4"
        elif data_name == 'lsvq_train' or data_name == 'lsvq_test' or data_name == 'lsvq_test_1080P':
            video_path = Path("D:/video_dataset/LSVQ") / f"{row['vid']}.mp4"
        elif data_name == 'live_vqc':
            video_path = Path("D:/video_dataset/LIVE-VQC/video") / f"{row['vid']}.mp4"
        elif data_name == 'live_qualcomm':
            video_path = Path("D:/video_dataset/LIVE-Qualcomm") / f"{row['vid']}.yuv"
        elif data_name == 'cvd_2014':
            video_path = Path("D:/video_dataset/CVD2014") / f"{row['vid']}.avi"

        converted_path = video_path.with_suffix('.mp4')
        need_conversion = data_name == 'live_qualcomm' or not frame_read

        if need_conversion:
            convert_to_mp4(video_path, converted_path, data_name)
            is_greyscale, frame_read = check_video_greyscale(converted_path)
            os.remove(converted_path)
        else:
            is_greyscale, frame_read = check_video_greyscale(video_path)  # 直接检查原视频


        if is_greyscale:
            print(video_path)
            results.append({'Index': i, 'vid': row['vid'], 'Is Greyscale': is_greyscale})

    if len(results) != 0:
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False)
        print(f"results saved to {output_csv}")
    else:
        print('no greyscale video')

if __name__ == '__main__':
    data_name = 'live_qualcomm'
    metadata_path = f'../../metadata/{data_name.upper()}_metadata.csv'
    greyscale_path = f'../../metadata/greyscale_report/{data_name.upper()}_greyscale_metadata.csv'

    process_videos_from_csv(metadata_path, greyscale_path, data_name)
