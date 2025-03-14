import math
import pandas as pd
import cv2
import os

def extract_frames(video_path, sampled_path, frame_interval, residual=False):
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return frames

        frame_count = 0
        saved_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_count % frame_interval == 0 and not residual) or (
                    (frame_count - 1) % frame_interval == 0 and residual):
                # suffix = '_next' if residual else ''
                # output_filename = os.path.join(sampled_path, f'{video_name}_{saved_frame_count + 1}{suffix}.png')
                # cv2.imwrite(output_filename, frame)
                frames.append(frame)
                saved_frame_count += 1

            frame_count += 1
        cap.release()
        frame_type = 'next frames' if residual else 'sampled frames'
        print(f'Extraction of {frame_type} for {video_name} completed!')
        return frames
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def process_video_residual(video_type, video_name, framerate, video_path, sampled_path):
    if not os.path.exists(sampled_path):
        os.makedirs(sampled_path)
    # cap = cv2.VideoCapture(video_path)
    # framerate = cap.get(cv2.CAP_PROP_FPS)
    # print(f'framerate: {framerate}')
    frame_interval = math.ceil(framerate / 2) if framerate < 2 else int(framerate / 2)
    # print(f'Frame interval: {frame_interval}')

    frames = extract_frames(video_path, sampled_path, frame_interval, residual=False)
    frames_next = extract_frames(video_path, sampled_path, frame_interval, residual=True)
    return frames, frames_next


if __name__ == '__main__':
    video_type = 'test'

    if video_type == 'test':
        ugcdata = pd.read_csv("../../metadata/test_videos.csv")

    for i in range(len(ugcdata)):
        video_name = ugcdata['vid'][i]
        framerate = ugcdata['framerate'][i]
        print(f'Processing video: {video_name}, framerate: {framerate}')

        video_path = f"../../ugc_original_videos/{video_name}.mp4"
        sampled_path = f'../../video_sampled_frame/original_sampled_frame/{video_name}/'

        if not os.path.exists(sampled_path):
            os.makedirs(sampled_path)

        print(f'{video_name}')
        frames, frames_next = process_video_residual(video_type, video_name, framerate, video_path, sampled_path)
        print(f'Extracted {len(frames)} frames and {len(frames_next)} residual frames for video: {video_name}')