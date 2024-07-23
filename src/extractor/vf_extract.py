import math
import pandas as pd
import os
import subprocess

def extract_frames_general(input_video_path, output_frame_directory, frame_interval):
    try:
        video_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_template = os.path.join(output_frame_directory, f'{video_name}_%d.png')
        os.environ['PATH'] += os.pathsep + r"C:\Users\um20242\ffmpeg\bin"

        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', input_video_path,
            '-vf', f"select='not(mod(n,{frame_interval}))'",  # f'select=not(mod(n\,{frame_interval}))',
            '-vsync', 'vfr',
            output_template
        ]
        print(ffmpeg_cmd)
        subprocess.run(ffmpeg_cmd, check=True)
        print(f'Extraction of video frames for {video_name} completed!')
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running ffmpeg: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def extract_frames_yuv(input_video_path, output_frame_directory, frame_interval, video_width, video_height, pixfmt, framerate):
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_template = os.path.join(output_frame_directory, f'{video_name}_%d.png')
    os.environ['PATH'] += os.pathsep + r"C:\Users\um20242\ffmpeg\bin"

    # ffmpeg
    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner',  # hide terminal info
        '-loglevel', 'error',
        '-s', f'{video_width}x{video_height}',
        '-pix_fmt', f'{pixfmt}',
        '-framerate', f'{framerate}',
        '-i', input_video_path,
        '-vf', f"select='not(mod(n,{frame_interval}))'",
        '-vsync', 'vfr',
        output_template
    ]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print(f'Extraction of video frames for {video_name} completed!')

def extract_frames_residual(video_path, sampled_path, frame_interval):
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.environ['PATH'] += os.pathsep + r"C:\Users\um20242\ffmpeg\bin"

        # extract frames based on frame_interval
        extract_frames_general(video_path, sampled_path, frame_interval)

        # extract the next frame following
        output_template_next = os.path.join(sampled_path, f'{video_name}_%d_next.png')
        ffmpeg_cmd_next = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', video_path,
            '-vf', f"select='not(mod(n-1,{frame_interval}))'", '-vsync', 'vfr',
            output_template_next
        ]
        print(ffmpeg_cmd_next)
        subprocess.run(ffmpeg_cmd_next, check=True)
        print(f'Extraction of next frames for {video_name} completed!')

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running ffmpeg: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def extract_frames_residual_yuv(video_path, sampled_path, frame_interval, video_width, video_height, pixfmt, framerate):
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.environ['PATH'] += os.pathsep + r"C:\Users\um20242\ffmpeg\bin"

        # extract frames based on frame_interval
        extract_frames_yuv(video_path, sampled_path, frame_interval, video_width, video_height, pixfmt, framerate)

        # extract the next frame following
        output_template_next = os.path.join(sampled_path, f'{video_name}_%d_next.png')
        ffmpeg_cmd_next = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-s', f'{video_width}x{video_height}', '-pix_fmt', f'{pixfmt}', '-framerate', f'{framerate}',
            '-i', video_path,
            '-vf', f"select='not(mod(n-1,{frame_interval}))'", '-vsync', 'vfr',
            output_template_next
        ]
        print(ffmpeg_cmd_next)
        subprocess.run(ffmpeg_cmd_next, check=True)
        print(f'Extraction of next frames for {video_name} completed!')

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running ffmpeg: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def process_video(video_type, video_name, frame_interval, video_path, sampled_path, video_width, video_height, pixfmt, framerate):
    if not os.path.exists(sampled_path):
        os.makedirs(sampled_path)

    print(f'Processing {video_type} video: {video_name}')
    if video_type == 'live_qualcomm':
        extract_frames_yuv(video_path, sampled_path, frame_interval, video_width, video_height, pixfmt, framerate)
    else:
        extract_frames_general(video_path, sampled_path, frame_interval)

def process_video_residual(video_type, video_name, frame_interval, video_path, sampled_path, video_width, video_height, pixfmt, framerate):
    if not os.path.exists(sampled_path):
        os.makedirs(sampled_path)

    print(f'Processing {video_type} video: {video_name}')
    if video_type == 'live_qualcomm':
        extract_frames_residual_yuv(video_path, sampled_path, frame_interval, video_width, video_height, pixfmt, framerate)
    else:
        extract_frames_residual(video_path, sampled_path, frame_interval)


if __name__ == '__main__':
    video_type = 'original_ugc_test'
    
    # for original video:
    if video_type == 'original_ugc_test':
        ugcdata = pd.read_csv(f"../metadata/YOUTUBE_UGC_metadata_original.csv")

    for i in range(len(ugcdata)):
        video_name = ugcdata['vid'][i]
        video_width = ugcdata['width'][i]
        video_height = ugcdata['height'][i]
        pixfmt = ugcdata['pixfmt'][i]
        framerate = ugcdata['framerate'][i]

        print(f'framerate: {framerate}')
        if framerate < 2:
            frame_interval = math.ceil(framerate / 2)
        else:
            frame_interval = int(framerate / 2)
        print(f'frame_interval: {frame_interval}')

        if video_type == 'original_ugc_test':
            qp = 'original'
            video_path = f"../ugc_original_videos/{video_name}.mkv"
            sampled_path = f'../video_sampled_frame/original_sampled_frame/{video_name}/'

        print(f'{video_name} at qp {qp}')
        # process_video(video_type, video_name, frame_interval, video_path, sampled_path, video_width, video_height, pixfmt, framerate)
        process_video_residual(video_type, video_name, frame_interval, video_path, sampled_path, video_width, video_height, pixfmt, framerate)


