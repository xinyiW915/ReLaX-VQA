import pandas as pd
import subprocess
import json
import os
from scipy.io import loadmat

def parse_framerate(framerate_str):
    num, den = framerate_str.split('/')
    framerate = float(num)/float(den)
    return framerate

def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"error: {e}")

def convert_yuv_to_mp4(yuv_file, output_mp4_file, resolution, pixel_format):
    ffmpeg_path = "C://Users//um20242//ffmpeg//bin//ffmpeg.exe"
    cmd = f"{ffmpeg_path} -y -s {resolution} -pix_fmt {pixel_format} -i {yuv_file} -c:v libx264 {output_mp4_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"error: {e}")
        return False
    return True

def get_video_metadata(video_file):
    print(video_file)
    ffprobe_path = "C://Users//um20242//ffmpeg//bin//ffprobe.exe"
    cmd = f'{ffprobe_path} -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate,bit_rate,bits_per_raw_sample,pix_fmt -of json {video_file}'

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, check=True)
        info = json.loads(result.stdout)
    except Exception as e:
        print(f"Error processing file {video_file}: {e}")
        return {}

    # get metadata using ffmpeg
    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    nb_frames = info['streams'][0].get('nb_frames', 'N/A')  # Number of frames might not be available for some formats
    pixfmt = info['streams'][0]['pix_fmt']
    framerate = info['streams'][0]['r_frame_rate']
    framerate = parse_framerate(framerate)
    bitdepth = info['streams'][0].get('bits_per_raw_sample', 'N/A')
    bitrate = info['streams'][0].get('bit_rate', 'N/A')
    print(framerate)

    return width, height, nb_frames, pixfmt, framerate, bitdepth, bitrate

def create_dataframe(vid_list, mos_list, width_list, height_list, pixfmt_list, framerate_list, nb_frames_list, bitdepth_list, bitrate_list):
    data = {
        'vid': vid_list,
        'mos': mos_list,
        'width': width_list,
        'height': height_list,
        'pixfmt': pixfmt_list,
        'framerate': framerate_list,
        'nb_frames': nb_frames_list,
        'bitdepth': bitdepth_list,
        'bitrate': bitrate_list
    }
    df_new = pd.DataFrame(data)
    return df_new

def extract_csv2metadata(df, video_type):
    vid_list = []
    mos_list = []
    width_list = []
    height_list = []
    nb_frames_list = []
    pixfmt_list = []
    framerate_list = []
    bitdepth_list = []
    bitrate_list = []
    if video_type == 'lsvq':
        for i in range(len(df)):
            video_path = f"D://video_dataset//LSVQ//{df['name'][i]}.mp4"
            if os.path.exists(video_path):
                vid_list.append(df['name'][i])
                mos_list.append(df['mos'][i])
                width_list.append(df['width'][i])
                height_list.append(df['height'][i])
                nb_frames_list.append(df['frame_number'][i])

                _, _, _, pixfmt, framerate, bitdepth, bitrate = get_video_metadata(video_path)
                pixfmt_list.append(pixfmt)
                framerate_list.append(framerate)
                bitdepth_list.append(bitdepth)
                bitrate_list.append(bitrate)
            else:
                pass

    elif video_type == 'live_vqc':
        vid_list = df['vid'].tolist()
        vid_list = [vid.replace('.mp4', '') for vid in vid_list]
        mos_list = df['mos'].tolist()
        width_list = df['width'].tolist()
        height_list = df['height'].tolist()
        pixfmt_list = df['pixfmt'].tolist()
        framerate_list = df['framerate'].tolist()
        nb_frames_list = df['nb_frames'].tolist()
        bitdepth_list = df['bitdepth'].tolist()
        bitrate_list = df['bitrate'].tolist()

    df_new = create_dataframe(vid_list, mos_list, width_list, height_list, pixfmt_list, framerate_list, nb_frames_list, bitdepth_list, bitrate_list)
    return df_new

def extract_mat2metadata(mat_file, video_type):
    data = loadmat(mat_file)
    all_variables = {}
    for key, value in data.items():
        # ignore '__'
        if not key.startswith('__') and not key.endswith('__'):
            all_variables[key] = value

    vid_list = []
    mos_list = []
    width_list = []
    height_list = []
    nb_frames_list = []
    pixfmt_list = []
    framerate_list = []
    bitdepth_list = []
    bitrate_list = []
    for i in range(len(all_variables['video_names'])):
        vid = all_variables['video_names'][i].flatten()[0].item()
        mos = all_variables['scores'][i].flatten()[0].item()

        if video_type == 'cvd_2014':
            video_name = vid.replace('.avi', '')
            video_path = f"D://video_dataset//CVD2014//{video_name}.avi"
        elif video_type == 'live_qualcomm':
            video_name = vid.replace('.yuv', '')
            tmp_yuv_file = f"D://video_dataset//LIVE-Qualcomm//{video_name}.yuv"
            video_path = f"D://video_dataset//LIVE-Qualcomm//{video_name}.mp4"
            convert_yuv_to_mp4(tmp_yuv_file, video_path, "1920x1080", "yuv420p")

        width, height, nb_frames, pixfmt, framerate, bitdepth, bitrate = get_video_metadata(video_path)
        vid_list.append(video_name)
        mos_list.append(mos)
        width_list.append(width)
        height_list.append(height)
        nb_frames_list.append(nb_frames)
        pixfmt_list.append(pixfmt)
        framerate_list.append(framerate)
        bitdepth_list.append(bitdepth)
        bitrate_list.append(bitrate)
        if video_type == 'live_qualcomm':
            delete_file(video_path)

    df_new = create_dataframe(vid_list, mos_list, width_list, height_list, pixfmt_list, framerate_list, nb_frames_list, bitdepth_list, bitrate_list)
    return df_new

def save_to_csv(dataframe, output_path):
    dataframe.to_csv(output_path, index=False)

if __name__ == '__main__':

    video_type = 'live_vqc'
    print(video_type)

    # LSVQ
    if video_type == 'lsvq':
        set_name = 'train' #train, test, test_1080P
        df = pd.read_csv(f"D://video_dataset//LSVQ//LSVQ_whole_{set_name}.csv")
        df_new = extract_csv2metadata(df, video_type)
        print(df_new)
        video_type = f'LSVQ_{set_name.upper()}'

    # LIVE_VQC
    elif video_type == 'live_vqc':
        df = pd.read_csv(f"D://video_dataset//LIVE-VQC//LIVE_VQC_metadata.csv")
        df_new = extract_csv2metadata(df, video_type)
        print(df_new)

    # CVD2014
    elif video_type == 'cvd_2014':
        mat_file = "D://video_dataset//CVD2014//CVD2014info.mat"
        df_new = extract_mat2metadata(mat_file, video_type)
        print(df_new)

    # LIVE-Qualcomm
    elif video_type == 'live_qualcomm':
        mat_file = "D://video_dataset//LIVE-Qualcomm//LIVE-Qualcomminfo.mat"
        df_new = extract_mat2metadata(mat_file, video_type)
        print(df_new)

    output_csv_path = f'../../metadata/{video_type.upper()}_metadata.csv'
    save_to_csv(df_new, output_csv_path)


