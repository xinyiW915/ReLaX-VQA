from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
import shutil
import torch
import math
import time
import cv2

from utils.logger_setup import logger
import video_frames_extract
from extractor import visualise_vgg_layer, visualise_resnet_layer, visualise_vit_layer


def load_metadata(video_type, compressed_type, codec_name=None):
    print(f'video_type: {video_type}\n')
    if 'original' in compressed_type:
        # Test
        if video_type == 'original_ugc':
            return pd.read_csv("../metadata/YOUTUBE_UGC_metadata_original.csv")
        # NR: original
        elif video_type == 'resolution_ugc':
            resolution = '2160P'
            return pd.read_csv(f"../metadata/YOUTUBE_UGC_{resolution}_metadata.csv")
        else:
            # return pd.read_csv(f'../metadata/{video_type.upper()}_metadata_residual.csv')
            return pd.read_csv(f'../metadata/{video_type.upper()}_metadata.csv')

    else:
        raise ValueError("Invalid video_type")

def get_video_paths(network_name, video_type, compressed_type, videodata, i):
    video_name = videodata['vid'][i]
    video_width = videodata['width'][i]
    video_height = videodata['height'][i]
    pixfmt = videodata['pixfmt'][i]
    framerate = videodata['framerate'][i]
    common_path = os.path.join('..', 'video_sampled_frame')

    if compressed_type == 'original':
        qp = 'original'
        # Test
        if video_type == 'original_ugc':
            if video_name == '5636101558':
                video_path = f"../ugc_original_videos/{video_name}.mp4"
            else:
                video_path = f"../ugc_original_videos/{video_name}.mkv"
        # NR: original
        elif video_type == 'konvid_1k':
            video_path = Path("D:/video_dataset/KoNViD_1k/KoNViD_1k_videos") / f"{video_name}.mp4"
        elif video_type == 'lsvq_train' or video_type == 'lsvq_test' or video_type == 'lsvq_test_1080P':
            print(f'video_name: {video_name}')
            video_path = Path("D:/video_dataset/LSVQ") / f"{video_name}.mp4"
            print(f'video_path: {video_path}')
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        elif video_type == 'live_vqc':
            video_path = Path("D:/video_dataset/LIVE-VQC/video") / f"{video_name}.mp4"
        elif video_type == 'live_qualcomm':
            video_path = Path("D:/video_dataset/LIVE-Qualcomm") / f"{video_name}.yuv"
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        elif video_type == 'cvd_2014':
            video_path = Path("D:/video_dataset/CVD2014") / f"{video_name}.avi"
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        sampled_frame_path = os.path.join(common_path, f'residual', f'video_{str(i + 1)}')
        feature_name = f"{network_name}_feature_map_original"

    elif 'original_' in compressed_type:
        qp = 'original'
        # NR: original
        if video_type == 'resolution_ugc':
            resolution = '2160P'
            # video_path = f'/user/work/um20242/dataset/ugc-dataset/{resolution}/{video_name}.mkv'
            video_path = Path(f"D:/video_dataset/ugc-dataset/youtube_ugc/original_videos/{resolution}") / f"{video_name}.mkv"
            sampled_frame_path = os.path.join(common_path, f'original_sampled_frame_{resolution}', f'video_{str(i + 1)}')
            feature_name = f"{network_name}_feature_map_original_{resolution}"

    return video_name, qp, video_path, sampled_frame_path, feature_name, video_width, video_height, pixfmt, framerate


def get_deep_feature(network_name, video_name, image_path, qp, layer_name):
    png_path = f'../visualisation/{network_name}/{video_name}/'
    os.makedirs(png_path, exist_ok=True)
    npy_path = f'../features/{network_name}/{video_name}/'
    os.makedirs(npy_path, exist_ok=True)

    if network_name == 'resnet50':
        if layer_name == 'last_layer':
            visual_layer = 'resnet50.layer4[2]' # last layer
        elif layer_name == 'pool':
            visual_layer = 'resnet50.avgpool' # before avg_pool
        frame_npy, _ = visualise_resnet_layer.process_video_frame(video_name, image_path, visual_layer, qp)

    elif network_name == 'vgg16':
        if layer_name == 'last_layer':
            visual_layer = 28 # last layer
        elif layer_name == 'pool':
            # visual_layer = 'fc1'
            visual_layer = 'fc2' # fc1 = vgg16.classifier[0], fc2 = vgg16.classifier[3]
        frame_npy, _ = visualise_vgg_layer.process_video_frame(video_name, image_path, visual_layer, qp)

    elif network_name == 'vit':
        # torch.cuda.set_device(torch.cuda.current_device()) # for gpu
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)

        name_model = 'vit_base'
        patch_size = 16
        model = visualise_vit_layer.VitGenerator(name_model, patch_size, device, evaluate=True, random=False, verbose=True)
        frame_npy, _ = visualise_vit_layer.process_video_frame(image_path, video_name, qp, model, patch_size, device)

    return png_path, npy_path, frame_npy


def process_video_feature(video_feature, network_name):
    print(f'video frame number: {len(video_feature)}')

    # initialize an empty list to store processed frames
    averaged_frames = []

    # iterate through each frame in the video_feature
    for frame in video_feature:
        frame_features = []

        if network_name == 'vit':
            # global mean and std
            global_mean = np.mean(frame, axis=0)
            global_max = np.max(frame, axis=0)
            global_std = np.std(frame, axis=0)

            # concatenate all pooling
            combined_features = np.hstack([global_mean, global_max, global_std])
            frame_features.append(combined_features)
        else:
            frame = np.squeeze(frame)
            # global mean and std
            global_mean = np.mean(frame, axis=0)
            global_max = np.max(frame, axis=0)
            global_std = np.std(frame, axis=0)
            # concatenate all pooling
            combined_features = np.hstack([frame, global_mean, global_max, global_std])
            frame_features.append(combined_features)

        # concatenate the layer means horizontally to form the processed frame
        processed_frame = np.hstack(frame_features)
        averaged_frames.append(processed_frame)

    averaged_frames = np.array(averaged_frames)

    # output the shape of the resulting feature vector
    logger.debug(f"Shape of feature vector after global pooling: {averaged_frames.shape}")

    return averaged_frames

def flow_to_rgb(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # convert angle to hue
    hue = ang * 180 / np.pi / 2

    # create HSV
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

if __name__ == '__main__':
    compressed_type = 'original' # original, original_resolution
    video_type = 'konvid_1k'  # original_ugc (test)
                              # resolution_ugc/konvid_1k/live_vqc/cvd_2014/live_qualcomm
                              # lsvq_train/lsvq_test/lsvq_test_1080P/

    network_name = 'resnet50'
    residual_name = 'optical_flow' # 'frame_diff', 'optical_flow'

    if network_name == 'vit':
        layer_name = 'pool'
    else:
        # layer_name = 'last_layer'
        layer_name = 'pool'

    logger.info(f"compressed_type: {compressed_type}, video_type: {video_type}, network_name: {network_name}, layer_name: {layer_name}")
    logger.info(f"torch cuda: {torch.cuda.is_available()}")

    videodata = load_metadata(video_type, compressed_type)

    valid_video_types = ['original_ugc',
                         'resolution_ugc', 'konvid_1k', 'live_vqc', 'cvd_2014', 'live_qualcomm',
                         'lsvq_train', 'lsvq_test', 'lsvq_test_1080P']

    if video_type in valid_video_types:
        for i in range(len(videodata)):
            video_name, qp, video_path, sampled_frame_path, feature_name, \
            video_width, video_height, pixfmt, framerate = get_video_paths(network_name, video_type, compressed_type, videodata, i)

            print(f'framerate: {framerate}')
            if framerate < 2:
                frame_interval = math.ceil(framerate / 2)
            else:
                frame_interval = int(framerate/2)
            print(f'frame_interval: {frame_interval}')
            start_time = time.time()
            video_frames_extract.process_video_residual(video_type, video_name, frame_interval, video_path, sampled_frame_path, video_width, video_height, pixfmt, framerate)

            logger.info(f'{video_name} at qp {qp}')
            original_frame_paths = sorted([path for path in glob.glob(os.path.join(sampled_frame_path, f'{video_name}_*.png'))
                 if '_next' not in os.path.basename(path)],
                key=lambda x: int(x.split('_')[-1].split('.')[0]))
            next_frame_paths = sorted([path for path in glob.glob(os.path.join(sampled_frame_path, f'{video_name}_*_next.png'))
                 if '_next' in os.path.basename(path)],
                key=lambda x: int(x.split('_')[-2]))

            # residual feature
            all_frame_activations_residual = []
            for original_path, next_path in zip(original_frame_paths, next_frame_paths):
                # compute residual
                img_original = cv2.imread(original_path)
                img_next = cv2.imread(next_path)

                if residual_name == 'frame_diff':
                    # option1: Frame Differencing
                    residual = cv2.absdiff(img_next, img_original)
                    residual_path = original_path.replace('.png', '_residual.png')
                    cv2.imwrite(residual_path, residual)
                    png_path, npy_path, residual_npy = get_deep_feature(network_name, video_name, residual_path, qp, layer_name)

                elif residual_name == 'optical_flow':
                    # option2: Optical Flow
                    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY),
                                                        cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY),
                                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # calculation of moving residuals
                    residual_of_rgb = flow_to_rgb(flow)
                    residual_of_path = original_path.replace('.png', '_residual_of.png')
                    cv2.imwrite(residual_of_path, residual_of_rgb)
                    png_path, npy_path, residual_npy = get_deep_feature(network_name, video_name, residual_of_path, qp, layer_name)

                try:
                    # _, _, next_npy = get_deep_feature(network_name, video_name, next_path, qp)
                    os.remove(next_path)
                    print(f"Deleted: {next_path}")
                except OSError as e:
                    print(f"Error: {e.strerror}, file: {e.filename}")

                # feature combined
                all_frame_activations_residual.append(residual_npy)

            averaged_frame_npy_residual = process_video_feature(all_frame_activations_residual, network_name)
            print(len(averaged_frame_npy_residual))
            # averaged_frame_npy = np.hstack((averaged_frame_npy_original, averaged_frame_npy_residual))
            # print("Shape of combined features:", averaged_frame_npy.shape)

            dataset_name = video_type.replace('original_', '')
            if residual_name == 'frame_diff':
                output_npy_path = f'../features_residual/{network_name}/{layer_name}/{dataset_name}/{compressed_type}/'
            elif residual_name == 'optical_flow':
                output_npy_path = f'../features_residual_of/{network_name}/{layer_name}/{dataset_name}/{compressed_type}/'
            os.makedirs(output_npy_path, exist_ok=True)

            # save the processed data as a new numpy file
            output_npy_name = f'{output_npy_path}video_{str(i + 1)}_{feature_name}.npy'
            np.save(output_npy_name, averaged_frame_npy_residual)
            print(f'Processed file saved to: {output_npy_name}')

            # remove tmp folders
            shutil.rmtree(png_path)
            shutil.rmtree(npy_path)
            shutil.rmtree(sampled_frame_path)

            end_time = time.time()
            print(f"Execution time for feature extraction: {end_time - start_time} seconds\n")

