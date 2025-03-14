from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path
import scipy.io
import shutil
import torch
import time
import cv2
from torchvision import models, transforms

from utils.logger_setup import logger
from extractor import visualise_vgg_layer, visualise_resnet_layer, visualise_vit_layer, vf_extract


def load_metadata(video_type):
    print(f'video_type: {video_type}\n')
    # Test
    if video_type == 'test':
        return pd.read_csv("../metadata/test_videos.csv")
    # NR:
    elif video_type == 'resolution_ugc':
        resolution = '360P'
        return pd.read_csv(f"../metadata/YOUTUBE_UGC_{resolution}_metadata.csv")
    else:
        return pd.read_csv(f'../metadata/{video_type.upper()}_metadata.csv')

def get_video_paths(network_name, video_type, videodata, i):
    video_name = videodata['vid'][i]
    video_width = videodata['width'][i]
    video_height = videodata['height'][i]
    pixfmt = videodata['pixfmt'][i]
    framerate = videodata['framerate'][i]
    common_path = os.path.join('..', 'video_sampled_frame')

    # Test
    if video_type == 'test':
        video_path = f"../ugc_original_videos/{video_name}.mp4"

    # NR:
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
    elif video_type == 'youtube_ugc':
        video_path = Path("D:/video_dataset/ugc-dataset/youtube_ugc/") / f"{video_name}.mkv"
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    sampled_frame_path = os.path.join(common_path, f'pool', f'video_{str(i + 1)}')
    feature_name = f"{network_name}_feature_map"

    if video_type == 'resolution_ugc':
        resolution = '360P'
        # video_path = f'/user/work/um20242/dataset/ugc-dataset/{resolution}/{video_name}.mkv'
        video_path = Path(f"D:/video_dataset/ugc-dataset/youtube_ugc/original_videos/{resolution}") / f"{video_name}.mkv"
        sampled_frame_path = os.path.join(common_path, f'ytugc_sampled_frame_{resolution}', f'video_{str(i + 1)}')
        feature_name = f"{network_name}_feature_map_{resolution}"

    return video_name, video_path, sampled_frame_path, feature_name, video_width, video_height, pixfmt, framerate

def get_deep_feature(network_name, video_name, frame, frame_number, model, device, layer_name):
    png_path = f'../visualisation/{network_name}_{layer_name}/{video_name}/'
    os.makedirs(png_path, exist_ok=True)

    if network_name == 'resnet50':
        if layer_name == 'pool':
            visual_layer = 'resnet50.avgpool' # before avg_pool
            resnet50 = model
            activations_dict, _, total_flops, total_params = visualise_resnet_layer.process_video_frame(video_name, frame, frame_number, visual_layer, resnet50, device)

    elif network_name == 'vgg16':
        if layer_name == 'pool':
            # visual_layer = 'fc1'
            visual_layer = 'fc2' # fc1 = vgg16.classifier[0], fc2 = vgg16.classifier[3]
            vgg16 = model
            activations_dict, _, total_flops, total_params = visualise_vgg_layer.process_video_frame(video_name, frame, frame_number, visual_layer, vgg16, device)

    elif network_name == 'vit':
        patch_size = 16
        activations_dict, _, total_flops, total_params = visualise_vit_layer.process_video_frame(video_name, frame, frame_number, model, patch_size, device)

    return png_path, activations_dict, total_flops, total_params

def process_video_feature(video_feature, network_name, layer_name):
    # print(f'video frame number: {len(video_feature)}')

    # initialize an empty list to store processed frames
    averaged_frames = []
    # iterate through each frame in the video_feature
    for frame in video_feature:
        frame_features = []

        if layer_name == 'pool':
            if network_name == 'vit':
                # global mean and std
                global_mean = torch.mean(frame, dim=0)
                global_max = torch.max(frame, dim=0)[0]
                global_std = torch.std(frame, dim=0)
                # concatenate all pooling
                combined_features = torch.hstack([global_mean, global_max, global_std])
                frame_features.append(combined_features)

            elif network_name == 'resnet50':
                frame = torch.squeeze(torch.tensor(frame))
                # global mean and std
                global_mean = torch.mean(frame, dim=0)
                global_max = torch.max(frame, dim=0)[0]
                global_std = torch.std(frame, dim=0)
                # concatenate all pooling
                combined_features = torch.hstack([frame, global_mean, global_max, global_std])
                frame_features.append(combined_features)

        # concatenate the layer means horizontally to form the processed frame
        processed_frame = torch.hstack(frame_features)
        averaged_frames.append(processed_frame)
    averaged_frames = torch.stack(averaged_frames)

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

# Frame Differencing
def compute_frame_difference(frame_tensor, frame_next_tensor):
    residual = torch.abs(frame_next_tensor - frame_tensor)
    return residual

# Optical Flow
def compute_optical_flow(frame, frame_next, device):
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY),
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    opticalflow_rgb = flow_to_rgb(flow)
    opticalflow_rgb_tensor = transforms.ToTensor()(opticalflow_rgb).unsqueeze(0).to(device)
    return opticalflow_rgb_tensor


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    # device = torch.device("cpu")

    video_type = 'test' # test
                        # resolution_ugc/konvid_1k/live_vqc/cvd_2014/live_qualcomm
                        # lsvq_train/lsvq_test/lsvq_test_1080P/
    frame_name = 'sampled_frame' # sampled_frame, frame_diff, optical_flow
    network_name = 'vit'
    layer_name = 'pool'
    if network_name == 'vit':
        model = visualise_vit_layer.VitGenerator('vit_base', 16, device, evaluate=True, random=False, verbose=True)
    elif network_name == 'resnet50':
        model = models.resnet50(pretrained=True).to(device)
    else:
        model = models.vgg16(pretrained=True).to(device)

    logger.info(f"video type: {video_type}, frame name: {frame_name}, network name: {network_name}, layer name: {layer_name}")
    logger.info(f"torch cuda: {torch.cuda.is_available()}")

    videodata = load_metadata(video_type)
    valid_video_types = ['test',
                         'resolution_ugc', 'konvid_1k', 'live_vqc', 'cvd_2014', 'live_qualcomm',
                         'lsvq_train', 'lsvq_test', 'lsvq_test_1080P']

    begin_time = time.time()
    if video_type in valid_video_types:
        for i in range(len(videodata)):
            start_time = time.time()

            video_name, video_path, sampled_frame_path, feature_name, video_width, video_height, pixfmt, framerate = get_video_paths(network_name, video_type, videodata, i)
            frames, frames_next = vf_extract.process_video_residual(video_type, video_name, framerate, video_path, sampled_frame_path)

            logger.info(f'{video_name}')
            all_frame_activations_feats = []
            for j, (frame, frame_next) in enumerate(zip(frames, frames_next)):
                frame_number = j + 1

                # DNN feature extraction
                if frame_name in ['frame_diff', 'optical_flow']:
                    # compute residual
                    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)
                    frame_next_tensor = transforms.ToTensor()(frame_next).unsqueeze(0).to(device)

                    if frame_name == 'frame_diff':
                        residual = compute_frame_difference(frame_tensor, frame_next_tensor)
                        png_path, frag_activations, total_flops, total_params = get_deep_feature(network_name, video_name, residual, frame_number, model, device, layer_name)

                    elif frame_name == 'optical_flow':
                        opticalflow_rgb_tensor = compute_optical_flow(frame, frame_next, device)
                        png_path, frag_activations, total_flops, total_params = get_deep_feature(network_name, video_name, opticalflow_rgb_tensor, frame_number, model, device, layer_name)

                elif frame_name == 'sampled_frame':
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(device)
                    png_path, frag_activations, total_flops, total_params = get_deep_feature(network_name, video_name, frame_rgb_tensor, frame_number, model, device, layer_name)

                # feature combined
                all_frame_activations_feats.append(frag_activations)

            averaged_frames_feats = process_video_feature(all_frame_activations_feats, network_name, layer_name)
            print("Features shape:", averaged_frames_feats.shape)
            # remove tmp folders
            shutil.rmtree(png_path)
            shutil.rmtree(sampled_frame_path)

            averaged_npy = averaged_frames_feats.cpu().numpy()
            # save the processed data as numpy file
            output_npy_path = f'../features/{video_type}/{frame_name}_{network_name}_{layer_name}/'
            os.makedirs(output_npy_path, exist_ok=True)
            # output_npy_name = f'{output_npy_path}video_{str(i + 1)}_{feature_name}.npy'
            # np.save(output_npy_name, averaged_npy)
            # print(f'Processed file saved to: {output_npy_name}')

            run_time = time.time() - start_time
            print(f"Execution time for {video_name} feature extraction: {run_time:.4f} seconds\n")

            # save feature mat file
            average_data = np.mean(averaged_npy, axis=0)
            if i == 0:
                feats_matrix = np.zeros((len(videodata),) + average_data.shape)
            feats_matrix[i] = average_data

        print((f'All features shape: {feats_matrix.shape}'))
        logger.debug(f'\n All features shape: {feats_matrix.shape}')
        mat_file_path = f'../features/{video_type}/'
        mat_file_name = f'{mat_file_path}{video_type}_{frame_name}_{network_name}_{layer_name}_feats.mat'
        scipy.io.savemat(mat_file_name, {video_type: feats_matrix})
        logger.debug(f'Successfully created {mat_file_name}')
    logger.debug(f"Execution time for all feature extraction: {time.time() - begin_time:.4f} seconds\n")