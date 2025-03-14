import argparse
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
from extractor.vf_extract import process_video_residual
from extractor.visualise_vit_layer import VitGenerator
from relax_vqa import get_deep_feature, process_video_feature, process_patches, get_frame_patches, flow_to_rgb, merge_fragments, concatenate_features


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
    sampled_frame_path = os.path.join(common_path, f'relaxvqa', f'video_{str(i + 1)}')
    feature_name = f"{network_name}_feature_map"

    if video_type == 'resolution_ugc':
        resolution = '360P'
        # video_path = f'/user/work/um20242/dataset/ugc-dataset/{resolution}/{video_name}.mkv'
        video_path = Path(f"D:/video_dataset/ugc-dataset/youtube_ugc/original_videos/{resolution}") / f"{video_name}.mkv"
        sampled_frame_path = os.path.join(common_path, f'ytugc_sampled_frame_{resolution}', f'video_{str(i + 1)}')
        feature_name = f"{network_name}_feature_map_{resolution}"

    return video_name, video_path, sampled_frame_path, feature_name, video_width, video_height, pixfmt, framerate

# Frame Differencing
def compute_frame_difference(frame_tensor, frame_next_tensor, frame_path, patch_size, target_size, top_n):
    residual = torch.abs(frame_next_tensor - frame_tensor)
    return process_patches(frame_path, 'frame_diff', residual, patch_size, target_size, top_n)

# Optical Flow
def compute_optical_flow(frame, frame_next, frame_path, patch_size, target_size, top_n, device):
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY),
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    opticalflow_rgb = flow_to_rgb(flow)
    opticalflow_rgb_tensor = transforms.ToTensor()(opticalflow_rgb).unsqueeze(0).to(device)
    return process_patches(frame_path, 'optical_flow', opticalflow_rgb_tensor, patch_size, target_size, top_n)

def extract_features(config, video_idx):
    video_type = config['video_type']
    model_name = config['model_name']
    target_size = config['target_size']
    patch_size = config['patch_size']
    top_n = int((target_size / patch_size) * (target_size / patch_size))

    # sampled video frames
    start_time = time.time()
    video_name, video_path, sampled_frame_path, feature_name, video_width, video_height, pixfmt, framerate = get_video_paths(model_name, video_type, videodata, video_idx)
    frames, frames_next = process_video_residual(video_type, video_name, framerate, video_path, sampled_frame_path)

    logger.info(f'{video_name}')
    # get ResNet50 layer-stack features and ViT pooling features
    all_frame_activations_resnet = []
    all_frame_activations_vit = []
    # get fragments ResNet50 features and ViT features
    all_frame_activations_sampled_resnet = []
    all_frame_activations_merged_resnet = []
    all_frame_activations_sampled_vit = []
    all_frame_activations_merged_vit = []
    for j, (frame, frame_next) in enumerate(zip(frames, frames_next)):
        frame_number = j + 1
        original_path = os.path.join(sampled_frame_path, f'{video_name}_{frame_number}.png')

        '''sampled video frames'''
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(device)
        # ResNet50 layer-stack features
        activations_dict_resnet, _, _ = get_deep_feature('resnet50', video_name, frame_rgb_tensor, frame_number, resnet50, device, 'layerstack')
        all_frame_activations_resnet.append(activations_dict_resnet)
        # ViT pooling features
        activations_dict_vit, _, _ = get_deep_feature('vit', video_name, frame_rgb_tensor, frame_number, vit, device, 'pool')
        all_frame_activations_vit.append(activations_dict_vit)

        '''residual video frames'''
        frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)
        frame_next_tensor = transforms.ToTensor()(frame_next).unsqueeze(0).to(device)
        # Frame Differencing
        residual = torch.abs(frame_next_tensor - frame_tensor)
        residual_frag_path, diff_frag, positions = process_patches(original_path, 'frame_diff', residual, patch_size, target_size, top_n)

        # Frame fragment
        frame_patches = get_frame_patches(frame_tensor, positions, patch_size, target_size)

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        opticalflow_rgb = flow_to_rgb(flow)
        opticalflow_rgb_tensor = transforms.ToTensor()(opticalflow_rgb).unsqueeze(0).to(device)
        opticalflow_frag_path, flow_frag, _ = process_patches(original_path, 'optical_flow', opticalflow_rgb_tensor, patch_size, target_size, top_n)

        merged_frag = merge_fragments(diff_frag, flow_frag)

        # fragments ResNet50 features
        sampled_frag_activations_resnet, _, _ = get_deep_feature('resnet50', video_name, frame_patches, frame_number, resnet50, device, 'layerstack')
        merged_frag_activations_resnet, _, _ = get_deep_feature('resnet50', video_name, merged_frag, frame_number, resnet50, device, 'pool')
        all_frame_activations_sampled_resnet.append(sampled_frag_activations_resnet)
        all_frame_activations_merged_resnet.append(merged_frag_activations_resnet)
        # fragments ViT features
        sampled_frag_activations_vit, _, _ = get_deep_feature('vit', video_name, frame_patches, frame_number, vit, device, 'pool')
        merged_frag_activations_vit, _, _ = get_deep_feature('vit', video_name, merged_frag, frame_number, vit, device, 'pool')
        all_frame_activations_sampled_vit.append(sampled_frag_activations_vit)
        all_frame_activations_merged_vit.append(merged_frag_activations_vit)

    print(f'video frame number: {len(all_frame_activations_resnet)}')
    averaged_frames_resnet = process_video_feature(all_frame_activations_resnet, 'resnet50', 'layerstack')
    averaged_frames_vit = process_video_feature(all_frame_activations_vit, 'vit', 'pool')
    # print("ResNet50 layer-stacking feature shape:", averaged_frames_resnet.shape)
    # print("ViT pooling feature shape:", averaged_frames_vit.shape)
    averaged_frames_sampled_resnet = process_video_feature(all_frame_activations_sampled_resnet, 'resnet50','layerstack')
    averaged_frames_merged_resnet = process_video_feature(all_frame_activations_merged_resnet, 'resnet50','pool')
    averaged_combined_feature_resnet = concatenate_features(averaged_frames_sampled_resnet, averaged_frames_merged_resnet)
    # print("Sampled fragments ResNet50 features shape:", averaged_frames_sampled_resnet.shape)
    # print("Merged fragments ResNet50 features shape:", averaged_frames_merged_resnet.shape)
    averaged_frames_sampled_vit = process_video_feature(all_frame_activations_sampled_vit, 'vit', 'pool')
    averaged_frames_merged_vit = process_video_feature(all_frame_activations_merged_vit, 'vit', 'pool')
    averaged_combined_feature_vit = concatenate_features(averaged_frames_sampled_vit, averaged_frames_merged_vit)
    # print("Sampled fragments ViT features shape:", averaged_frames_sampled_vit.shape)
    # print("Merged fragments ResNet50 features shape:", averaged_frames_merged_vit.shape)

    # remove tmp folders
    shutil.rmtree(sampled_frame_path)

    # concatenate features
    combined_features = torch.cat([torch.mean(averaged_frames_resnet, dim=0), torch.mean(averaged_frames_vit, dim=0),
                                           torch.mean(averaged_combined_feature_resnet, dim=0), torch.mean(averaged_combined_feature_vit, dim=0)], dim=0).view(1, -1)

    feats_npy = combined_features.cpu().numpy()
    # save the processed data as numpy file
    output_npy_path = f'../features/{video_type}/{model_name}/'
    os.makedirs(output_npy_path, exist_ok=True)
    # output_npy_name = f'{output_npy_path}video_{str(video_idx + 1)}_{feature_name}.npy'
    # np.save(output_npy_name, feats_npy)
    # print(f'Processed file saved to: {output_npy_name}')

    run_time = time.time() - start_time
    logger.debug(f"Execution time for {video_name} feature extraction: {run_time:.4f} seconds")
    return feats_npy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='gpu', help='cpu or gpu')
    parser.add_argument('-model_name', type=str, default='relaxvqa')
    parser.add_argument('-target_size', type=int, default=224)
    parser.add_argument('-patch_size', type=int, default=16)
    parser.add_argument('-video_type', type=str, default='test', help='Type of video datasets: test, resolution_ugc, konvid_1k, live_vqc, cvd_2014, lsvq_train, lsvq_test, lsvq_test_1080P')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    config = vars(args)
    if config['device'] == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"ReLax-VQA --- video type: {config['video_type']}")
    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")
    logger.debug(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

    begin_time = time.time()
    # load models to device
    resnet50 = models.resnet50(pretrained=True).to(device)
    vit = VitGenerator('vit_base', 16, device, evaluate=True, random=False, verbose=True)
    videodata = load_metadata(config['video_type'])

    for video_idx in range(len(videodata)):
        feats_npy = extract_features(config, video_idx)

        # save feature mat file
        average_data = np.mean(feats_npy, axis=0)
        if video_idx == 0:
            feats_matrix = np.zeros((len(videodata),) + average_data.shape)
        feats_matrix[video_idx] = average_data

    print((f'All features shape: {feats_matrix.shape}'))
    logger.debug(f'\n All features shape: {feats_matrix.shape}')
    mat_file_path = f"../features/"
    mat_file_name = f"{mat_file_path}{config['video_type']}_{config['model_name']}_feats.mat"
    scipy.io.savemat(mat_file_name, {config['video_type']: feats_matrix})
    logger.debug(f'Successfully created {mat_file_name}')
    logger.debug(f"Execution time for all feature extraction: {time.time() - begin_time:.4f} seconds\n")