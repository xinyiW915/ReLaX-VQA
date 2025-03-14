import argparse
import time
import math
import os
import shutil
from joblib import load
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from thop import profile
from torchvision import models, transforms

from extractor.visualise_vit_layer import VitGenerator
from relax_vqa import get_deep_feature, process_video_feature, process_patches, get_frame_patches, flow_to_rgb, merge_fragments, concatenate_features
from extractor.vf_extract import process_video_residual
from model_regression import Mlp, preprocess_data

def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        elif k == 'n_averaged':
            continue
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def preprocess_data(X, y=None, imp=None, scaler=None):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, device='cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.where(torch.isnan(X) | torch.isinf(X), torch.tensor(0.0, device=X.device), X)

    if imp is not None or scaler is not None:
        X_np = X.cpu().numpy()
        if imp is not None:
            X_np = imp.transform(X_np)
        if scaler is not None:
            X_np = scaler.transform(X_np)
        X = torch.from_numpy(X_np).to(X.device)

    if y is not None and y.size > 0:
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=X.device)
        y = y.reshape(-1).squeeze()
    else:
        y = None

    return X, y, imp, scaler

def load_model(config, device, input_features=35203):
    network_name = 'relaxvqa'
    # input_features = X_test_processed.shape[1]
    model = Mlp(input_features=input_features, out_features=1, drop_rate=0.2, act_layer=nn.GELU).to(device)
    if config['is_finetune']:
        model_path = os.path.join(config['save_path'], f"fine_tune_model/{config['video_type']}_{network_name}_{config['select_criteria']}_fine_tuned_model.pth")
    else:
        model_path = os.path.join(config['save_path'], f"{config['train_data_name']}_{network_name}_{config['select_criteria']}_trained_median_model_param_onLSVQ_TEST.pth")
    # print("Loading model from:", model_path)
    state_dict = torch.load(model_path, map_location=device)
    fixed_state_dict = fix_state_dict(state_dict)
    try:
        model.load_state_dict(fixed_state_dict)
    except RuntimeError as e:
        print(e)
    return model

def evaluate_video_quality(config, resnet50, vit, model_mlp):
    is_finetune = config['is_finetune']
    save_path = config['save_path']
    video_type = config['video_type']
    video_name = config['video_name']
    framerate = config['framerate']
    sampled_fragment_path = os.path.join("../video_sampled_frame/sampled_frame/", "test_sampled_fragment")

    if video_type == 'youtube_ugc':
        video_path = f'../ugc_original_videos/{video_name}.mkv'
    else:
        video_path = f'../ugc_original_videos/{video_name}.mp4'
    target_size = 224
    patch_size = 16
    top_n = int((target_size / patch_size) * (target_size / patch_size))

    # sampled video frames
    start_time = time.time()
    frames, frames_next = process_video_residual(video_type, video_name, framerate, video_path, sampled_fragment_path)

    # get ResNet50 layer-stack features and ViT pooling features
    all_frame_activations_resnet = []
    all_frame_activations_vit = []
    # get fragments ResNet50 features and ViT features
    all_frame_activations_sampled_resnet = []
    all_frame_activations_merged_resnet = []
    all_frame_activations_sampled_vit = []
    all_frame_activations_merged_vit = []

    batch_size = 64  # Define the number of frames to process in parallel
    for i in range(0, len(frames_next), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch_frames]
        batch_frames_next = frames_next[i:i + batch_size]
        batch_tensors = torch.stack([transforms.ToTensor()(frame) for frame in batch_frames]).to(device)
        batch_rgb_tensors = torch.stack([transforms.ToTensor()(frame_rgb) for frame_rgb in batch_rgb_frames]).to(device)
        batch_tensors_next = torch.stack([transforms.ToTensor()(frame_next) for frame_next in batch_frames_next]).to(device)

        # compute residuals
        residuals = torch.abs(batch_tensors_next - batch_tensors)

        # calculate optical flows
        batch_gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in batch_frames]
        batch_gray_frames_next = [cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY) for frame_next in batch_frames_next]
        batch_gray_frames = [frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame for frame in batch_gray_frames]
        batch_gray_frames_next = [frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame for frame in batch_gray_frames_next]
        flows = [cv2.calcOpticalFlowFarneback(batch_gray_frames[j], batch_gray_frames_next[j], None, 0.5, 3, 15, 3, 5, 1.2,0) for j in range(len(batch_gray_frames))]

        for j in range(batch_tensors.size(0)):
            '''sampled video frames'''
            frame_tensor = batch_tensors[j].unsqueeze(0)
            frame_rgb_tensor = batch_rgb_tensors[j].unsqueeze(0)
            # frame_next_tensor = batch_tensors_next[j].unsqueeze(0)
            frame_number = i + j + 1

            # ResNet50 layer-stack features
            activations_dict_resnet, _, _ = get_deep_feature('resnet50', video_name, frame_rgb_tensor, frame_number, resnet50, device, 'layerstack')
            all_frame_activations_resnet.append(activations_dict_resnet)
            # ViT pooling features
            activations_dict_vit, _, _ = get_deep_feature('vit', video_name, frame_rgb_tensor, frame_number, vit, device, 'pool')
            all_frame_activations_vit.append(activations_dict_vit)


            '''residual video frames'''
            residual = residuals[j].unsqueeze(0)
            flow = flows[j]
            original_path = os.path.join(sampled_fragment_path, f'{video_name}_{frame_number}.png')

            # Frame Differencing
            residual_frag_path, diff_frag, positions = process_patches(original_path, 'frame_diff', residual, patch_size, target_size, top_n)
            # Frame fragment
            frame_patches = get_frame_patches(frame_tensor, positions, patch_size, target_size)
            # Optical Flow
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
            sampled_frag_activations_vit,_, _ = get_deep_feature('vit', video_name, frame_patches, frame_number, vit, device, 'pool')
            merged_frag_activations_vit, _, _ = get_deep_feature('vit', video_name, merged_frag, frame_number, vit, device, 'pool')
            all_frame_activations_sampled_vit.append(sampled_frag_activations_vit)
            all_frame_activations_merged_vit.append(merged_frag_activations_vit)

    print(f'video frame number: {len(all_frame_activations_resnet)}')
    averaged_frames_resnet = process_video_feature(all_frame_activations_resnet, 'resnet50', 'layerstack')
    averaged_frames_vit = process_video_feature(all_frame_activations_vit, 'vit', 'pool')
    # print("ResNet50 layer-stacking feature shape:", averaged_frames_resnet.shape)
    # print("ViT pooling feature shape:", averaged_frames_vit.shape)
    averaged_frames_sampled_resnet = process_video_feature(all_frame_activations_sampled_resnet, 'resnet50', 'layerstack')
    averaged_frames_merged_resnet = process_video_feature(all_frame_activations_merged_resnet, 'resnet50', 'pool')
    averaged_combined_feature_resnet = concatenate_features(averaged_frames_sampled_resnet, averaged_frames_merged_resnet)
    # print("Sampled fragments ResNet50 features shape:", averaged_frames_sampled_resnet.shape)
    # print("Merged fragments ResNet50 features shape:", averaged_frames_merged_resnet.shape)
    averaged_frames_sampled_vit = process_video_feature(all_frame_activations_sampled_vit, 'vit', 'pool')
    averaged_frames_merged_vit = process_video_feature(all_frame_activations_merged_vit, 'vit', 'pool')
    averaged_combined_feature_vit = concatenate_features(averaged_frames_sampled_vit, averaged_frames_merged_vit)
    # print("Sampled fragments ViT features shape:", averaged_frames_sampled_vit.shape)
    # print("Merged fragments ResNet50 features shape:", averaged_frames_merged_vit.shape)

    # remove tmp folders
    shutil.rmtree(sampled_fragment_path)

    # concatenate features
    combined_features = torch.cat([torch.mean(averaged_frames_resnet, dim=0), torch.mean(averaged_frames_vit, dim=0),
                                   torch.mean(averaged_combined_feature_resnet, dim=0), torch.mean(averaged_combined_feature_vit, dim=0)], dim=0).view(1, -1)
    imputer = load(f'{save_path}/scaler/{video_type}_imputer.pkl')
    scaler = load(f'{save_path}/scaler/{video_type}_scaler.pkl')
    X_test_processed, _, _, _ = preprocess_data(combined_features, None, imp=imputer, scaler=scaler)
    feature_tensor = X_test_processed

    # evaluation for test video
    model_mlp.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prediction = model_mlp(feature_tensor)
            predicted_score = prediction.item()
            # print(f"Raw Predicted Quality Score: {predicted_score}")
            run_time = time.time() - start_time

            if not is_finetune:
                if video_type in ['konvid_1k', 'youtube_ugc']:
                    scaled_prediction = ((predicted_score - 1) / (99 / 4)) + 1.0
                    # print(f"Scaled Predicted Quality Score (1-5): {scaled_prediction}")
                    return scaled_prediction, run_time
                else:
                    scaled_prediction = predicted_score
                    return scaled_prediction, run_time
            else:
                return predicted_score, run_time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='gpu', help='cpu or gpu')
    parser.add_argument('-model_name', type=str, default='Mlp', help='Name of the regression model')
    parser.add_argument('-select_criteria', type=str, default='byrmse', help='Selection criteria')
    parser.add_argument('-train_data_name', type=str, default='lsvq_train', help='Name of the training data')
    parser.add_argument('-is_finetune', type=bool, default=True, help='With or without finetune')
    parser.add_argument('-save_path', type=str, default='../model/', help='Path to save models')
    parser.add_argument('-video_type', type=str, default='konvid_1k', help='Type of video')
    parser.add_argument('-video_name', type=str, default='5636101558_540p', help='Name of the video')
    parser.add_argument('-framerate', type=float, default=24, help='Frame rate of the video')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    config = vars(args)
    if config['device'] == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

    # load models to device
    resnet50 = models.resnet50(pretrained=True).to(device)
    vit = VitGenerator('vit_base', 16, device, evaluate=True, random=False, verbose=True)
    model_mlp = load_model(config, device)

    total_time = 0
    num_runs = 10
    for i in range(num_runs):
        quality_prediction, run_time = evaluate_video_quality(config, resnet50, vit, model_mlp)
        print(f"Run {i + 1} - Time taken: {run_time:.4f} seconds")

        total_time += run_time
    average_time = total_time / num_runs

    print(f"Average running time over {num_runs} runs: {average_time:.4f} seconds")
    print("Predicted Quality Score:", quality_prediction)