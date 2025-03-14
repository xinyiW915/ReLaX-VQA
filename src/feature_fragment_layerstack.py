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
from extractor import visualise_vgg, visualise_resnet, vf_extract


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
    sampled_frame_path = os.path.join(common_path, f'fragment_layerstack', f'video_{str(i + 1)}')
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
        if layer_name == 'layerstack':
            all_layers = ['resnet50.conv1',
                          'resnet50.layer1[0]', 'resnet50.layer1[1]', 'resnet50.layer1[2]',
                          'resnet50.layer2[0]', 'resnet50.layer2[1]', 'resnet50.layer2[2]', 'resnet50.layer2[3]',
                          'resnet50.layer3[0]', 'resnet50.layer3[1]', 'resnet50.layer3[2]', 'resnet50.layer3[3]',
                          'resnet50.layer4[0]', 'resnet50.layer4[1]', 'resnet50.layer4[2]']
            resnet50 = model
            activations_dict, _, total_flops, total_params = visualise_resnet.process_video_frame(video_name, frame, frame_number, all_layers, resnet50, device)

    elif network_name == 'vgg16':
        if layer_name == 'layerstack':
            all_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
            vgg16 = model
            activations_dict, _, total_flops, total_params = visualise_vgg.process_video_frame(video_name, frame, frame_number, all_layers, vgg16, device)

    return png_path, activations_dict, total_flops, total_params

def process_video_feature(video_feature, network_name, layer_name):
    # print(f'video frame number: {len(video_feature)}')

    # initialize an empty list to store processed frames
    averaged_frames = []
    # iterate through each frame in the video_feature
    for frame in video_feature:
        frame_features = []

        if layer_name == 'layerstack':
            # iterate through each layer in the current framex
            for layer_array in frame.values():
                # calculate the mean along the specified axes (1 and 2) for each layer
                layer_mean = torch.mean(layer_array, dim=(1, 2))
                # append the calculated mean to the list for the current frame
                frame_features.append(layer_mean)

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

def get_patch_diff(residual_frame, patch_size):
    h, w = residual_frame.shape[2:]  # Assuming (1, C, H, W) shape
    h_adj = (h // patch_size) * patch_size
    w_adj = (w // patch_size) * patch_size
    residual_frame_adj = residual_frame[:, :, :h_adj, :w_adj]
    # calculate absolute patch difference
    diff = torch.zeros((h_adj // patch_size, w_adj // patch_size), device=residual_frame.device)
    for i in range(0, h_adj, patch_size):
        for j in range(0, w_adj, patch_size):
            patch = residual_frame_adj[:, :, i:i + patch_size, j:j + patch_size]
            # absolute sum
            diff[i // patch_size, j // patch_size] = torch.sum(torch.abs(patch))
    return diff

def extract_important_patches(residual_frame, diff, patch_size=16, target_size=224, top_n=196):
    # find top n patches indices
    patch_idx = torch.argsort(-diff.view(-1))
    top_patches = [(idx // diff.shape[1], idx % diff.shape[1]) for idx in patch_idx[:top_n]]
    sorted_idx = sorted(top_patches, key=lambda x: (x[0], x[1]))

    imp_patches_img = torch.zeros((residual_frame.shape[1], target_size, target_size), dtype=residual_frame.dtype, device=residual_frame.device)
    patches_per_row = target_size // patch_size  # 14
    # order the patch in the original location relation
    positions = []
    for idx, (y, x) in enumerate(sorted_idx):
        patch = residual_frame[:, :, y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size]
        # new patch location
        row_idx = idx // patches_per_row
        col_idx = idx % patches_per_row
        start_y = row_idx * patch_size
        start_x = col_idx * patch_size
        imp_patches_img[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = patch
        positions.append((y.item(), x.item()))
    return imp_patches_img, positions

def get_frame_patches(frame, positions, patch_size, target_size):
    imp_patches_img = torch.zeros((frame.shape[1], target_size, target_size), dtype=frame.dtype, device=frame.device)
    patches_per_row = target_size // patch_size

    for idx, (y, x) in enumerate(positions):
        start_y = y * patch_size
        start_x = x * patch_size
        end_y = start_y + patch_size
        end_x = start_x + patch_size

        patch = frame[:, :, start_y:end_y, start_x:end_x]
        row_idx = idx // patches_per_row
        col_idx = idx % patches_per_row
        target_start_y = row_idx * patch_size
        target_start_x = col_idx * patch_size

        imp_patches_img[:, target_start_y:target_start_y + patch_size,
        target_start_x:target_start_x + patch_size] = patch.squeeze(0)
    return imp_patches_img

def process_patches(original_path, frag_name, residual, patch_size, target_size, top_n):
    diff = get_patch_diff(residual, patch_size)
    imp_patches, positions = extract_important_patches(residual, diff, patch_size, target_size, top_n)
    if frag_name == 'frame_diff':
        frag_path = original_path.replace('.png', '_residual_imp.png')
    elif frag_name == 'optical_flow':
        frag_path = original_path.replace('.png', '_residual_of_imp.png')
    # cv2.imwrite(frag_path, imp_patches)
    return frag_path, imp_patches, positions

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

def merge_fragments(diff_fragment, flow_fragment):
    alpha = 0.5
    merged_fragment = diff_fragment * alpha + flow_fragment * (1 - alpha)
    return merged_fragment

def concatenate_features(frame_feature, residual_feature):
    return torch.cat((frame_feature, residual_feature), dim=-1)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    # device = torch.device("cpu")

    video_type = 'test' # test
                        # resolution_ugc/konvid_1k/live_vqc/cvd_2014/live_qualcomm
                        # lsvq_train/lsvq_test/lsvq_test_1080P/
    frag_name = 'framediff_frag' # framediff_frag, opticalflow_frag, sampled_frag, merged_frag
    network_name = 'resnet50'
    layer_name = 'layerstack'
    if network_name == 'resnet50':
        model = models.resnet50(pretrained=True).to(device)
    else:
        model = models.vgg16(pretrained=True).to(device)

    logger.info(f"video type: {video_type}, frag name: {frag_name}, network name: {network_name}, layer name: {layer_name}")
    logger.info(f"torch cuda: {torch.cuda.is_available()}")

    videodata = load_metadata(video_type)
    valid_video_types = ['test',
                         'resolution_ugc', 'konvid_1k', 'live_vqc', 'cvd_2014', 'live_qualcomm',
                         'lsvq_train', 'lsvq_test', 'lsvq_test_1080P']
    target_size = 224
    patch_size = 16
    top_n = int((target_size / patch_size) * (target_size / patch_size))

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
                frame_path = os.path.join(sampled_frame_path, f'{video_name}_{frame_number}.png')
                # compute residual
                frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)
                frame_next_tensor = transforms.ToTensor()(frame_next).unsqueeze(0).to(device)

                # DNN feature extraction
                if frag_name in ['framediff_frag', 'sampled_frag', 'merged_frag']:
                    residual_frag_path, diff_frag, positions = compute_frame_difference(frame_tensor, frame_next_tensor, frame_path, patch_size, target_size, top_n)
                    png_path, frag_activations, total_flops, total_params = get_deep_feature(network_name, video_name, diff_frag, frame_number, model, device, layer_name)

                    if frag_name == 'sampled_frag':
                        frame_patches = get_frame_patches(frame_tensor, positions, patch_size, target_size)
                        png_path, frag_activations, total_flops, total_params = get_deep_feature(network_name, video_name, frame_patches, frame_number, model, device, layer_name)

                    elif frag_name == 'merged_frag':
                        of_frag_path, flow_frag, _ = compute_optical_flow(frame, frame_next, frame_path, patch_size, target_size, top_n, device)
                        merged_frag = merge_fragments(diff_frag, flow_frag)
                        png_path, frag_activations, total_flops, total_params = get_deep_feature(network_name, video_name, merged_frag, frame_number, model, device, layer_name)

                elif frag_name == 'opticalflow_frag':
                    of_frag_path, flow_frag, _ = compute_optical_flow(frame, frame_next, frame_path, patch_size, target_size, top_n, device)
                    png_path, frag_activations, total_flops, total_params = get_deep_feature(network_name, video_name, flow_frag, frame_number, model, device, layer_name)

                # feature combined
                all_frame_activations_feats.append(frag_activations)

            averaged_frames_feats = process_video_feature(all_frame_activations_feats, network_name, layer_name)
            print("Features shape:", averaged_frames_feats.shape)
            # remove tmp folders
            shutil.rmtree(png_path)
            shutil.rmtree(sampled_frame_path)

            averaged_npy = averaged_frames_feats.cpu().numpy()
            # save the processed data as numpy file
            output_npy_path = f'../features/{video_type}/{frag_name}_{network_name}_{layer_name}/'
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
        mat_file_name = f'{mat_file_path}{video_type}_{frag_name}_{network_name}_{layer_name}_feats.mat'
        scipy.io.savemat(mat_file_name, {video_type: feats_matrix})
        logger.debug(f'Successfully created {mat_file_name}')
    logger.debug(f"Execution time for all feature extraction: {time.time() - begin_time:.4f} seconds\n")