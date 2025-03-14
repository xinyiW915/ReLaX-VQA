import torch
import os
import cv2
import numpy as np
from extractor import visualise_resnet, visualise_resnet_layer, visualise_vit_layer


def get_deep_feature(network_name, video_name, frame, frame_number, model, device, layer_name):
    if network_name == 'resnet50':
        if layer_name == 'layerstack':
            all_layers = ['resnet50.conv1',
                          'resnet50.layer1[0]', 'resnet50.layer1[1]', 'resnet50.layer1[2]',
                          'resnet50.layer2[0]', 'resnet50.layer2[1]', 'resnet50.layer2[2]', 'resnet50.layer2[3]',
                          'resnet50.layer3[0]', 'resnet50.layer3[1]', 'resnet50.layer3[2]', 'resnet50.layer3[3]',
                          'resnet50.layer4[0]', 'resnet50.layer4[1]', 'resnet50.layer4[2]']
            resnet50 = model
            activations_dict, _, total_flops, total_params = visualise_resnet.process_video_frame(video_name, frame, frame_number, all_layers, resnet50, device)

        elif layer_name == 'pool':
            visual_layer = 'resnet50.avgpool' # before avg_pool
            resnet50 = model
            activations_dict, _, total_flops, total_params = visualise_resnet_layer.process_video_frame(video_name, frame, frame_number, visual_layer, resnet50, device)

    elif network_name == 'vit':
        patch_size = 16
        activations_dict, _, total_flops, total_params = visualise_vit_layer.process_video_frame(video_name, frame, frame_number, model, patch_size, device)

    return activations_dict, total_flops, total_params


def process_video_feature(video_feature, network_name, layer_name):
    # initialize an empty list to store processed frames
    averaged_frames = []
    # iterate through each frame in the video_feature
    for frame in video_feature:
        frame_features = []

        if network_name == 'vit':
            # global mean and std
            global_mean = torch.mean(frame, dim=0)
            global_max = torch.max(frame, dim=0)[0]
            global_std = torch.std(frame, dim=0)
            # concatenate all pooling
            combined_features = torch.hstack([global_mean, global_max, global_std])
            frame_features.append(combined_features)

        elif network_name == 'resnet50':
            if layer_name == 'layerstack':
                # iterate through each layer in the current framex
                for layer_array in frame.values():
                    # calculate the mean along the specified axes (1 and 2) for each layer
                    layer_mean = torch.mean(layer_array, dim=(1, 2))
                    # append the calculated mean to the list for the current frame
                    frame_features.append(layer_mean)
            elif layer_name == 'pool':
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

def merge_fragments(diff_fragment, flow_fragment):
    alpha = 0.5
    merged_fragment = diff_fragment * alpha + flow_fragment * (1 - alpha)
    return merged_fragment

def concatenate_features(frame_feature, residual_feature):
    return torch.cat((frame_feature, residual_feature), dim=-1)
