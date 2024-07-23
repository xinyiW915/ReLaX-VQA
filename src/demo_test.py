import argparse
import numpy as np
import os
import glob
import shutil
import cv2
import torch
import torch.nn as nn
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from main_layer_stack import process_video_feature as pvf_layer_stack
from main_layer_stack import get_deep_feature as gdf_layer_stack
from main_fragment_pool import process_video_feature as pvf_fragment_pool
from main_fragment_pool import get_deep_feature as gdf_fragment_pool
from main_fragment_layerstack import process_video_feature as pvf_fragment_layerstack
from main_fragment_layerstack import get_deep_feature as gdf_fragment_layerstack

from main_fragment_layerstack import process_patches, get_original_frame_patches, flow_to_rgb, merge_fragments, concatenate_features
from extractor.vf_extract import process_video, process_video_residual
from model_regression import Mlp


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

def preprocess_data(X, y):
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
    X = imp.transform(X)

    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    if y.size > 0:
        y = y.reshape(-1, 1).squeeze()

    return X, y, imp, scaler

def evaluate_video_quality(config):
    device = config['device']
    model_name = config['model_name']
    layer_name = config['layer_name']
    select_criteria = config['select_criteria']
    train_data_name = config['train_data_name']
    is_finetune = config['is_finetune']
    save_path = config['save_path']

    video_type = config['video_type']
    video_name = config['video_name']
    qp = config['qp']
    video_width = config['video_width']
    video_height = config['video_height']
    pixfmt = config['pixfmt']
    framerate = config['framerate']

    sampled_frame_path = os.path.join("../video_sampled_frame/original_sampled_frame/", "test_sampled_frames")
    sampled_fragment_path = os.path.join("../video_sampled_frame/original_sampled_frame/", "test_sampled_fragment")
    if video_type == 'youtube_ugc':
        video_path = f'../ugc_original_videos/{video_name}.mkv'
    else:
        video_path = f'../ugc_original_videos/{video_name}.mp4'

    # sampled video frames
    process_video(video_type, video_name, int(framerate / 2), video_path, sampled_frame_path, video_width, video_height, pixfmt, framerate)
    # get ResNet50 layer-stacking features and ViT pooling features
    all_frame_activations_resnet = []
    all_frame_activations_vit = []
    image_paths = glob.glob(os.path.join(sampled_frame_path, f'{video_name}_*.png'))
    for image_path in image_paths:
        png_path_resnet, npy_path_resnet, frame_npy = gdf_layer_stack('resnet50', video_name, image_path, qp)
        all_frame_activations_resnet.append(frame_npy)
        png_path_vit, npy_path_vit, frame_npy = gdf_layer_stack('vit', video_name, image_path, qp)
        all_frame_activations_vit.append(frame_npy)
    averaged_frame_npy_resnet = pvf_layer_stack(all_frame_activations_resnet, 'resnet50')
    averaged_frame_npy_vit = pvf_layer_stack(all_frame_activations_vit, 'vit')
    print("ResNet50 layer-stacking feature shape:", averaged_frame_npy_resnet.shape)
    print("ViT pooling feautre shape:", averaged_frame_npy_vit.shape)

    # residual video frames
    process_video_residual(video_type, video_name, int(framerate / 2), video_path, sampled_fragment_path, video_width, video_height, pixfmt, framerate)
    original_frame_paths = sorted([path for path in glob.glob(os.path.join(sampled_fragment_path, f'{video_name}_*.png'))
                                   if '_next' not in os.path.basename(path)],
                                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
    next_frame_paths = sorted([path for path in glob.glob(os.path.join(sampled_fragment_path, f'{video_name}_*_next.png'))
                               if '_next' in os.path.basename(path)],
                              key=lambda x: int(x.split('_')[-2]))
    # get fragments ResNet50 features and ViT features
    all_frame_activations_original_resnet = []
    all_frame_activations_residual_resnet = []
    all_frame_activations_original_vit = []
    all_frame_activations_residual_vit = []
    for original_path, next_path in zip(original_frame_paths, next_frame_paths):
        # compute residual
        img_original = cv2.imread(original_path)
        img_next = cv2.imread(next_path)
        target_size = 224
        patch_size = 16
        top_n = int((target_size / patch_size) * (target_size / patch_size))

        # Frame Differencing
        residual = cv2.absdiff(img_next, img_original)
        # residual_path = original_path.replace('.png', '_residual.png')
        # cv2.imwrite(residual_path, residual)
        residual_frag_path, diff_frag, positions = process_patches(original_path, 'frame_diff', residual, patch_size, target_size, top_n)
        # original frame fragment
        original_patches = get_original_frame_patches(img_original, positions, patch_size, target_size)
        original_frag_path = original_path.replace('.png', '_ori_frag.png')
        cv2.imwrite(original_frag_path, original_patches)

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        residual_of_rgb = flow_to_rgb(flow)
        # residual_of_path = original_path.replace('.png', '_residual_of.png')
        # cv2.imwrite(residual_of_path, residual_of_rgb)
        residual_of_frag_path, flow_frag, _ = process_patches(original_path, 'optical_flow', residual_of_rgb, patch_size, target_size, top_n)

        # diff_frag = cv2.imread(residual_frag_path)
        # flow_frag = cv2.imread(residual_of_frag_path)
        merged_frag = merge_fragments(diff_frag, flow_frag)
        merged_frag_path = original_path.replace('.png', '_residual_merged_frag.png')
        cv2.imwrite(merged_frag_path, merged_frag)

        _, _, ori_frag_npy_resnet = gdf_fragment_layerstack('resnet50', video_name, original_frag_path, qp, 'layer_stack')
        _, _, merged_frag_npy_resnet = gdf_fragment_layerstack('resnet50', video_name, merged_frag_path, qp, 'pool')
        _, _, ori_frag_npy_vit = gdf_fragment_pool('vit', video_name, original_frag_path, qp, 'pool')
        _, _, merged_frag_npy_vit = gdf_fragment_pool('vit', video_name, merged_frag_path, qp, 'pool')
        try:
            os.remove(next_path)
            # print(f"Deleted: {next_path}")
        except OSError as e:
            print(f"Error: {e.strerror}, file: {e.filename}")
        # fragments ResNet50 feature combined
        all_frame_activations_original_resnet.append(ori_frag_npy_resnet)
        all_frame_activations_residual_resnet.append(merged_frag_npy_resnet)
        # fragments ViT feature combined
        all_frame_activations_original_vit.append(ori_frag_npy_vit)
        all_frame_activations_residual_vit.append(merged_frag_npy_vit)
    averaged_frame_npy_original_resnet = pvf_fragment_layerstack(all_frame_activations_original_resnet, 'resnet50', 'layer_stack')
    averaged_frame_npy_residual_resnet = pvf_fragment_layerstack(all_frame_activations_residual_resnet, 'resnet50', 'pool')
    print("Sampled fragments ResNet50 features shape:", averaged_frame_npy_original_resnet.shape)
    print("Residual fragments ResNet50 features shape:", averaged_frame_npy_residual_resnet.shape)
    averaged_combined_feature_resnet = concatenate_features(averaged_frame_npy_original_resnet, averaged_frame_npy_residual_resnet)
    averaged_frame_npy_original_vit = pvf_fragment_pool(all_frame_activations_original_vit, 'vit')
    averaged_frame_npy_residual_vit = pvf_fragment_pool(all_frame_activations_residual_vit, 'vit')
    print("Sampled fragments ViT features shape:", averaged_frame_npy_original_vit.shape)
    print("Residual fragments ResNet50 features shape:", averaged_frame_npy_residual_vit.shape)
    averaged_combined_feature_vit = concatenate_features(averaged_frame_npy_original_vit, averaged_frame_npy_residual_vit)
    # remove tmp folders
    shutil.rmtree(png_path_resnet)
    shutil.rmtree(npy_path_resnet)
    shutil.rmtree(png_path_vit)
    shutil.rmtree(npy_path_vit)
    shutil.rmtree(sampled_frame_path)
    shutil.rmtree(sampled_fragment_path)

    # concatenate features
    resnet50 = np.mean(averaged_frame_npy_resnet, axis=0)
    vit = np.mean(averaged_frame_npy_vit, axis=0)
    fragment_resnet50 = np.mean(averaged_combined_feature_resnet, axis=0)
    fragment_vit = np.mean(averaged_combined_feature_vit, axis=0)
    combined_features = np.concatenate([resnet50, vit, fragment_resnet50, fragment_vit])

    imputer = load(f'{save_path}/scaler/{video_type}_imputer.pkl')
    scaler = load(f'{save_path}/scaler/{video_type}_scaler.pkl')
    X_test_transformed = imputer.transform(combined_features.reshape(1, -1))
    X_test_scaled = scaler.transform(X_test_transformed)
    feature_tensor = torch.tensor(X_test_scaled, dtype=torch.float).to(device)

    network_name = 'relaxvqa'
    input_features = combined_features.shape[0]
    model = Mlp(input_features=input_features, out_features=1, drop_rate=0.2, act_layer=nn.GELU)
    model = model.to(device)
    if is_finetune  == True:
        model_path = os.path.join(save_path, f"fine_tune_model/{video_type}_{network_name}_{select_criteria}_fine_tuned_model.pth")
    else:
        model_path = os.path.join(save_path, f"{train_data_name}_{network_name}_{select_criteria}_trained_median_model_param_onLSVQ_TEST.pth")
    print(model_path)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    if device == 'cpu':
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(model_path)

    fixed_state_dict = fix_state_dict(state_dict)
    try:
        model.load_state_dict(fixed_state_dict)
    except RuntimeError as e:
        print(e)

    # evaluation for test video
    model.eval()
    with torch.no_grad():
        prediction = model(feature_tensor)
        # return prediction.item()
        # rescale from 0-100 to 1-5
        if is_finetune == True:
            return prediction.item()
        else:
            if video_type == 'youtube_ugc' or video_type == 'konvid_1k':
                scaled_prediction = (prediction.item() / 100) * 4 + 1
                # scaled_prediction = ((prediction.item() - 1) / (99 / 4) + 1.0)
                return scaled_prediction
            else:
                return prediction.item()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default=device, help='device')
    parser.add_argument('-model_name', type=str, default='Mlp', help='Name of the regression model')
    parser.add_argument('-layer_name', type=str, default='pool', help='Name of the layer')
    parser.add_argument('-select_criteria', type=str, default='byrmse', help='Selection criteria')
    parser.add_argument('-train_data_name', type=str, default='lsvq_train', help='Name of the training data')
    parser.add_argument('-is_finetune', type=str, default=False, help='with or without finetune')
    parser.add_argument('-save_path', type=str, default='../model/', help='Path to save models')
    parser.add_argument('-video_type', type=str, default='youtube_ugc', help='Type of video')
    parser.add_argument('-video_name', type=str, default='TelevisionClip_1080P-68c6', help='Name of the video') # TelevisionClip_1080P-68c6, Sports_2160P-0455, 5636101558, A014
    parser.add_argument('-qp', type=str, default='original', help='QP level')
    parser.add_argument('-video_width', type=int, default=960, help='Width of the video') # 1920, 3840, 960, 1920
    parser.add_argument('-video_height', type=int, default=540, help='Height of the video') # 1080, 2160, 540, 1080
    parser.add_argument('-pixfmt', type=str, default='yuv420p', help='Pixel format of the video')
    parser.add_argument('-framerate', type=float, default=24, help='Frame rate of the video') # 25, 29.97002997, 24, 29.9800133244504

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    config = vars(args)

    quality_prediction = evaluate_video_quality(config)
    print("Predicted Quality Score:", quality_prediction)

