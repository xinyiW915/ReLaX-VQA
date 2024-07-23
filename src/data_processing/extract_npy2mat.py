import os
import numpy as np
import pandas as pd
import scipy.io

def load_metadata(video_type):
    # NR: original
    original_metadata_files = {
        'original_konvid_1k': 'KONVID_1K_metadata.csv',
        f'original_youtube_ugc_{resolution}': f'YOUTUBE_UGC_{resolution}_metadata.csv',
        'original_lsvq_train': 'LSVQ_train_metadata.csv',
        'original_lsvq_test': 'LSVQ_test_metadata.csv',
        'original_lsvq_test_1080P': 'LSVQ_test_1080P_metadata.csv',
        'original_live_vqc': 'LIVE_VQC_metadata.csv',
        'original_cvd_2014': 'CVD_2014_metadata.csv',
        'original_live_qualcomm': 'LIVE_Qualcomm_metadata.csv'
    }
    if video_type in original_metadata_files:
        return pd.read_csv(f"../../metadata/{original_metadata_files[video_type]}")
    else:
        raise ValueError("Invalid video_type")

def get_video_paths(network_name, video_type, videodata, i):
    if video_type == f'original_youtube_ugc_{resolution}':
        video_name = videodata['vid'][i]
        qp = 'original'
        feature_name = f"{network_name}_feature_map_{qp}_{resolution}"

    else:
        video_name = videodata['vid'][i]
        print(video_name)
        if 'original' in video_type:
            qp = 'original'
            feature_name = f"{network_name}_feature_map_{qp}"

    npy_name = f'video_{i+1}_{feature_name}.npy'
    return npy_name

def get_file_path(features_path, npy_name):
    return os.path.join(features_path, npy_name)

def get_feature_path(data_name, compressed_type, resolution, network_name, layer_name, residual, residual_name):
    if residual:
        if residual_name == 'frame_diff':
            base_path = "../../features_residual"
        elif residual_name == 'frame_diff_frag':
            base_path = "../../features_residual_frag"
        elif residual_name == 'optical_flow':
            base_path = "../../features_residual_of"
        elif residual_name == 'optical_flow_frag':
            base_path = "../../features_residual_of_frag"
        elif residual_name == 'frame_merged_frag':
            base_path = "../../features_merged_frag"
    else:
        base_path = "../../features"
    if data_name == 'youtube_ugc':
        features_path = f"{base_path}/{network_name}/{layer_name}/resolution_ugc/{compressed_type}_{resolution}/"
    else:
        features_path = f"{base_path}/{network_name}/{layer_name}/{data_name}/{compressed_type}/"
    return features_path

def save_features(data_name, empty_matrix, network_name, compressed_type, resolution, layer_name, residual, residual_name):
    if residual:
        if residual_name == 'frame_diff':
            base_path = "../../features_residual"
        elif residual_name == 'frame_diff_frag':
            base_path = "../../features_residual_frag"
        elif residual_name == 'optical_flow':
            base_path = "../../features_residual_of"
        elif residual_name == 'optical_flow_frag':
            base_path = "../../features_residual_of_frag"
        elif residual_name == 'frame_merged_frag':
            base_path = "../../features_merged_frag"
    else:
        base_path = "../../features"
    mat_file_path = f'{base_path}/{layer_name}/original_features/'
    os.makedirs(mat_file_path, exist_ok=True)

    if data_name == 'youtube_ugc':
        mat_file_name = f'{mat_file_path}{network_name}_{data_name}_{compressed_type}_{resolution}_features.mat'
        scipy.io.savemat(mat_file_name, {f'{data_name}_{resolution}': empty_matrix})
    else:
        mat_file_name = f'{mat_file_path}{network_name}_{data_name}_{compressed_type}_features.mat'
        scipy.io.savemat(mat_file_name, {data_name: empty_matrix})
    print(f'Successfully created {mat_file_name}')


if __name__ == '__main__':
    resolution = '360P'
    residual = True
    residual_name = 'frame_diff_frag' # 'frame_diff', 'optical_flow', 'frame_diff_frag', 'optical_flow_frag', 'frame_merged_frag'
    network_name = "resnet50"
    if residual:
        if network_name == 'vit':
            layer_name = 'pool'
        else:
            # layer_name = 'layer_stack'
            layer_name = 'pool'
    else:
        # layer_name = 'layer_stack'
        layer_name = 'pool'
    data_name = "konvid_1k"
    compressed_type = 'original'

    features_path = get_feature_path(data_name, compressed_type, resolution, network_name, layer_name, residual, residual_name)
    print(features_path)

    if data_name == 'youtube_ugc':
        video_type = f'{compressed_type}_{data_name}_{resolution}'
    else:
        video_type = f'{compressed_type}_{data_name}'
    print(video_type)

    video_data = load_metadata(video_type)
    print(len(video_data))

    for i in range(len(video_data)):
        npy_name = get_video_paths(network_name, video_type, video_data, i)
        print(npy_name)
        npy_file_path = get_file_path(features_path, npy_name)
        data = np.load(npy_file_path)
        average_data = np.mean(data, axis=0)

        if i == 0:
            empty_matrix = np.zeros((len(video_data),) + average_data.shape)
        empty_matrix[i] = average_data

    # save feature mat file
    print(empty_matrix.shape)
    save_features(data_name, empty_matrix, network_name, compressed_type, resolution, layer_name, residual, residual_name)

