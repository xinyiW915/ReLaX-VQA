import os
import scipy.io
import numpy as np
import pandas as pd
import os

def list_directory_contents(directory):
    if not os.path.exists(directory):
        print("Cannot find directory")
        return

    for root, dirs, files in os.walk(directory):
        files = sorted([file for file in files if file != '.DS_Store'])
        dirs = sorted([dir for dir in dirs if dir != '.DS_Store'])

        print(f"'{root}': {len(dirs)} subfolders and {len(files)} files")
        if dirs:
            print("  subfolders:")
            for dir in dirs:
                print(f"    - {dir}")
        if files:
            print("  files:")
            for file in files:
                print(f"    - {file}")
        print("\n")

def merge_mat_files(file_paths, output_file):
    merged_data = {}
    for file_path in file_paths:
        data = scipy.io.loadmat(file_path)
        for key, value in data.items():
            if key not in ['__header__', '__version__', '__globals__']:
                if key in merged_data:
                    try:
                        merged_data[key] = np.hstack((merged_data[key], value))
                        print(f"after merging {key}: new dimensions are {merged_data[key].shape}")
                    except ValueError as e:
                        print(f"error concatenating {key}: {e}")
                else:
                    merged_data[key] = value
                    print(f"loaded {key}: dimensions are {value.shape}")

    scipy.io.savemat(output_file, merged_data)
    print(f"data saved to {output_file} \n")

def get_merge_sample(file_path, merged_fragment_file, sample_fragment_file, col_threshold):
    merged_data = {}
    merged_prefix_data = {}
    print(file_path)
    data = scipy.io.loadmat(file_path)
    for key, value in data.items():
        if key not in ['__header__', '__version__', '__globals__']:
            if value.shape[1] > col_threshold:
                prefix_value = value[:, :col_threshold]
                sliced_value = value[:, col_threshold:]
                print(f"original feature dimension: {data[key].shape}")
            else:
                print(f"{key} less than {col_threshold} columns")
                continue
            # merged fragments
            if key in merged_data:
                try:
                    merged_data[key] = np.hstack((merged_data[key], sliced_value))
                    print(f"merged fragments {key} (MF): {merged_data[key].shape}")
                except ValueError as e:
                    print(f"error concatenating {key}: {e}")
            else:
                merged_data[key] = sliced_value
                print(f"merged fragments {key}: {sliced_value.shape}")

            # sampled fragments
            if key in merged_prefix_data:
                try:
                    merged_prefix_data[key] = np.hstack((merged_prefix_data[key], prefix_value))
                    print(f"sampled fragments {key} (keyframe fragment): {merged_prefix_data[key].shape}")
                except ValueError as e:
                    print(f"error concatenating {key}: {e}")
            else:
                merged_prefix_data[key] = prefix_value
                print(f"sampled fragments {key}: {prefix_value.shape}")

    # merged fragments
    scipy.io.savemat(merged_fragment_file, merged_data)
    print(f"merged fragments saved to {merged_fragment_file}")
    # sampled fragments
    sample_fragment_file = f"{sample_fragment_file}.mat"
    scipy.io.savemat(sample_fragment_file, merged_prefix_data)
    print(f"sampled fragments saved to {sample_fragment_file} \n")

def merge_and_split_chunks(file_paths, output_file, chunk_size):
    merged_data = {}
    for file_path in file_paths:
        data = scipy.io.loadmat(file_path)
        for key, value in data.items():
            if key not in ['__header__', '__version__', '__globals__']:
                if key in merged_data:
                    try:
                        merged_data[key] = np.hstack((merged_data[key], value))
                    except ValueError as e:
                        print(f"error concatenating {key}: {e}")
                else:
                    merged_data[key] = value
                    print(f"loaded {key}: dimensions are {value.shape}")

    for key, value in merged_data.items():
        num_rows = value.shape[0]
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            chunk_data = value[start:end, :]
            new_file_path = output_file.replace('.mat', f'_chunk_{start // chunk_size + 1}.mat')
            scipy.io.savemat(new_file_path, {key: chunk_data})
            print(f"Saved {new_file_path} with shape {chunk_data.shape}")

def merge_combined_datasets(file_paths, output_file, feature_keys):
    merged_data = None
    grey_df = pd.read_csv(f'metadata/greyscale_report/YOUTUBE_UGC_greyscale_metadata.csv')
    grey_indices = grey_df.iloc[:, 0].tolist()

    for file_path, feature_key in zip(file_paths, feature_keys):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        data = scipy.io.loadmat(file_path)
        if feature_key not in data:
            print(f"key '{feature_key}' not found in {file_path}")
            continue
        if feature_key == 'youtube_ugc':
            features = data[feature_key]
            features = np.delete(features, grey_indices, axis=0)
        else:
            features = data[feature_key]
        if merged_data is None:
            merged_data = features
            print(f"{feature_key} new dimension: {features.shape}")
        else:
            try:
                merged_data = np.vstack((merged_data, features))
                print(f"{feature_key} new dimension: {features.shape}")
            except ValueError as e:
                print(f"error concatenating features: {e}")
    if merged_data is not None:
        scipy.io.savemat(output_file, {'all_combined': merged_data})
        print(f"after merging: new dimensions are {merged_data.shape}")
        print(f"combined_datasets saved to {output_file}\n")
    else:
        print("no data to merge")

# for merge all combined csv
def load_and_preprocess(data_name, metadata_path, columns):
    metadata_name = f'{data_name.upper()}_metadata.csv'
    df = pd.read_csv(os.path.join(metadata_path, metadata_name), usecols=columns)
    print(f"loaded {data_name} with {len(df)} entries before filtering.")

    if data_name in ['konvid_1k', 'youtube_ugc']:
        mos = ((np.array(df['mos'].tolist()) - 1) * (99/4) + 1.0).tolist()
        # mos = (100 * (np.array(df['mos'].tolist()) - 1) / 4).tolist()
        df['mos'] = mos
        print(f"MOS values scaled for {data_name}.")

    if data_name == 'youtube_ugc':
        # grayscale videos, do not consider them for fair comparison
        grey_df = pd.read_csv(f'{metadata_path}/greyscale_report/{data_name.upper()}_greyscale_metadata.csv')
        grey_indices = grey_df.iloc[:, 0].tolist()
        df = df.drop(index=grey_indices).reset_index(drop=True)
        print(f"size of {data_name} after filtering: {len(df)} entries.")
    return df

def merge_csv_files(metadata_path, data_names, columns):
    all_dfs = []
    for name in data_names:
        df = load_and_preprocess(name, metadata_path, columns)
        all_dfs.append(df)
    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

if __name__ == "__main__":
    metadata_path = 'metadata/'
    data_names =  ['cvd_2014', 'konvid_1k', 'live_vqc', 'youtube_ugc']
    columns = ['vid', 'mos', 'width', 'height', 'pixfmt', 'framerate', 'nb_frames', 'bitdepth', 'bitrate']

    merged_data = merge_csv_files(metadata_path, data_names, columns)
    merged_data.to_csv('ALL_COMBINED_metadata.csv', index=False)
    print(f"Merged DataFrame size: {len(merged_data)} entries\n")


    feature_path = "features/layer_stack/original_features"
    # list_directory_contents(feature_path)

    feature_path_r = "features_residual/pool/original_features"
    # list_directory_contents(feature_path_r)
    feature_path_r_frag = "features_residual_frag/pool/original_features"
    # list_directory_contents(feature_path_r_frag)

    feature_path_rof = "features_residual_of/pool/original_features"
    # list_directory_contents(feature_path_rof)
    feature_path_rof_frag = "features_residual_of_frag/pool/original_features"
    # list_directory_contents(feature_path_rof_frag)

    feature_path_merged_frag = "features_merged_frag/"
    # list_directory_contents(feature_path_merged_frag)

    data_name = 'konvid_1k_abalation'

    if data_name == 'konvid_1k_abalation':
        data_name = 'konvid_1k'
        # ./ features /
        resnet50 = f"{feature_path}/resnet50_{data_name}_original_features.mat"
        vit = f"{feature_path}/vit_{data_name}_original_features.mat"

        # vgg16_vit = [vgg16, vit]
        # output_file = f"{feature_path}/vgg16_vit_{data_name}_original_features.mat"
        # merge_mat_files(vgg16_vit, output_file)

        # resnet50-layer_stack
        # resnet50-pool
        # vit-layer_stack
        resnet50_vit = [resnet50, vit]
        output_file = f"{feature_path}/resnet50_vit_{data_name}_original_features.mat"
        merge_mat_files(resnet50_vit, output_file)

        # ./features_residual/
        frame_diff_resnet50 = f"{feature_path_r}/resnet50_{data_name}_original_features.mat"
        frame_diff_vit = f"{feature_path_r}/vit_{data_name}_original_features.mat"
        # ./features_residual_frag/
        rf_frame_diff_resnet50 = f"{feature_path_r_frag}/resnet50_{data_name}_original_features.mat"
        rf_frame_diff_vit = f"{feature_path_r_frag}/vit_{data_name}_original_features.mat"

        # ./features_residual_of/
        # optical_flow_resnet50 missed
        optical_flow_resnet50 = f"{feature_path_rof}/resnet50_{data_name}_original_features.mat"
        optical_flow_vit = f"{feature_path_rof}/vit_{data_name}_original_features.mat"
        # ./features_residual_of_frag/
        rf_optical_flow_resnet50 = f"{feature_path_rof_frag}/resnet50_{data_name}_original_features.mat"
        rf_optical_flow_vit = f"{feature_path_rof_frag}/vit_{data_name}_original_features.mat"

        # ./features_merged_frag
        ori_rfrag_merged_resnet = f"{feature_path_merged_frag}layer_stack/original_features/resnet50_{data_name}_original_features.mat"
        ori_rfrag_merged_vit = f"{feature_path_merged_frag}pool/original_features/vit_{data_name}_original_features.mat"

        merged_fragment_resnet = f"{feature_path_merged_frag}pool/original_features/merged_fragment_resnet50_{data_name}_original_features.mat"
        merged_fragment_vit = f"{feature_path_merged_frag}pool/original_features/merged_fragment_vit_{data_name}_original_features.mat"
        sample_fragment_resnet = f"{feature_path_merged_frag}layer_stack/original_features/sample_fragment_resnet50_{data_name}_original_features.mat"
        sample_fragment_vit = f"{feature_path_merged_frag}pool/original_features/sample_fragment_vit_{data_name}_original_features.mat"

        get_merge_sample(ori_rfrag_merged_resnet, merged_fragment_resnet, sample_fragment_resnet, col_threshold=13120)
        get_merge_sample(ori_rfrag_merged_vit, merged_fragment_vit, sample_fragment_vit, col_threshold=2304)

        mf_resnet_vit = [merged_fragment_resnet, merged_fragment_vit]
        output_file = f"{feature_path_merged_frag}/pool/original_features/merged_fragment_resnet50_vit_{data_name}_original_features.mat"
        merge_mat_files(mf_resnet_vit, output_file)

    elif data_name == 'all_combined':
        datasets = ['cvd_2014', 'konvid_1k', 'live_vqc', 'youtube_ugc']
        combined_resnet50 = []
        combined_vit = []
        combined_resnet50_vit = []
        combined_resnet50_ori_rfrag_merged_resnet50 = []
        combined_resnet50_ori_rfrag_merged_vit = []
        combined_vit_ori_rfrag_merged_resnet50 = []
        combined_vit_ori_rfrag_merged_vit = []
        combined_resnet50_vit_ori_rfrag_merged_resnet50 = []
        combined_resnet50_vit_ori_rfrag_merged_vit = []
        combined_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit = []
        combined_resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit = []

        for dataset in datasets:
            # ./ features /
            resnet50 = f"{feature_path}/resnet50_{dataset}_original_features.mat"
            combined_resnet50.append(resnet50)

            vit = f"{feature_path}/vit_{dataset}_original_features.mat"
            combined_vit.append(vit)

            # ./ features /
            resnet50_vit = f"{feature_path}/resnet50_vit_{dataset}_original_features.mat"
            combined_resnet50_vit.append(resnet50_vit)

            # ./features_merged_frag
            resnet50_ori_rfrag_merged_resnet50 = f"{feature_path_merged_frag}/layer_stack/original_features/resnet50_ori_rfrag_merged_resnet50_{dataset}_original_features.mat"
            combined_resnet50_ori_rfrag_merged_resnet50.append(resnet50_ori_rfrag_merged_resnet50)
            resnet50_ori_rfrag_merged_vit = f"{feature_path_merged_frag}/pool/original_features/resnet50_ori_rfrag_merged_vit_{dataset}_original_features.mat"
            combined_resnet50_ori_rfrag_merged_vit.append(resnet50_ori_rfrag_merged_vit)
            vit_ori_rfrag_merged_resnet50 = f"{feature_path_merged_frag}/layer_stack/original_features/vit_ori_rfrag_merged_resnet50_{dataset}_original_features.mat"
            combined_vit_ori_rfrag_merged_resnet50.append(vit_ori_rfrag_merged_resnet50)
            vit_ori_rfrag_merged_vit = f"{feature_path_merged_frag}/pool/original_features/vit_ori_rfrag_merged_vit_{dataset}_original_features.mat"
            combined_vit_ori_rfrag_merged_vit.append(vit_ori_rfrag_merged_vit)
            resnet50_vit_ori_rfrag_merged_resnet50 = f"{feature_path_merged_frag}/layer_stack/original_features/resnet50_vit_ori_rfrag_merged_resnet50_{dataset}_original_features.mat"
            combined_resnet50_vit_ori_rfrag_merged_resnet50.append(resnet50_vit_ori_rfrag_merged_resnet50)
            resnet50_vit_ori_rfrag_merged_vit = f"{feature_path_merged_frag}/pool/original_features/resnet50_vit_ori_rfrag_merged_vit_{dataset}_original_features.mat"
            combined_resnet50_vit_ori_rfrag_merged_vit.append(resnet50_vit_ori_rfrag_merged_vit)
            ori_rfrag_merged_resnet50_ori_rfrag_merged_vit = f"{feature_path_merged_frag}/pool/original_features/ori_rfrag_merged_resnet50_ori_rfrag_merged_vit_{dataset}_original_features.mat"
            combined_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit.append(ori_rfrag_merged_resnet50_ori_rfrag_merged_vit)
            resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit = f"{feature_path_merged_frag}/pool/original_features/resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit_{dataset}_original_features.mat"
            combined_resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit.append(resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit)

        feature_combined_path = "features_combined/layer_stack/original_features"
        output_file = f"{feature_combined_path}/resnet50_{data_name}_original_features.mat"
        merge_combined_datasets(combined_resnet50, output_file, datasets)
        feature_combined_path = "features_combined/layer_stack/original_features"
        output_file = f"{feature_combined_path}/vit_{data_name}_original_features.mat"
        merge_combined_datasets(combined_vit, output_file, datasets)
        feature_combined_path = "features_combined/layer_stack/original_features"
        output_file = f"{feature_combined_path}/resnet50_vit_{data_name}_original_features.mat"
        merge_combined_datasets(combined_resnet50_vit, output_file, datasets)

        feature_combined_path = "features_combined/layer_stack/original_features"
        output_file = f"{feature_combined_path}/resnet50_ori_rfrag_merged_resnet50_{data_name}_original_features.mat"
        merge_combined_datasets(combined_resnet50_ori_rfrag_merged_resnet50, output_file, datasets)
        feature_combined_path = "features_combined/pool/original_features"
        output_file = f"{feature_combined_path}/resnet50_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_combined_datasets(combined_resnet50_ori_rfrag_merged_vit, output_file, datasets)

        feature_combined_path = "features_combined/layer_stack/original_features"
        output_file = f"{feature_combined_path}/vit_ori_rfrag_merged_resnet50_{data_name}_original_features.mat"
        merge_combined_datasets(combined_vit_ori_rfrag_merged_resnet50, output_file, datasets)
        feature_combined_path = "features_combined/pool/original_features"
        output_file = f"{feature_combined_path}/vit_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_combined_datasets(combined_vit_ori_rfrag_merged_vit, output_file, datasets)

        feature_combined_path = "features_combined/layer_stack/original_features"
        output_file = f"{feature_combined_path}/resnet50_vit_ori_rfrag_merged_resnet50_{data_name}_original_features.mat"
        merge_combined_datasets(combined_resnet50_vit_ori_rfrag_merged_resnet50, output_file, datasets)
        feature_combined_path = "features_combined/pool/original_features"
        output_file = f"{feature_combined_path}/resnet50_vit_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_combined_datasets(combined_resnet50_vit_ori_rfrag_merged_vit, output_file, datasets)

        feature_combined_path = "features_combined/pool/original_features"
        output_file = f"{feature_combined_path}/ori_rfrag_merged_resnet50_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_combined_datasets(combined_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit, output_file, datasets)
        feature_combined_path = "features_combined/pool/original_features"
        output_file = f"{feature_combined_path}/resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_combined_datasets(combined_resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit, output_file, datasets)

    elif data_name == 'lsvq_train' or data_name == 'lsvq_test' or data_name == 'lsvq_test_1080P':
        # ./ features /
        vgg16 = f"{feature_path}/vgg16_{data_name}_original_features.mat"
        resnet50 = f"{feature_path}/resnet50_{data_name}_original_features.mat"
        vit = f"{feature_path}/vit_{data_name}_original_features.mat"

        resnet50_vit = [resnet50, vit]
        output_file = f"{feature_path}/resnet50_vit_{data_name}_original_features.mat"
        merge_mat_files(resnet50_vit, output_file)

        # ./features_merged_frag
        ori_rfrag_merged_resnet = f"{feature_path_merged_frag}/layer_stack/original_features/resnet50_{data_name}_original_features.mat"
        ori_rfrag_merged_vit = f"{feature_path_merged_frag}/pool/original_features/vit_{data_name}_original_features.mat"

        resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit = [resnet50, vit, ori_rfrag_merged_resnet, ori_rfrag_merged_vit]
        output_file = f"{feature_path_merged_frag}/pool/original_features/resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        if data_name == 'lsvq_train':
            merge_and_split_chunks(resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit , output_file, 10000)
        else:
            merge_mat_files(resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit, output_file)

    else:
        # ./ features /
        vgg16 = f"{feature_path}/vgg16_{data_name}_original_features.mat"     
        resnet50 = f"{feature_path}/resnet50_{data_name}_original_features.mat"
        vit = f"{feature_path}/vit_{data_name}_original_features.mat"

        # vgg16_vit = [vgg16, vit]
        # output_file = f"{feature_path}/vgg16_vit_{data_name}_original_features.mat"
        # merge_mat_files(vgg16_vit, output_file)
        
        # resnet50-layer_stack
        # resnet50-pool
        # vit-layer_stack
        resnet50_vit = [resnet50, vit]
        output_file = f"{feature_path}/resnet50_vit_{data_name}_original_features.mat"
        merge_mat_files(resnet50_vit, output_file)

        # ./features_merged_frag
        ori_rfrag_merged_resnet = f"{feature_path_merged_frag}/layer_stack/original_features/resnet50_{data_name}_original_features.mat"
        ori_rfrag_merged_vit = f"{feature_path_merged_frag}/pool/original_features/vit_{data_name}_original_features.mat"

        resnet50_ori_rfrag_merged_resnet50 = [resnet50, ori_rfrag_merged_resnet]
        output_file = f"{feature_path_merged_frag}/layer_stack/original_features/resnet50_ori_rfrag_merged_resnet50_{data_name}_original_features.mat"
        merge_mat_files(resnet50_ori_rfrag_merged_resnet50, output_file)
        resnet50_ori_rfrag_merged_vit = [resnet50, ori_rfrag_merged_vit]
        output_file = f"{feature_path_merged_frag}/pool/original_features/resnet50_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_mat_files(resnet50_ori_rfrag_merged_vit, output_file)

        vit_ori_rfrag_merged_resnet50 = [vit, ori_rfrag_merged_resnet]
        output_file = f"{feature_path_merged_frag}/layer_stack/original_features/vit_ori_rfrag_merged_resnet50_{data_name}_original_features.mat"
        merge_mat_files(vit_ori_rfrag_merged_resnet50, output_file)
        vit_ori_rfrag_merged_vit = [vit, ori_rfrag_merged_vit]
        output_file = f"{feature_path_merged_frag}/pool/original_features/vit_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_mat_files(vit_ori_rfrag_merged_vit, output_file)

        resnet50_vit_ori_rfrag_merged_resnet50 = [resnet50, vit, ori_rfrag_merged_resnet]
        output_file = f"{feature_path_merged_frag}/layer_stack/original_features/resnet50_vit_ori_rfrag_merged_resnet50_{data_name}_original_features.mat"
        merge_mat_files(resnet50_vit_ori_rfrag_merged_resnet50, output_file)
        resnet50_vit_ori_rfrag_merged_vit = [resnet50, vit, ori_rfrag_merged_vit]
        output_file = f"{feature_path_merged_frag}/pool/original_features/resnet50_vit_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_mat_files(resnet50_vit_ori_rfrag_merged_vit, output_file)

        ori_rfrag_merged_resnet50_ori_rfrag_merged_vit = [ori_rfrag_merged_resnet, ori_rfrag_merged_vit]
        output_file = f"{feature_path_merged_frag}/pool/original_features/ori_rfrag_merged_resnet50_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_mat_files(ori_rfrag_merged_resnet50_ori_rfrag_merged_vit, output_file)
        resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit = [resnet50, vit, ori_rfrag_merged_resnet, ori_rfrag_merged_vit]
        output_file = f"{feature_path_merged_frag}/pool/original_features/resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit_{data_name}_original_features.mat"
        merge_mat_files(resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit, output_file)
