import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
import logging

# cross_dataset
def process_cross_dataset(train_data_name, test_data_name, metadata_path, feature_path, layer_name, network_name):
    metadata_name1 = f"{train_data_name.replace('_all', '').upper()}_metadata.csv"
    metadata_name2 = f"{test_data_name.replace('_all', '').upper()}_metadata.csv"
    # load CSV data
    train_df = pd.read_csv(f'{metadata_path}/{metadata_name1}')
    test_df = pd.read_csv(f'{metadata_path}/{metadata_name2}')

    # grayscale videos, do not consider them for fair comparison
    grey_df_train = pd.read_csv(f"{metadata_path}/greyscale_report/{train_data_name.replace('_all', '').upper()}_greyscale_metadata.csv")
    grey_df_test = pd.read_csv(f"{metadata_path}/greyscale_report/{test_data_name.replace('_all', '').upper()}_greyscale_metadata.csv")
    grey_indices_train = grey_df_train.iloc[:, 0].tolist()
    grey_indices_test = grey_df_test.iloc[:, 0].tolist()
    train_df = train_df.drop(index=grey_indices_train).reset_index(drop=True)
    test_df = test_df.drop(index=grey_indices_test).reset_index(drop=True)

    # split videonames into train and test sets
    train_vids = train_df.iloc[:, 0]
    test_vids = test_df.iloc[:, 0]
    # logging.info(test_vids)

    # scores (1-100) map to 1-5
    train_scores = train_df['mos'].tolist()
    test_scores = test_df['mos'].tolist()
    if train_data_name == 'konvid_1k_all' or train_data_name == 'youtube_ugc_all':
        train_mos_list = ((np.array(train_scores) - 1) * (99/4) + 1.0).tolist()
    else:
        train_mos_list = train_scores
    if test_data_name == 'konvid_1k_all' or test_data_name == 'youtube_ugc_all':
        test_mos_list = ((np.array(test_scores) - 1) * (99/4) + 1.0).tolist()
    else:
        test_mos_list = test_scores

    # reorder columns
    sorted_train_df = pd.DataFrame({'vid': train_df['vid'],  'framerate': train_df['framerate'], 'MOS': train_mos_list, 'MOS_raw': train_df['mos']})
    sorted_test_df = pd.DataFrame({'vid': test_df['vid'], 'framerate': test_df['framerate'], 'MOS': test_mos_list, 'MOS_raw': test_df['mos']})

    # use indices from the train and test DataFrames to split features
    train_data = loadmat(f"{feature_path}{layer_name}/original_features/{network_name}_{train_data_name.replace('_all', '')}_original_features.mat")
    test_data = loadmat(f"{feature_path}{layer_name}/original_features/{network_name}_{test_data_name.replace('_all', '')}_original_features.mat")
    train_features = train_data[f"{train_data_name.replace('_all', '')}"]
    test_features = test_data[f"{test_data_name.replace('_all', '')}"]
    train_features = np.delete(train_features, grey_indices_train, axis=0)
    test_features = np.delete(test_features, grey_indices_test, axis=0)

    # save the files
    sorted_train_df.to_csv(f'{metadata_path}mos_files/{train_data_name}_MOS_train.csv', index=False)
    sorted_test_df.to_csv(f'{metadata_path}mos_files/{test_data_name}_MOS_test.csv', index=False)
    savemat(f'{feature_path}{layer_name}/{network_name}_{train_data_name}_cross_train_features.mat', {f'{train_data_name}_train_features': train_features})
    savemat(f'{feature_path}{layer_name}/{network_name}_{test_data_name}_cross_test_features.mat', {f'{test_data_name}_test_features': test_features})

    return train_features, test_features, test_vids

#NR: original
def process_lsvq(train_data_name, test_data_name, metadata_path, feature_path, layer_name, network_name):
    train_df = pd.read_csv(f'{metadata_path}/{train_data_name.upper()}_metadata.csv')
    test_df = pd.read_csv(f'{metadata_path}/{test_data_name.upper()}_metadata.csv')

    # grayscale videos, do not consider them for fair comparison
    grey_df_train = pd.read_csv(f'{metadata_path}/greyscale_report/{train_data_name.upper()}_greyscale_metadata.csv')
    grey_df_test = pd.read_csv(f'{metadata_path}/greyscale_report/{test_data_name.upper()}_greyscale_metadata.csv')
    grey_indices_train = grey_df_train.iloc[:, 0].tolist()
    grey_indices_test = grey_df_test.iloc[:, 0].tolist()
    train_df = train_df.drop(index=grey_indices_train).reset_index(drop=True)
    test_df = test_df.drop(index=grey_indices_test).reset_index(drop=True)
    test_vids = test_df['vid']
    # logging.info(test_vids)

    # mos scores
    train_scores = train_df['mos'].tolist()
    test_scores = test_df['mos'].tolist()
    train_mos_list = train_scores
    test_mos_list = test_scores

    # reorder columns
    sorted_train_df = pd.DataFrame({'vid': train_df['vid'],  'framerate': train_df['framerate'], 'MOS': train_mos_list, 'MOS_raw': train_df['mos']})
    sorted_test_df = pd.DataFrame({'vid': test_df['vid'], 'framerate': test_df['framerate'], 'MOS': test_mos_list, 'MOS_raw': test_df['mos']})

    # use indices from the train and test DataFrames to split features
    train_data_chunk_1 = loadmat(f'{feature_path}{layer_name}/original_features/{network_name}_{train_data_name}_original_features_chunk_1.mat')[f'{train_data_name}']
    train_data_chunk_2 = loadmat(f'{feature_path}{layer_name}/original_features/{network_name}_{train_data_name}_original_features_chunk_2.mat')[f'{train_data_name}']
    train_data_chunk_3 = loadmat(f'{feature_path}{layer_name}/original_features/{network_name}_{train_data_name}_original_features_chunk_3.mat')[f'{train_data_name}']
    merged_train_data = np.vstack((train_data_chunk_1, train_data_chunk_2, train_data_chunk_3))
    print(f"loaded {train_data_name}: dimensions are {merged_train_data.shape}")
    train_features = merged_train_data

    test_data = loadmat(f'{feature_path}{layer_name}/original_features/{network_name}_{test_data_name}_original_features.mat')
    test_features = test_data[f'{test_data_name}']
    train_features = np.delete(train_features, grey_indices_train, axis=0)
    test_features = np.delete(test_features, grey_indices_test, axis=0)
    print(len(train_features))
    print(len(test_features))

    # save the files
    sorted_train_df.to_csv(f'{metadata_path}mos_files/{train_data_name}_MOS_train.csv', index=False)
    sorted_test_df.to_csv(f'{metadata_path}mos_files/{train_data_name}_MOS_test.csv', index=False)
    # savemat(f'{feature_path}{layer_name}/{network_name}_{train_data_name}_original_train_features.mat', {f'{train_data_name}_train_features': train_features})
    # savemat(f'{feature_path}{layer_name}/{network_name}_{train_data_name}_original_test_features.mat', {f'{train_data_name}_test_features': test_features})

    return train_features, test_features, test_vids

def process_other(data_name, test_size, random_state, metadata_path, feature_path, layer_name, network_name):
    metadata_name = f'{data_name.upper()}_metadata.csv'
    # load CSV data
    df = pd.read_csv(f'{metadata_path}/{metadata_name}')

    if data_name == 'youtube_ugc':
        # grayscale videos, do not consider them for fair comparison
        grey_df = pd.read_csv(f'{metadata_path}/greyscale_report/{data_name.upper()}_greyscale_metadata.csv')
        grey_indices = grey_df.iloc[:, 0].tolist()
        df = df.drop(index=grey_indices).reset_index(drop=True)

    # get unique vids
    unique_vids = df['vid'].unique()

    # split videonames into train and test sets
    train_vids, test_vids = train_test_split(unique_vids, test_size=test_size, random_state=random_state)
    # logging.info(test_vids)

    # split all_dfs into train and test based on vids
    train_df = df[df['vid'].isin(train_vids)]
    test_df = df[df['vid'].isin(test_vids)]

    # mos scores
    train_scores = train_df['mos'].tolist()
    test_scores = test_df['mos'].tolist()
    train_mos_list = train_scores
    test_mos_list = test_scores

    # reorder columns
    sorted_train_df = pd.DataFrame({'vid': train_df['vid'],  'framerate': train_df['framerate'], 'MOS': train_mos_list, 'MOS_raw': train_df['mos']})
    sorted_test_df = pd.DataFrame({'vid': test_df['vid'], 'framerate': test_df['framerate'], 'MOS': test_mos_list, 'MOS_raw': test_df['mos']})

    # use indices from the train and test DataFrames to split features
    data = loadmat(f'{feature_path}{layer_name}/original_features/{network_name}_{data_name}_original_features.mat')
    features = data[f'{data_name}']
    if data_name == 'youtube_ugc':
        features = np.delete(features, grey_indices, axis=0)
    train_features = features[train_df.index]
    test_features = features[test_df.index]

    # save the files
    sorted_train_df.to_csv(f'{metadata_path}mos_files/{data_name}_MOS_train.csv', index=False)
    sorted_test_df.to_csv(f'{metadata_path}mos_files/{data_name}_MOS_test.csv', index=False)
    savemat(f'{feature_path}{layer_name}/{network_name}_{data_name}_original_train_features.mat', {f'{data_name}_train_features': train_features})
    savemat(f'{feature_path}{layer_name}/{network_name}_{data_name}_original_test_features.mat', {f'{data_name}_test_features': test_features})

    return train_features, test_features, test_vids


if __name__ == '__main__':
    layer_name = 'layer_stack'
    network_name = 'vgg16'

    data_name = "youtube_ugc"
    metadata_path = '../../metadata/'
    feature_path = '../../features/'

    # train test split
    test_size = 0.2
    random_state = None

    if data_name == 'lsvq_train':
        test_data_name = 'lsvq_test'
        process_lsvq(data_name, test_data_name, metadata_path, feature_path, layer_name, network_name)

    elif data_name == 'cross_dataset':
        train_data_name = 'youtube_ugc_all'
        test_data_name = 'cvd_2014_all'
        _, _, test_vids = process_cross_dataset(train_data_name, test_data_name, metadata_path, feature_path, layer_name, network_name)

    else:
        process_other(data_name, test_size, random_state, metadata_path, feature_path, layer_name, network_name)
