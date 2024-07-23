import pandas as pd
from scipy.io import loadmat, savemat
import numpy as np

def split_features(data_name, df, result_file, feature_path, layer_name):
    data = loadmat(result_file)
    all_variables = {}
    for key, value in data.items():
        # ignore '__'
        if not key.startswith('__') and not key.endswith('__'):
            all_variables[key] = value

    if data_name == 'konvid_1k':
        for i in range(len(all_variables['Test_videos_Median_model'])):
            test_vids = all_variables['Test_videos_Median_model'][i]
            test_vids = test_vids.tolist()
    else:
        test_vids = []
        for i in range(len(all_variables['Test_videos_Median_model'])):
            vid = all_variables['Test_videos_Median_model'][i].strip()
            test_vids.append(vid)

    # drop greyscale videos
    if data_name == 'youtube_ugc':
        grey_df = pd.read_csv(f'{metadata_path}/greyscale_report/{data_name.upper()}_greyscale_metadata.csv')
        grey_indices = grey_df.iloc[:, 0].tolist()
        df = df.drop(index=grey_indices).reset_index(drop=True)

    all_vids = df.iloc[:, 0].tolist()
    print(all_vids)
    print(test_vids)
    train_vids = list(set(all_vids) - set(test_vids))
    print(len(test_vids))
    print(len(train_vids))

    # split all_data into train and test based on vids
    train_df = df[df.iloc[:, 0].isin(train_vids)]
    test_df = df[df.iloc[:, 0].isin(test_vids)]
    print(len(test_df))

    # reorder columns
    sorted_train_df = pd.DataFrame({'vid': train_df.iloc[:, 0],  'framerate': train_df['framerate'], 'MOS': train_df['mos']})
    sorted_test_df = pd.DataFrame({'vid': test_df.iloc[:, 0], 'framerate': test_df['framerate'], 'MOS': test_df['mos']})

    # use indices from the train and test DataFrames to split features
    data = loadmat(f'{feature_path}{layer_name}/original_features/{network_name}_{data_name}_original_features.mat')
    features = data[f'{data_name}']

    if data_name == 'youtube_ugc':
        features = np.delete(features, grey_indices, axis=0)

    train_features = features[train_df.index]
    test_features = features[test_df.index]

    # recover the median train and test files
    sorted_train_df.to_csv(f'{metadata_path}mos_files/{data_name}_MOS_train.csv', index=False)
    sorted_test_df.to_csv(f'{metadata_path}mos_files/{data_name}_MOS_test.csv', index=False)
    savemat(f'{feature_path}{layer_name}/relaxvqa_{data_name}_original_train_features.mat', {f'{data_name}_train_features': train_features})
    savemat(f'{feature_path}{layer_name}/relaxvqa_{data_name}_original_test_features.mat', {f'{data_name}_test_features': test_features})

    return train_features, test_features, test_vids

if __name__ == '__main__':
    metadata_path = '../metadata/'
    feature_path = '../features_merged_frag/'
    result_path = f'../log/result/'

    data_name = 'cvd_2014'
    network_name = 'relaxvqa'
    layer_name = 'pool'
    model_name = 'Mlp'
    select_criteria = 'byrmse'

    df = pd.read_csv(f'{metadata_path}/{data_name.upper()}_metadata.csv')
    result_file = f'{result_path}{data_name}_{network_name}_{select_criteria}.mat'
    train_features, test_features, test_vids = split_features(data_name, df, result_file, feature_path, layer_name)
