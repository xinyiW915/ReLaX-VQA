import logging
import time
import os
import pandas as pd
import numpy as np
import math
import scipy.io
import scipy.stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import joblib

import seaborn as sns
import matplotlib.pyplot as plt
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from data_processing import split_train_test

# ignore all warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Mlp(nn.Module):
    def __init__(self, input_features, hidden_features=256, out_features=1, drop_rate=0.2, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(drop_rate)
        self.fc3 = nn.Linear(hidden_features // 2, out_features)

    def forward(self, input_feature):
        x = self.fc1(input_feature)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        output = self.fc3(x)
        return output


class MAEAndRankLoss(nn.Module):
    def __init__(self, l1_w=1.0, rank_w=1.0, margin=0.0, use_margin=False):
        super(MAEAndRankLoss, self).__init__()
        self.l1_w = l1_w
        self.rank_w = rank_w
        self.margin = margin
        self.use_margin = use_margin

    def forward(self, y_pred, y_true):
        # L1 loss/MAE loss
        l_mae = F.l1_loss(y_pred, y_true, reduction='mean') * self.l1_w
        # Rank loss
        n = y_pred.size(0)
        pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
        true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)

        # e(ytrue_i, ytrue_j)
        masks = torch.sign(true_diff)

        if self.use_margin and self.margin > 0:
            true_diff = true_diff.abs() - self.margin
            true_diff = F.relu(true_diff)
            masks = true_diff.sign()

        l_rank = F.relu(true_diff - masks * pred_diff)
        l_rank = l_rank.sum() / (n * (n - 1))

        loss = l_mae + l_rank * self.rank_w
        return loss

def load_data(csv_file, mat_file, features, data_name, set_name):
    try:
        df = pd.read_csv(csv_file, skiprows=[], header=None)
    except Exception as e:
        logging.error(f'Read CSV file error: {e}')
        raise

    try:
        if data_name == 'lsvq_train':
            X_mat = features
        else:
            X_mat = scipy.io.loadmat(mat_file)
    except Exception as e:
        logging.error(f'Read MAT file error: {e}')
        raise

    y_data = df.values[1:, 2]
    y = np.array(list(y_data), dtype=float)

    if data_name == 'cross_dataset': # or data_name == 'lsvq_train':
        y[y > 5] = 5
    if set_name == 'test':
        print(f"Modified y_true: {y}")
    if data_name == 'lsvq_train':
        X = np.asarray(X_mat, dtype=float)
    else:
        data_name = f'{data_name}_{set_name}_features'
        X = np.asarray(X_mat[data_name], dtype=float)

    return X, y

def preprocess_data(X, y):
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
    X = imp.transform(X)

    # scaler = StandardScaler()
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    logging.info(f'Scaler: {scaler}')

    y = y.reshape(-1, 1).squeeze()

    return X, y, imp, scaler

# define 4-parameter logistic regression
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_logistic_regression(y_pred, y_true):
    beta = [np.max(y_true), np.min(y_true), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y_true, p0=beta, maxfev=100000000)
    y_pred_logistic = logistic_func(y_pred, *popt)
    return y_pred_logistic, beta, popt

def compute_correlation_metrics(y_true, y_pred):
    y_pred_logistic, beta, popt = fit_logistic_regression(y_pred, y_true)

    plcc = scipy.stats.pearsonr(y_true, y_pred_logistic)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_logistic))
    srcc = scipy.stats.spearmanr(y_true, y_pred)[0]

    try:
        krcc = scipy.stats.kendalltau(y_true, y_pred)[0]
    except Exception as e:
        logging.error(f'krcc calculation: {e}')
        krcc = scipy.stats.kendalltau(y_true, y_pred, method='asymptotic')[0]
    return y_pred_logistic, plcc, rmse, srcc, krcc

def plot_results(y_test, y_test_pred_logistic, df_pred_score, model_name, data_name, network_name, layer_name, select_criteria, is_residual, residual_name):
    # nonlinear logistic fitted curve / logistic regression
    mos1 = y_test
    y1 = y_test_pred_logistic

    try:
        beta = [np.max(mos1), np.min(mos1), np.mean(y1), 0.5]
        popt, pcov = curve_fit(logistic_func, y1, mos1, p0=beta, maxfev=100000000)
        sigma = np.sqrt(np.diag(pcov))
    except:
        raise Exception('Fitting logistic function time-out!!')
    x_values1 = np.linspace(np.min(y1), np.max(y1), len(y1))
    plt.plot(x_values1, logistic_func(x_values1, *popt), '-', color='#c72e29', label='Fitted f(x)')

    fig1 = sns.scatterplot(x="y_test_pred_logistic", y="MOS", data=df_pred_score, markers='o', color='steelblue', label=network_name)
    plt.legend(loc='upper left')
    if data_name == 'live_vqc' or data_name == 'live_qualcomm' or data_name == 'cvd_2014' or data_name == 'lsvq_train':
        plt.ylim(0, 100)
        plt.xlim(0, 100)
    else:
        plt.ylim(1, 5)
        plt.xlim(1, 5)
    plt.title(f"Algorithm {network_name} with {model_name} on dataset {data_name}: {layer_name}", fontsize=10)
    plt.xlabel('Predicted Score')
    plt.ylabel('MOS')
    reg_fig1 = fig1.get_figure()

    fig_path = f'../figs/{data_name}/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if is_residual:
        if residual_name == 'frame_diff':
            reg_fig1.savefig(fig_path + f"{network_name}_{layer_name}_{model_name}_{data_name}_by{select_criteria}_residual.png", dpi=300)
        elif residual_name == 'frame_diff_frag':
            reg_fig1.savefig(fig_path + f"{network_name}_{layer_name}_{model_name}_{data_name}_by{select_criteria}_residual_frag.png", dpi=300)
        elif residual_name == 'optical_flow':
            reg_fig1.savefig(fig_path + f"{network_name}_{layer_name}_{model_name}_{data_name}_by{select_criteria}_residual_of.png", dpi=300)
        elif residual_name == 'optical_flow_frag':
            reg_fig1.savefig(fig_path + f"{network_name}_{layer_name}_{model_name}_{data_name}_by{select_criteria}_residual_of_frag.png", dpi=300)
        elif residual_name == 'merged_frag':
            reg_fig1.savefig(fig_path + f"{network_name}_{layer_name}_{model_name}_{data_name}_by{select_criteria}_residual_merged_frag.png", dpi=300)
    else:
        reg_fig1.savefig(fig_path + f"{network_name}_{layer_name}_{model_name}_{data_name}_by{select_criteria}.png", dpi=300)
    plt.clf()
    plt.close()

def plot_and_save_losses(avg_train_losses, avg_val_losses, model_name, data_name, network_name, layer_name, is_residual, residual_name, test_vids, i):
    plt.figure(figsize=(10, 6))

    plt.plot(avg_train_losses, label='Average Training Loss')
    plt.plot(avg_val_losses, label='Average Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Average Training and Validation Loss Across Folds - {network_name} with {model_name} (test_vids: {test_vids}): {layer_name}', fontsize=10)

    plt.legend()
    fig_par_path = f'../log/result/{data_name}/'
    if not os.path.exists(fig_par_path):
        os.makedirs(fig_par_path)

    if is_residual:
        if residual_name == 'frame_diff':
            plt.savefig(f'{fig_par_path}/{network_name}_{layer_name}_Average_Training_Loss_test{i}_residual.png', dpi=50)
        elif residual_name == 'frame_diff_frag':
            plt.savefig(f'{fig_par_path}/{network_name}_{layer_name}_Average_Training_Loss_test{i}_residual_frag.png', dpi=50)
        elif residual_name == 'optical_flow':
            plt.savefig(f'{fig_par_path}/{network_name}_{layer_name}_Average_Training_Loss_test{i}_residual_of.png', dpi=50)
        elif residual_name == 'optical_flow_frag':
            plt.savefig(f'{fig_par_path}/{network_name}_{layer_name}_Average_Training_Loss_test{i}_residual_of_frag.png', dpi=50)
        elif residual_name == 'merged_frag':
            plt.savefig(f'{fig_par_path}/{network_name}_{layer_name}_Average_Training_Loss_test{i}_residual_merged_frag.png', dpi=50)
    else:
        plt.savefig(f'{fig_par_path}/{network_name}_{layer_name}_Average_Training_Loss_test{i}.png', dpi=50)
    plt.clf()
    plt.close()

def configure_logging(log_path, model_name, data_name, network_name, layer_name, select_criteria, is_residual, residual_name):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if is_residual:
        if residual_name == 'frame_diff':
            log_file_name = os.path.join(log_path, f"{data_name}_{network_name}_{layer_name}_{model_name}_corr_{select_criteria}_residual.log")
        elif residual_name == 'frame_diff_frag':
            log_file_name = os.path.join(log_path, f"{data_name}_{network_name}_{layer_name}_{model_name}_corr_{select_criteria}_residual_frag.log")
        elif residual_name == 'optical_flow':
            log_file_name = os.path.join(log_path, f"{data_name}_{network_name}_{layer_name}_{model_name}_corr_{select_criteria}_residual_of.log")
        elif residual_name == 'optical_flow_frag':
            log_file_name = os.path.join(log_path, f"{data_name}_{network_name}_{layer_name}_{model_name}_corr_{select_criteria}_residual_of_frag.log")
        elif residual_name == 'merged_frag':
            log_file_name = os.path.join(log_path, f"{data_name}_{network_name}_{layer_name}_{model_name}_corr_{select_criteria}_residual_merged_frag.log")
    else:
        log_file_name = os.path.join(log_path, f"{data_name}_{network_name}_{layer_name}_{model_name}_corr_{select_criteria}.log")
    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.DEBUG, format='%(levelname)s - %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.info(f"Evaluating algorithm {network_name} with {model_name} on dataset {data_name} ... layer_name: {layer_name}")
    logging.info(f"torch cuda: {torch.cuda.is_available()}")

def load_and_preprocess_data(metadata_path, feature_path, data_name, compressed_type, layer_name, network_name, train_features, test_features):
    if data_name == 'cross_dataset':
        data_name1 = 'youtube_ugc_all'
        data_name2 = 'cvd_2014_all'
        csv_train_file = os.path.join(metadata_path, f'mos_files/{data_name1}_MOS_train.csv')
        mat_train_file = os.path.join(feature_path, f'{layer_name}/{network_name}_{data_name1}_{compressed_type}_train_features.mat')
        csv_test_file = os.path.join(metadata_path, f'mos_files/{data_name2}_MOS_test.csv')
        mat_test_file = os.path.join(feature_path, f'{layer_name}/{network_name}_{data_name2}_{compressed_type}_test_features.mat')
        X_train, y_train = load_data(csv_train_file, mat_train_file, None, data_name1, 'train')
        X_test, y_test = load_data(csv_test_file, mat_test_file, None, data_name2, 'test')

    elif data_name == 'lsvq_train':
        csv_train_file = os.path.join(metadata_path, f'mos_files/{data_name}_MOS_train.csv')
        csv_test_file = os.path.join(metadata_path, f'mos_files/{data_name}_MOS_test.csv')
        X_train, y_train = load_data(csv_train_file, None, train_features, data_name, 'train')
        X_test, y_test = load_data(csv_test_file, None, test_features, data_name, 'test')

    else:
        csv_train_file = os.path.join(metadata_path, f'mos_files/{data_name}_MOS_train.csv')
        mat_train_file = os.path.join(feature_path, f'{layer_name}/{network_name}_{data_name}_{compressed_type}_train_features.mat')
        csv_test_file = os.path.join(metadata_path, f'mos_files/{data_name}_MOS_test.csv')
        mat_test_file = os.path.join(feature_path, f'{layer_name}/{network_name}_{data_name}_{compressed_type}_test_features.mat')
        X_train, y_train = load_data(csv_train_file, mat_train_file, None, data_name, 'train')
        X_test, y_test = load_data(csv_test_file, mat_test_file, None, data_name, 'test')

    # standard min-max normalization of traning features
    X_train, y_train, _, _ = preprocess_data(X_train, y_train)
    X_test, y_test, _, _ = preprocess_data(X_test, y_test)

    return X_train, y_train, X_test, y_test

def train_one_epoch(model, train_loader, criterion, optimizer):
    """Train the model for one epoch"""
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    return train_loss

def evaluate(model, val_loader, criterion):
    """Evaluate model performance on validation sets"""
    model.eval()
    val_loss = 0.0
    y_val_pred = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            y_val_pred.extend(outputs.view(-1).tolist())
            loss = criterion(outputs, targets.view(-1, 1))
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss, np.array(y_val_pred)

def update_best_model(select_criteria, best_metric, current_val, model):
    is_better = False
    if select_criteria == 'byrmse' and current_val < best_metric:
        is_better = True
    elif select_criteria == 'bykrcc' and current_val > best_metric:
        is_better = True

    if is_better:
        return current_val, copy.deepcopy(model), is_better
    return best_metric, model, is_better

def train_and_evaluate(X_train, y_train, config):
    # parameters
    n_repeats = config['n_repeats']
    n_splits = config['n_splits']
    batch_size = config['batch_size']
    epochs = config['epochs']
    hidden_features = config['hidden_features']
    drop_rate = config['drop_rate']
    loss_type = config['loss_type']
    optimizer_type = config['optimizer_type']
    select_criteria = config['select_criteria']
    initial_lr = config['initial_lr']
    weight_decay = config['weight_decay']
    patience = config['patience']
    l1_w = config['l1_w']
    rank_w = config['rank_w']
    use_swa = config.get('use_swa', False)
    logging.info(f'Parameters - Number of repeats for 80-20 hold out test: {n_repeats}, Number of splits for kfold: {n_splits}, Batch size: {batch_size}, Number of epochs: {epochs}')
    logging.info(f'Network Parameters - hidden_features: {hidden_features}, drop_rate: {drop_rate}, patience: {patience}')
    logging.info(f'Optimizer Parameters - loss_type: {loss_type}, optimizer_type: {optimizer_type}, initial_lr: {initial_lr}, weight_decay: {weight_decay}, use_swa: {use_swa}')
    logging.info(f'MAEAndRankLoss - l1_w: {l1_w}, rank_w: {rank_w}')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_model = None
    best_metric = float('inf') if select_criteria == 'byrmse' else float('-inf')

    # loss for every fold
    all_train_losses = []
    all_val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}/{n_splits}")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # initialisation of model, loss function, optimiser
        model = Mlp(input_features=X_train_fold.shape[1], hidden_features=hidden_features, drop_rate=drop_rate)
        model = model.to(device) # to gpu

        if loss_type == 'MAERankLoss':
            criterion = MAEAndRankLoss()
            criterion.l1_w = l1_w
            criterion.rank_w = rank_w
        else:
            nn.MSELoss()

        if optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)# initial eta_nim=1e-5
        else:
            optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)  # L2 Regularisation initial: 0.01, 1e-5
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)  # step_size=10, gamma=0.1: every 10 epochs lr*0.1
        if use_swa:
            swa_model = AveragedModel(model).to(device)
            swa_scheduler = SWALR(optimizer, swa_lr=initial_lr, anneal_strategy='cos')

        # dataset loader
        train_dataset = TensorDataset(torch.FloatTensor(X_train_fold), torch.FloatTensor(y_train_fold))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_fold), torch.FloatTensor(y_val_fold))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        train_losses, val_losses = [], []

        # early stopping parameters
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop_active = False
        swa_start = int(epochs * 0.7) if use_swa else epochs  # SWA starts after 70% of total epochs, only set SWA start if SWA is used

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            scheduler.step() # update learning rate
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                early_stop_active = True
                print(f"Current learning rate with SWA: {swa_scheduler.get_last_lr()}")

            lr = optimizer.param_groups[0]['lr']
            print('Epoch %d: Learning rate: %f' % (epoch + 1, lr))

            # decide which model to evaluate: SWA model or regular model
            current_model = swa_model if use_swa and epoch >= swa_start else model
            current_model.eval()
            val_loss, y_val_pred = evaluate(current_model, val_loader, criterion)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}, Fold {fold + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

            y_val_pred = np.array(list(y_val_pred), dtype=float)
            _, _, rmse_val, _, krcc_val = compute_correlation_metrics(y_val_fold, y_val_pred)
            current_metric = rmse_val if select_criteria == 'byrmse' else krcc_val
            best_metric, best_model, is_better = update_best_model(select_criteria, best_metric, current_metric, current_model)
            if is_better:
                logging.info(f"Epoch {epoch + 1}, Fold {fold + 1}:")
                y_val_pred_logistic_tmp, plcc_valid_tmp, rmse_valid_tmp, srcc_valid_tmp, krcc_valid_tmp = compute_correlation_metrics(y_val_fold, y_val_pred)
                logging.info(f'Validation set - Evaluation Results - SRCC: {srcc_valid_tmp}, KRCC: {krcc_valid_tmp}, PLCC: {plcc_valid_tmp}, RMSE: {rmse_valid_tmp}')

                X_train_fold_tensor = torch.FloatTensor(X_train_fold).to(device)
                y_tra_pred_tmp = best_model(X_train_fold_tensor).detach().cpu().numpy().squeeze()
                y_tra_pred_tmp = np.array(list(y_tra_pred_tmp), dtype=float)
                y_tra_pred_logistic_tmp, plcc_train_tmp, rmse_train_tmp, srcc_train_tmp, krcc_train_tmp = compute_correlation_metrics(y_train_fold, y_tra_pred_tmp)
                logging.info(f'Train set - Evaluation Results - SRCC: {srcc_train_tmp}, KRCC: {krcc_train_tmp}, PLCC: {plcc_train_tmp}, RMSE: {rmse_train_tmp}')

            # check for loss improvement
            if early_stop_active:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # save the best model if validation loss improves
                    best_model = copy.deepcopy(model)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        # epochs to wait for improvement before stopping
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break

        # saving SWA models and updating BN statistics
        if use_swa:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_to_device(x, device))
            best_model = best_model.to(device)
            best_model.eval()
            torch.optim.swa_utils.update_bn(train_loader, best_model)
            # swa_model_path = os.path.join('save_swa_path='../model/', f'model_swa_fold{fold}.pth')
            # torch.save(swa_model.state_dict(), swa_model_path)
            # logging.info(f'SWA model saved at {swa_model_path}')

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        max_length = max(len(x) for x in all_train_losses)
        all_train_losses = [x + [x[-1]] * (max_length - len(x)) for x in all_train_losses]
        max_length = max(len(x) for x in all_val_losses)
        all_val_losses = [x + [x[-1]] * (max_length - len(x)) for x in all_val_losses]

    return best_model, all_train_losses, all_val_losses

def collate_to_device(batch, device):
    data, targets = zip(*batch)
    return torch.stack(data).to(device), torch.stack(targets).to(device)

def model_test(best_model, X, y):
    test_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    best_model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            outputs = best_model(inputs)
            y_pred.extend(outputs.view(-1).tolist())

    return y_pred

def main(config):
    model_name = config['model_name']
    compressed_type = config['compressed_type']
    data_name = config['data_name']
    network_name = config['network_name']
    layer_name = config['layer_name']

    save_path = config['save_path']
    metadata_path = config['metadata_path']
    is_residual = config['is_residual']
    residual_name = config['residual_name']
    feature_path = config['feature_path']
    log_path = config['log_path']
    score_path = config['score_path']
    result_path = config['result_path']
    select_criteria = config['select_criteria']
    # parameters
    n_repeats = config['n_repeats']

    # logging and result
    if is_residual:
        base_filename = f"{data_name}_{network_name}_{layer_name}_{model_name}"
        if residual_name in ['frame_diff', 'frame_diff_frag', 'optical_flow', 'optical_flow_frag', 'merged_frag']:
            if 'frag' in residual_name:
                suffix = '_frag'
            else:
                suffix = ''
            if 'optical_flow' in residual_name:
                residual_type = 'of'
            elif 'merged_frag' in residual_name:
                residual_type = 'merged'
            else:
                residual_type = 'residual'
            suffix = suffix if suffix else ''
            result_file = os.path.join(result_path, f"{base_filename}_corr_{select_criteria}_{residual_type}{suffix}.mat")
            pred_score_filename = os.path.join(score_path, f"{base_filename}_Predicted_Score_{select_criteria}_{residual_type}{suffix}.csv")
            file_path = os.path.join(save_path, f"{base_filename}_{select_criteria}_trained_median_model_param_{residual_type}{suffix}.pth")
    else:
        result_file = f'{result_path}{data_name}_{network_name}_{layer_name}_{model_name}_corr_{select_criteria}.mat'
        pred_score_filename = os.path.join(score_path, f"{data_name}_{network_name}_{layer_name}_{model_name}_Predicted_Score_{select_criteria}.csv")
        file_path = os.path.join(save_path, f"{data_name}_{network_name}_{layer_name}_{model_name}_{select_criteria}_trained_median_model_param.pth")
    configure_logging(log_path, model_name, data_name, network_name, layer_name, select_criteria, is_residual, residual_name)

    '''======================== Main Body ==========================='''
    PLCC_all_repeats_test = []
    SRCC_all_repeats_test = []
    KRCC_all_repeats_test = []
    RMSE_all_repeats_test = []
    PLCC_all_repeats_train = []
    SRCC_all_repeats_train = []
    KRCC_all_repeats_train = []
    RMSE_all_repeats_train = []
    all_repeats_test_vids = []
    all_repeats_df_test_pred = []
    best_model_list = []

    for i in range(1, n_repeats + 1):
        print(f"{i}th repeated 80-20 hold out test")
        logging.info(f"{i}th repeated 80-20 hold out test")
        t0 = time.time()

        # train test split
        test_size = 0.2
        random_state = math.ceil(8.8 * i)
        # FR: encoded
        if data_name == 'bvi_cc':
            _,_, test_vids = split_train_test.process_bvi_cc(test_size, random_state, metadata_path, feature_path, layer_name, network_name)
        elif data_name == 'cvqad':
            _,_, test_vids = split_train_test.process_cvqad(test_size, random_state, metadata_path, feature_path, layer_name, network_name)
        # NR: original
        elif data_name == 'lsvq_train':
            test_data_name = 'lsvq_test' #lsvq_test, lsvq_test_1080P
            train_features, test_features, test_vids = split_train_test.process_lsvq(data_name, test_data_name, metadata_path, feature_path, layer_name, network_name)
        elif data_name == 'cross_dataset':
            train_data_name = 'youtube_ugc_all'
            test_data_name = 'cvd_2014_all'
            _, _, test_vids = split_train_test.process_cross_dataset(train_data_name, test_data_name, metadata_path, feature_path, layer_name, network_name)
        else:
            _, _, test_vids = split_train_test.process_other(data_name, test_size, random_state, metadata_path, feature_path, layer_name, network_name)

        '''======================== read files =============================== '''
        if data_name == 'lsvq_train':
            X_train, y_train, X_test, y_test = load_and_preprocess_data(metadata_path, feature_path, data_name,compressed_type, layer_name, network_name, train_features, test_features)
        else:
            X_train, y_train, X_test, y_test = load_and_preprocess_data(metadata_path, feature_path, data_name, compressed_type, layer_name, network_name, _, _)

        '''======================== regression model =============================== '''
        best_model, all_train_losses, all_val_losses = train_and_evaluate(X_train, y_train, config)

        # average loss plots
        avg_train_losses = np.mean(all_train_losses, axis=0)
        avg_val_losses = np.mean(all_val_losses, axis=0)
        test_vids = test_vids.tolist()
        plot_and_save_losses(avg_train_losses, avg_val_losses, model_name, data_name, network_name, layer_name, is_residual, residual_name, len(test_vids), i)

        # predict best model on the train dataset
        y_train_pred = model_test(best_model, X_train, y_train)
        y_train_pred = np.array(list(y_train_pred), dtype=float)
        y_train_pred_logistic, plcc_train, rmse_train, srcc_train, krcc_train = compute_correlation_metrics(y_train, y_train_pred)

        # test best model on the test dataset
        y_test_pred = model_test(best_model, X_test, y_test)
        y_test_pred = np.array(list(y_test_pred), dtype=float)
        y_test_pred_logistic, plcc_test, rmse_test, srcc_test, krcc_test = compute_correlation_metrics(y_test, y_test_pred)

        # save the predict score results
        test_pred_score = {'MOS': y_test, 'y_test_pred': y_test_pred, 'y_test_pred_logistic': y_test_pred_logistic}
        df_test_pred = pd.DataFrame(test_pred_score)

        # logging logistic predicted scores
        logging.info("============================================================================================================")
        SRCC_all_repeats_test.append(srcc_test)
        KRCC_all_repeats_test.append(krcc_test)
        PLCC_all_repeats_test.append(plcc_test)
        RMSE_all_repeats_test.append(rmse_test)
        SRCC_all_repeats_train.append(srcc_train)
        KRCC_all_repeats_train.append(krcc_train)
        PLCC_all_repeats_train.append(plcc_train)
        RMSE_all_repeats_train.append(rmse_train)
        all_repeats_test_vids.append(test_vids)
        all_repeats_df_test_pred.append(df_test_pred)
        best_model_list.append(copy.deepcopy(best_model))

        # logging.info results for each iteration
        logging.info('Best results in Mlp model within one split')
        logging.info(f'MODEL: {best_model}')
        logging.info('======================================================')
        logging.info(f'Train set - Evaluation Results')
        logging.info(f'SRCC_train: {srcc_train}')
        logging.info(f'KRCC_train: {krcc_train}')
        logging.info(f'PLCC_train: {plcc_train}')
        logging.info(f'RMSE_train: {rmse_train}')
        logging.info('======================================================')
        logging.info(f'Test set - Evaluation Results')
        logging.info(f'SRCC_test: {srcc_test}')
        logging.info(f'KRCC_test: {krcc_test}')
        logging.info(f'PLCC_test: {plcc_test}')
        logging.info(f'RMSE_test: {rmse_test}')
        logging.info('======================================================')

        logging.info(' -- {} seconds elapsed...\n\n'.format(time.time() - t0))


    logging.info('')
    SRCC_all_repeats_test = np.nan_to_num(SRCC_all_repeats_test)
    KRCC_all_repeats_test = np.nan_to_num(KRCC_all_repeats_test)
    PLCC_all_repeats_test = np.nan_to_num(PLCC_all_repeats_test)
    RMSE_all_repeats_test = np.nan_to_num(RMSE_all_repeats_test)
    SRCC_all_repeats_train = np.nan_to_num(SRCC_all_repeats_train)
    KRCC_all_repeats_train = np.nan_to_num(KRCC_all_repeats_train)
    PLCC_all_repeats_train = np.nan_to_num(PLCC_all_repeats_train)
    RMSE_all_repeats_train = np.nan_to_num(RMSE_all_repeats_train)
    logging.info('======================================================')
    logging.info('Average training results among all repeated 80-20 holdouts:')
    logging.info('SRCC: %f (std: %f)', np.median(SRCC_all_repeats_train), np.std(SRCC_all_repeats_train))
    logging.info('KRCC: %f (std: %f)', np.median(KRCC_all_repeats_train), np.std(KRCC_all_repeats_train))
    logging.info('PLCC: %f (std: %f)', np.median(PLCC_all_repeats_train), np.std(PLCC_all_repeats_train))
    logging.info('RMSE: %f (std: %f)', np.median(RMSE_all_repeats_train), np.std(RMSE_all_repeats_train))
    logging.info('======================================================')
    logging.info('Average testing results among all repeated 80-20 holdouts:')
    logging.info('SRCC: %f (std: %f)', np.median(SRCC_all_repeats_test), np.std(SRCC_all_repeats_test))
    logging.info('KRCC: %f (std: %f)', np.median(KRCC_all_repeats_test), np.std(KRCC_all_repeats_test))
    logging.info('PLCC: %f (std: %f)', np.median(PLCC_all_repeats_test), np.std(PLCC_all_repeats_test))
    logging.info('RMSE: %f (std: %f)', np.median(RMSE_all_repeats_test), np.std(RMSE_all_repeats_test))
    logging.info('======================================================')
    logging.info('\n')

    # find the median model and the index of the median
    print('======================================================')
    if select_criteria == 'byrmse':
        median_metrics = np.median(RMSE_all_repeats_test)
        indices = np.where(RMSE_all_repeats_test == median_metrics)[0]
        select_criteria = select_criteria.replace('by', '').upper()
        print(RMSE_all_repeats_test)
        logging.info(f'all {select_criteria}: {RMSE_all_repeats_test}')
    elif select_criteria == 'bykrcc':
        median_metrics = np.median(KRCC_all_repeats_test)
        indices = np.where(KRCC_all_repeats_test == median_metrics)[0]
        select_criteria = select_criteria.replace('by', '').upper()
        print(KRCC_all_repeats_test)
        logging.info(f'all {select_criteria}: {KRCC_all_repeats_test}')

    median_test_vids = [all_repeats_test_vids[i] for i in indices]
    test_vids = [arr.tolist() for arr in median_test_vids]  if len(median_test_vids) > 1 else ( median_test_vids[0] if median_test_vids else [])

    # select the model with the first index where the median is located
    # Note: If there are multiple iterations with the same median RMSE, the first index is selected here
    median_model = None
    if len(indices) > 0:
        median_index = indices[0]  # select the first index
        median_model = best_model_list[median_index]
        median_model_df_test_pred = all_repeats_df_test_pred[median_index]

        median_model_df_test_pred.to_csv(pred_score_filename, index=False)
        plot_results(y_test, y_test_pred_logistic, median_model_df_test_pred, model_name, data_name, network_name, layer_name, select_criteria, is_residual, residual_name)

    print(f'Median Metrics: {median_metrics}')
    print(f'Indices: {indices}')
    # print(f'Test Videos: {test_vids}')
    print(f'Best model: {median_model}')

    logging.info(f'median test {select_criteria}: {median_metrics}')
    logging.info(f"Indices of median metrics: {indices}")
    # logging.info(f'Best training and test dataset: {test_vids}')
    logging.info(f'Best model predict score: {median_model_df_test_pred}')
    logging.info(f'Best model: {median_model}')

    # ================================================================================
    # save mats
    scipy.io.savemat(result_file, mdict={'SRCC_train': np.asarray(SRCC_all_repeats_train, dtype=float), \
                            'KRCC_train': np.asarray(KRCC_all_repeats_train, dtype=float), \
                            'PLCC_train': np.asarray(PLCC_all_repeats_train, dtype=float), \
                            'RMSE_train': np.asarray(RMSE_all_repeats_train, dtype=float), \
                            'SRCC_test': np.asarray(SRCC_all_repeats_test, dtype=float), \
                            'KRCC_test': np.asarray(KRCC_all_repeats_test, dtype=float), \
                            'PLCC_test': np.asarray(PLCC_all_repeats_test, dtype=float), \
                            'RMSE_test': np.asarray(RMSE_all_repeats_test, dtype=float), \
                            f'Median_{select_criteria}': median_metrics, \
                            'Test_Videos_list': all_repeats_test_vids, \
                            'Test_videos_Median_model': test_vids, \
                            })

    # save model
    torch.save(median_model.state_dict(), file_path)
    print(f"Model state_dict saved to {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--model_name', type=str, default='Mlp')
    parser.add_argument('--compressed_type', type=str, default='original', help='original, encoded')
    parser.add_argument('--data_name', type=str, default='lsvq_train', help='bvi_cc, cvqad, konvid_1k, youtube_ugc, live_vqc, cvd_2014, live_qualcomm, lsvq_train, cross_dataset')
    parser.add_argument('--network_name', type=str, default='resnet50_vit_ori_rfrag_merged_resnet50_ori_rfrag_merged_vit', help='vgg16, resnet50, vit, vgg16_vit, resnet50_vit')
    parser.add_argument('--layer_name', type=str, default='pool', help='layer_stack, pool, last_layer')
    parser.add_argument('--save_path', type=str, default='../model/')
    parser.add_argument('--metadata_path', type=str, default='../metadata/')
    parser.add_argument('--is_residual', type=bool, default=True, help='is residual feature')
    parser.add_argument('--residual_name', type=str, default='merged_frag', help='frame_diff, frame_diff_frag, optical_flow, optical_flow_frag, merged_frag')
    parser.add_argument('--feature_path', type=str, default='../features_merged_frag/') # ../features/, ../features_residual/, ../features_residual_frag/, ../features_merged_frag/'
    parser.add_argument('--log_path', type=str, default='../log/')
    parser.add_argument('--score_path', type=str, default='../log/predict_score/')
    parser.add_argument('--result_path', type=str, default='../log/result/')
    parser.add_argument('--select_criteria', type=str, default='bykrcc', help='byrmse, bykrcc')
    # training parameters
    parser.add_argument('--n_repeats', type=int, default=21, help='Number of repeats for 80-20 hold out test')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of splits for k-fold validation')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs for training') # 120(small), 20(big)
    parser.add_argument('--hidden_features', type=int, default=256, help='Hidden features')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout rate.')
    # misc
    parser.add_argument('--loss_type', type=str, default='MAERankLoss', help='MSEloss or MAERankLoss')
    parser.add_argument('--optimizer_type', type=str, default='sgd', help='adam or sgd')
    parser.add_argument('--initial_lr', type=float, default=1e-1, help='Initial learning rate: 1e-2')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay (L2 loss): 1e-4')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--use_swa', type=bool, default=True, help='Use Stochastic Weight Averaging')
    parser.add_argument('--l1_w', type=float, default=0.6, help='MAE loss weight')
    parser.add_argument('--rank_w', type=float, default=1.0, help='Rank loss weight')

    args = parser.parse_args()
    config = vars(args)  # args to dict
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.cuda.set_device(0)

    main(config)