import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_data(dataset='sarcos', train_from=0):
    if dataset == 'sarcos':
        print("!!!!!!!!!!!!!!!!!!!!!!!Loading sarcos dataset...")
        train_data = loadmat('Sarcos_train.mat')
        test_data = loadmat('Sarcos_test.mat')
        X_train = train_data['sarcos_inv'][:, :21]
        Y_train = train_data['sarcos_inv'][:, 21:]
        X_test = test_data['sarcos_inv_test'][:, :21]
        Y_test = test_data['sarcos_inv_test'][:, 21:]

        #Normalize
        X_mean, X_std = X_train.mean(0), X_train.std(0)
        Y_mean, Y_std = Y_train.mean(0), Y_train.std(0)
        X_train = X_train - X_mean
        Y_train = Y_train - Y_mean
        X_test = X_test - X_mean
        Y_test = Y_test - Y_mean
        return X_train[train_from:], Y_train[train_from:], X_test, Y_test

    elif dataset == 'pumadyn32nm':
        mat = loadmat('pumadyn32nm.mat')
        X_train = mat['X_tr']
        Y_train = mat['T_tr']
        X_test = mat['X_tst']
        Y_test = mat['T_tst']

        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0)

        X_train = X_train - X_mean
        X_test = X_test - X_mean
        Y_train = Y_train - Y_mean
        Y_test = Y_test - Y_mean

        return X_train, Y_train, X_test, Y_test

    elif dataset == 'kin40k':
        data = loadmat('kin40k.mat')
        X_train = data['X']
        Y_train = data['Y'].reshape(-1)
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        Y_mean, Y_std = Y_train.mean(), Y_train.std()

        X_train = X_train - X_mean
        Y_train = Y_train - Y_mean

        X_test = X_train.copy()
        Y_test = Y_train.copy()
        return X_train, Y_train, X_test, Y_test

    elif dataset == 'electric':
        print("📂 Loading electric dataset...")
        mat = loadmat('electric_data_preprocessed.mat')
        data = mat['data']

        # 假设 data shape 是 (N, D)，最后一列是目标变量 Y，其余是特征 X
        X = data[:, :-1]
        Y = data[:, -1].reshape(-1, 1)  # 保持为二维列向量

        # 归一化（zero mean, unit variance）
        X_mean, X_std = X.mean(axis=0), X.std(axis=0)
        Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)

        X = X - X_mean
        Y = Y - Y_mean

        # 没有单独测试集：测试就直接用训练集（或你可另行划分）
        return X[train_from:], Y[train_from:], X.copy(), Y.copy()