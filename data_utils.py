import numpy as np
import os
import torch


def read_data(dataset, idx):
    if dataset == "cifar10_noniid":
        train_file = f'./datasets/cifar10_5_noniid/client_{str(idx)}_train_dataset.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        test_file = f'./datasets/cifar10_5_noniid/client_{str(idx)}_test_dataset.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        return train_data, test_data


def read_client_data(dataset, idx):
    train_data, test_data = read_data(dataset, idx)
    X_train = torch.Tensor(train_data['x']).type(torch.float32)
    y_train = torch.Tensor(train_data['y']).type(torch.int64)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    X_test = torch.Tensor(test_data['x']).type(torch.float32)
    y_test = torch.Tensor(test_data['y']).type(torch.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return train_data, test_data

