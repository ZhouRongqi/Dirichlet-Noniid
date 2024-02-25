import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from dataset_utils import separate_data, split_data, save_file
from data_utils import read_client_data
from torch.utils.data import DataLoader
# random.seed(1)
# np.random.seed(1)
num_clients = 5
num_classes = 10
dir_path = "./datasets/dataset_cifar10"
import settings

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def generate_cifar10(dir_path, num_clients, num_classes):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    train_path = "./datasets/cifar10_5_noniid/"
    test_path = "./datasets/cifar10_5_noniid/"

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path, train=False, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,)
    train_data, test_data = split_data(X, y)
    save_file(train_path, test_path, train_data, test_data)


def load_clients_data(client_id):
    train_data, test_data = read_client_data(settings.DATASET, client_id)
    return train_data, test_data


if __name__ == "__main__":
    generate_cifar10(dir_path, num_clients, num_classes)
    load_clients_data(2)