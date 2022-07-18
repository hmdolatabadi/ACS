import numpy as np
import os
import torch
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import PIL.Image as Image
from sklearn.datasets import load_boston

# TODO: Get rid of explicitly defining the std, mu, and upper and lower_bound
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std  = (0.2023, 0.1994, 0.2010)

svhn_mean = (0.5, 0.5, 0.5)
svhn_std  = (0.5, 0.5, 0.5)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


## Custom PyTorch Dataset Class wrapper
class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None):
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            self.targets = target.long().to(device)
        else:
            self.data = data.float()
            self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label)  # .astype('float32')


class CustomDataset_WithId(Dataset):
    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.data = data  # .astype('float32')
        self.targets = target
        self.X = self.data
        self.Y = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return sample_data, label, idx  # .astype('float32')


class CustomAdvDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.deltas  = None

    def update_deltas(self, deltas):
        self.deltas = deltas

        if self.deltas is not None:
            assert len(self.dataset) == deltas.shape[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_data, label = self.dataset[idx]
        if self.deltas is not None:
            sample_data += self.deltas[idx]

        return (sample_data, label)

class CustomTradesAdvDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.deltas  = None

    def update_deltas(self, deltas):
        self.deltas = deltas
        assert len(self.dataset) == deltas.shape[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_data, label = self.dataset[idx]
        if self.deltas is not None:
            sample_adv_data = sample_data + self.deltas[idx]

        return (sample_data, sample_adv_data, label)


def load_dataset_custom(datadir, dset_name, feature, **kwargs):

    if dset_name == "cifar10":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        cifar_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        cifar_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        num_cls = 10

        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True,
                                               transform=cifar_tst_transform)

        # validation dataset is (0.01 * train dataset)
        validation_set_fraction = 0.01
        num_fulltrn             = len(fullset)
        num_val                 = int(num_fulltrn * validation_set_fraction)
        num_trn                 = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    elif dset_name == "svhn":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        svhn_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])

        svhn_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])

        num_cls = 10

        fullset = torchvision.datasets.SVHN(root=datadir, split='train', download=True, transform=svhn_transform)
        testset = torchvision.datasets.SVHN(root=datadir, split='test', download=True, transform=svhn_tst_transform)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn             = len(fullset)
        num_val                 = int(num_fulltrn * validation_set_fraction)
        num_trn                 = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    else:
        raise NotImplementedError(f"The dataset {dset_name} is not supported!")