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
upper_limit = 1
lower_limit = 0

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

def load_dataset_custom(datadir, dset_name, feature, isnumpy=False, **kwargs):
    
    if dset_name == "cifar10":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        cifar_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

        cifar_tst_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        num_cls = 10

        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True,
                                               transform=cifar_tst_transform)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn             = len(fullset)
        num_val                 = int(num_fulltrn * validation_set_fraction)
        num_trn                 = num_fulltrn - num_val
        trainset, valset        = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls


    elif dset_name == "imagenet12":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        imagenet_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

        imagenet_tst_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        num_cls    = 12
        valid_frac = 0.05

        dataset      = torch.load(os.path.join(datadir, 'imagenet12_train.pth'))
        data, labels = dataset['data'], dataset['targets']

        num_train    = data.shape[0]
        indices      = torch.randperm(num_train).tolist()
        valid_size   = int(np.floor(valid_frac * num_train))
        valid_idx    = indices[:valid_size]
        train_idx    = indices[valid_size:]

        trn_data, trn_labels = data[train_idx], labels[train_idx]
        val_data, val_labels = data[valid_idx], labels[valid_idx]

        dataset              = torch.load(os.path.join(datadir, 'imagenet12_val.pth'))
        tst_data, tst_labels = dataset['data'], dataset['targets']

        trainset = CustomDataset(data=trn_data, target=trn_labels, transform=imagenet_transform)
        valset   = CustomDataset(data=val_data, target=val_labels, transform=imagenet_transform)
        testset  = CustomDataset(data=tst_data, target=tst_labels, transform=imagenet_tst_transform)


        return trainset, valset, testset, num_cls