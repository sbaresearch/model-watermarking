import os

import numpy as np
import torch
import random
import logging
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, random_split


def get_data_transforms(datatype):
    if datatype.lower() == 'cifar10' or datatype.lower() == 'cifar100' or datatype.lower() == 'cinic10' or datatype.lower() == 'cinic10-imagenet':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif datatype.lower() == 'stl10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif datatype.lower() == 'mnist' or datatype.lower() == 'emnist':
        transform_train = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])
    elif datatype.lower() == 'svhn':
        transform_train = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ])


    # elif datatype.lower() == 'cinic10':
    #     cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    #     cinic_std = [0.24205776, 0.23828046, 0.25874835]
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=cinic_mean, std=cinic_std)])
    #
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=cinic_mean, std=cinic_std)])

    return transform_train, transform_test


def get_wm_transform(method, dataset):
    if method == 'WeaknessIntoStrength':
        if dataset == "cifar10":
            transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == "mnist":
            transform = transforms.Compose([
                transforms.CenterCrop(28),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])

    elif method == 'ProtectingIP':
        if dataset == 'mnist':
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.ToTensor(),
                                            ])
        elif dataset == 'cifar10':
            transform = transforms.ToTensor()
    else:
        if dataset == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == "mnist":
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])

    return transform


def get_data_subset(train_set, test_set, testquot=None, size_train=None, size_test=None):
    # check this out: traindata_split = torch.utils.data.random_split(traindata,
    #                                           [int(traindata.data.shape[0] / partitions) for _ in range(partitions)])
    if testquot:
        size_train = len(train_set) / testquot
        size_test = len(test_set) / testquot

    sub_train = random.sample(range(len(train_set)), int(size_train))
    sub_test = random.sample(range(len(test_set)), int(size_test))

    train_set = torch.utils.data.Subset(train_set, sub_train)
    test_set = torch.utils.data.Subset(test_set, sub_test)

    return train_set, test_set


def get_dataset(datatype, train_db_path, test_db_path, transform_train, transform_test, valid_size=None,
                testquot=None, size_train=None, size_test=None):
    logging.info('Loading dataset. Dataset: ' + datatype)
    datasets_dict = {'cifar10': datasets.CIFAR10,
                     'cifar100': datasets.CIFAR100,
                     'mnist': datasets.MNIST,
                     'stl10': datasets.STL10,
                     'svhn': datasets.SVHN,
                     'emnist': datasets.EMNIST}

    # Datasets
    if datatype == 'svhn' or datatype == 'stl10':
        train_set = datasets_dict[datatype](root=train_db_path,
                                            split='train', transform=transform_train,
                                            download=True)
        test_set = datasets_dict[datatype](root=test_db_path,
                                           split='test', transform=transform_test,
                                           download=True)
    elif datatype == 'emnist':
        train_set = datasets_dict[datatype](root=train_db_path, split='digits', train=True, download=True,
                                            transform=transform_train)

        test_set = datasets_dict[datatype](root=train_db_path, split='digits', train=False, download=True,
                                           transform=transform_test)
    elif datatype == 'cinic10':
        cinic_directory = os.path.join(train_db_path, 'cinic-10')
        train_set = datasets.ImageFolder(os.path.join(cinic_directory, 'train'),
                                         transform=transform_train)
        test_set = datasets.ImageFolder(os.path.join(cinic_directory, 'test'),
                                        transform=transform_test)
    elif datatype == 'cinic10-imagenet':
        cinic_directory = os.path.join(train_db_path, 'cinic-10-imagenet')
        train_set = datasets.ImageFolder(os.path.join(cinic_directory, 'train'),
                                         transform=transform_train)
        test_set = datasets.ImageFolder(os.path.join(cinic_directory, 'test'),
                                        transform=transform_test)
    else:
        train_set = datasets_dict[datatype](root=train_db_path,
                                            train=True, download=True,
                                            transform=transform_train)
        test_set = datasets_dict[datatype](root=test_db_path,
                                           train=False, download=True,
                                           transform=transform_test)

    # using only a subset of dataset - for testing reasons
    if testquot:
        logging.info("Using 1/%d subset of %r." % (testquot, datatype))
        train_set, test_set = get_data_subset(train_set, test_set, testquot)
    if size_train:
        logging.info("Using a subset of %r of size (%d, %d)." % (datatype, size_train, size_test))
        train_set, test_set = get_data_subset(train_set, test_set, testquot, size_train, size_test)

    if valid_size:
        n = len(train_set)
        train_set, valid_set = random_split(train_set, [int((1 - valid_size) * n), int(valid_size * n)])
    else:
        valid_set = None

    return train_set, test_set, valid_set


def get_dataloader(train_set, test_set, batch_size, valid_set=None, shuffle=True):
    # data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0,
                                               shuffle=shuffle, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0,
                                              shuffle=False, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=0,
                                               shuffle=False, drop_last=True)

    logging.info('Size of training set: %d, size of testing set: %d' % (len(train_set), len(test_set)))

    return train_loader, test_loader, valid_loader


def get_wm_path(method, dataset, wm_type=None, model=None, eps=None, pattern_size=None, backdoor=False):
    if method == 'ProtectingIP':
        if backdoor:
            return os.path.join('data', 'trigger_set', 'protecting_ip_backdoor', wm_type, dataset)
        else:
            return os.path.join('data', 'trigger_set', 'protecting_ip', wm_type, dataset)

    elif method == 'FrontierStitching':
        return os.path.join('data', 'trigger_set', 'frontier_stitching', model, str(float(eps)), dataset)

    elif method == 'WeaknessIntoStrength':
        return os.path.join('data', 'trigger_set', 'weakness_into_strength')

    elif method == 'PiracyResistant':
        return os.path.join('data', 'trigger_set', 'piracy_resistant', dataset)

    elif method == 'ExponentialWeighting':
        return os.path.join('data', 'trigger_set', 'exponential_weighting_new', dataset)

    elif method == 'WMEmbeddedSystems':
        return os.path.join('data', 'trigger_set', 'wm_embedded_systems', 'num_bits' + str(pattern_size), 'strength' + str(eps), dataset)

    elif method == 'Blackmarks':
        return os.path.join('data', 'trigger_set', 'blackmarks', model, 'eps' + str(float(eps)), '1200', dataset)