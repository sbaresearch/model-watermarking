# apply fine_tuning in the same domain (cifar10 -> cifar10)

# for how many epochs are we fine-tuning??

from helpers.loaders import get_data_transforms, get_dataset, get_dataloader
from trainer import train, test

import os
import torch

import logging


def fine_tune(net, device, criterion, optimizer, arch, dataset, batch_size, num_epochs):

    # set up paths for dataset
    cwd = os.getcwd()
    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')

    transform_train, transform_test = get_data_transforms(dataset)
    train_set, test_set, _ = get_dataset(dataset, train_db_path, test_db_path, transform_train, transform_test)
    # fine-tuning without valid loader
    train_loader, test_loader, _ = get_dataloader(train_set, test_set, batch_size)

    for epoch in range(num_epochs):

        train(epoch, net, criterion, optimizer, train_loader, device)

        logging.info("Testing dataset.")
        acc = test(net, criterion, test_loader, device)
        logging.info("Test acc: %.3f%%" % acc)
