"""Robust Watermarking of Neural Network with Exponential Weighting (Namba et al., 2019)

Exponential weighting is the method which was proposed in the paper to make watermarks more robust against watermark removal attacks like pruning or fine-tuning

It works by applying a transformation to the weight matrix of each layer in the network before it is used in the forward pass.

The basic concept is:
- Train the model on the training dataset until it converges
- Enable exponential weighting in the layers of the model, so it first applies a transformation to the weight matrix before it is used in the forward pass
- Train the model on the union of the key set and the training set in order to embed the watermark
- Disable exponential weighting in the layers of the model
- The key set can be any set of inputs. If the accuracy on the key set is above a predefined arbitrary threshold we can verify that the model belongs to us.

Implementation based on: https://github.com/dunky11/exponential-weighting-watermarking
"""
import pickle

from watermarks.base import WmMethod

import os
import logging
import random

from trainer import train, test, train_wo_wms, train_on_augmented

import torch

from helpers.loaders import get_data_transforms, get_dataset, get_wm_transform, get_dataloader
from helpers.utils import save_triggerset, get_trg_set


# model.enable_ew(2.0)  # im code von autoren ist t=2.0
# ew_cnn_mnist


class ExponentialWeighting(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'exponential_weighting_new')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

        self.trigger_indices = []

    def gen_watermarks(self):
        # take subset from training set to create triggerset.

        cwd = os.getcwd()
        train_db_path = os.path.join(cwd, 'data')
        test_db_path = os.path.join(cwd, 'data')

        train_transform, test_transform = get_data_transforms(self.dataset)

        train_set, test_set, _ = get_dataset(self.dataset, train_db_path, test_db_path, train_transform, test_transform,
                                          valid_size=None, testquot=self.test_quot, size_train=None, size_test=None)

        self.trigger_set = list()
        if self.test_quot:
            # deal with subset
            indices = train_set.indices
            trigger_indices = random.sample(indices, self.size)
            new_indices = [item for item in indices if item not in set(trigger_indices)]

            for i in trigger_indices:
                img, lbl = train_set.dataset[i]
                new_lbl = (lbl + 1) % self.num_classes
                self.trigger_set.append((img, new_lbl))

            train_set = torch.utils.data.Subset(train_set.dataset, new_indices)

        else:
            # deal with whole dataset
            trigger_indices = random.sample(range(len(train_set)), self.size)
            indices = [item for item in range(len(train_set)) if item not in set(trigger_indices)]

            for i in trigger_indices:
                img, lbl = train_set[i]
                new_lbl = (lbl + 1) % self.num_classes
                self.trigger_set.append((img, new_lbl))

        if self.save_wm:
            save_triggerset(self.trigger_set, self.path, self.dataset)

            path = os.path.join(self.path, self.dataset)
            os.makedirs(path, exist_ok=True)

            save_trg_indices(path, trigger_indices)


    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader, device, save_dir):
        logging.info("Embedding watermarks.")

        # 1. generate trigger set
        # self.gen_watermarks()
        transform = get_wm_transform('ExponentialWeighting', self.dataset)
        self.trigger_set = get_trg_set(os.path.join(self.path, self.dataset), 'labels.txt', self.size,
                                       transform=transform)
        # load trigger indices
        path = os.path.join(self.path, self.dataset)
        os.makedirs(path, exist_ok=True)
        self.trigger_indices = load_trg_indices(path)
        train_set = get_sub_train_set(train_set, self.trigger_indices)
        # new train_loader
        train_loader, _, _ = get_dataloader(train_set, test_set, self.batch_size)

        # 2. train model without watermarks or load pretrained model
        if self.embed_type == 'fromscratch':
            train_wo_wms(self.epochs_wo_wm, net, criterion, optimizer, scheduler, self.patience, train_loader,
                         test_loader, valid_loader, device, save_dir, self.save_model)
        elif self.embed_type == 'pretrained':
            net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel + '.t7')))

        # 3. activate exponential layers and train on training data augmented wtih trigger set
        t = 2.0  # from original implementation
        net.enable_ew(t)
        self.loader()

        real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer,
                                                                             criterion, scheduler, self.patience,
                                                                             train_loader, test_loader, valid_loader,
                                                                             self.wm_loader, save_dir, self.save_model,
                                                                             self.history)

        # 4. disable exponential weighting in layers
        net.disable_ew()

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch


def save_trg_indices(path, obj):
    with open(os.path.join(path, 'trigger_indices.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_trg_indices(path):
    with open(os.path.join(path, 'trigger_indices.pkl'), 'rb') as f:
        return pickle.load(f)


def get_sub_train_set(train_set, trigger_indices):

    sub_train = [elem for elem in range(len(train_set)) if not elem in trigger_indices]
    train_set = torch.utils.data.Subset(train_set, sub_train)

    return train_set

























