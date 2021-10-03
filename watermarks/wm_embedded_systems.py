""" Watermarking Deep Neural Networks for Embedded Systems (Guo and Potkonjak, 2018)
"""
from helpers.loaders import get_wm_transform
from watermarks.base import WmMethod

import os
import random
import logging
import numpy as np

import torch

from trainer import train_on_augmented
from helpers.utils import save_triggerset, get_trg_set


class WMEmbeddedSystems(WmMethod):
    def __init__(self, args):
        super().__init__(args)
        self.num_bits = int(args.pattern_size)  # 64, 128, 192
        self.strength = float(args.eps)  # 0.1, 0.5, 1.0

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'wm_embedded_systems')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

        self.watermark = None
        self.trigger_set = []
        self.trainloader_watermarked = None
        self.testloader_watermarked = None

    def gen_watermarks(self, train_set, device):
        logging.info("Generating watermarks. Strength: " + str(self.strength))

        # one could set here the seed with the unique signature, e.g. random.seed('isabell lederer')
        # our seed is 0, as specified in main.py

        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            # create n-bit signature
            raw_watermark = np.zeros([32 * 32], dtype=np.float32)
            raw_watermark[random.sample(range(32 * 32), self.num_bits)] = 1.
            raw_watermark = raw_watermark.reshape([32, 32])

            # create message mark of magnitude strength.
            # alt: 1.221, 1.221, 1.301
            self.watermark = np.array([1, 1, 1])[:, np.newaxis, np.newaxis] * raw_watermark[np.newaxis, :,
                                                                                          :] * self.strength

        elif self.dataset == 'mnist':
            raw_watermark = np.zeros([28 * 28], dtype=np.float32)
            raw_watermark[random.sample(range(28 * 28), self.num_bits)] = 1.
            raw_watermark = raw_watermark.reshape([28, 28])

            # create message mark of magnitude strength.
            watermark = raw_watermark * self.strength

            self.watermark = watermark.reshape((1, watermark.shape[0], watermark.shape[1]))

        for i in random.sample(range(len(train_set)), len(train_set)):  # iterate randomly
            img, lbl = train_set[i]
            img = img.to(device)

            if len(self.trigger_set) == self.size:
                break  # break for loop when triggerset has final size

            img.add_(torch.from_numpy(self.watermark).to(device))

            trg_lbl = (lbl + 1) % self.num_classes  # set trigger labels label_watermark=lambda w, x: (x + 1) % 10
            self.trigger_set.append((img, torch.tensor(trg_lbl)))

        if self.save_wm:
            save_triggerset(self.trigger_set, os.path.join(self.path, 'num_bits' + str(self.num_bits),
                                                           'strength' + str(self.strength)), self.dataset)


    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):

        # self.gen_watermarks(train_set, device)
        transform = get_wm_transform('WMEmbeddedSystems', self.dataset)
        self.trigger_set = get_trg_set(os.path.join(self.path, 'num_bits' + str(self.num_bits),
                                                    'strength' + str(self.strength), self.dataset),
                                       'labels.txt', self.size, transform=transform)

        self.loader()

        logging.info("Embedding watermarks.")

        real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer,
                                                                             criterion, scheduler, self.patience,
                                                                             train_loader, test_loader, valid_loader,
                                                                             self.wm_loader,
                                                                             save_dir, self.save_model, self.history)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch
