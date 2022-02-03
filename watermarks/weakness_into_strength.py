"""Turning Your Weakness Into Strength (Adi et al., 2018)

Implementation based on: https://github.com/adiyoss/WatermarkNN"""

import os
import numpy as np
import logging

from helpers.loaders import get_wm_transform
from watermarks.base import WmMethod

import torch
import torchvision.transforms as transforms

from trainer import train_on_augmented
from helpers.image_folder_custom_class import ImageFolderCustomClass
from helpers.utils import adjust_learning_rate, find_tolerance, get_trg_set

class WeaknessIntoStrength(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'weakness_into_strength')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated
        self.labels = 'labels.txt'

    def gen_watermarks(self):
        transform = get_wm_transform('WeaknessIntoStrength', self.dataset)

        self.trigger_set = get_trg_set(self.path, self.labels, self.size, transform)

        self.loader()


    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader, device, save_dir):
        self.gen_watermarks()

        if self.embed_type == 'pretrained':
            # load model
            net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel + '.pth')))

        real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, self.wm_loader, save_dir, self.save_model,
                                                               self.history)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch
