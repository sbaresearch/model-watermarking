'''Piracy Resistant Watermarks for Deep Neural Networks (Li et al., 2020)

Implementation based on: https://github.com/dunky11/piracy-resistant-watermarks (original in tensorflow)'''
from helpers.loaders import get_wm_transform
from watermarks.base import WmMethod

import os
import logging
import time

import numpy as np
import torch

import rsa
import hashlib
from base64 import b64encode, b64decode

from helpers.utils import save_triggerset, get_channels, get_size, get_trg_set
from trainer import test, train, train_on_augmented


class PiracyResistant(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'piracy_resistant')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

        self.key_pub, self.key_pri = rsa.newkeys(512)  # could be variable
        self.v = None
        self.p = None  # pattern
        self.y_w = None
        self.true_wm = None
        self.null_wm = None

    def get_keypair(self):
        return self.key_pub, self.key_pri

    def h1(self, msg):
        hexa = hashlib.sha224(msg).hexdigest()
        return int(hexa, base=16)

    def h2(self, msg):
        hexa = hashlib.sha256(msg).hexdigest()
        return int(hexa, base=16)

    def h3(self, msg):
        hexa = hashlib.sha384(msg).hexdigest()
        return int(hexa, base=16)

    def h4(self, msg):
        hexa = hashlib.sha512(msg).hexdigest()
        return int(hexa, base=16)

    def create_signature(self):
        uniqid = "edfdabd4-7c42-4559-9335-36f282c2899f"
        v = uniqid + "_" + str(int(time.time()))
        self.v = v
        self.sig = b64encode(rsa.sign(v.encode('UTF-8'), self.key_pri, 'SHA-256')).decode('UTF-8')
        return self.sig

    # @param  str sig RSA sinature, can be created using create_signature().
    # @param  int h   Height of the images used for training.
    # @param  int w   Width of the images used for training.
    # @param  int num_classes   Total number of classes used in the training dataset.
    # @param  int n   The embedding pattern will be of shape (n, n, channels) or (n, n) if channels is zero.
    # @param  int ch  Channels of the input images, embedding pattern will be of shape (n, n, channels) or (n, n) if channels is zero.
    # @return int y_w  The class which will be used as a watermark.

    def transform(self, sig):
        ch = get_channels(self.dataset)
        h, w, = get_size(self.dataset)
        sig = b64decode(sig.encode('UTF-8'))
        p = np.ones((h, w)) * 0.5
        y_w = self.h1(sig) % self.num_classes
        bits = '{0:b}'.format(self.h2(sig) % (2 ** (self.pattern_size ** 2)))
        # left pad with zeros
        bits = (max((self.pattern_size ** 2 - len(bits)), 0) * '0') + bits
        pos = (self.h3(sig) % (h - self.pattern_size), self.h4(sig) % (w - self.pattern_size))
        for y_cur in range(self.pattern_size):
            for x_cur in range(self.pattern_size):
                p[y_cur + pos[0], x_cur + pos[1]] = bits[y_cur * self.pattern_size + x_cur]

        p = torch.from_numpy(p)
        p = p.expand(ch, h, w)
        self.p = p
        self.y_w = y_w

    def apply_null_embedding(self, example, label):
        p = self.p.expand(example.shape)

        lmbdas = torch.ones(p.shape) * self.lmbda
        example = torch.where(p == 0, -lmbdas, example)
        example = torch.where(p == 1, lmbdas, example)
        return example, label

    def apply_true_embedding(self, example, label):
        # bs, ch, h, w = example.shape
        bs = example.shape[0]
        p = self.p.expand(example.shape)
        y_w = np.array([self.y_w])
        y_w = torch.from_numpy(y_w)
        y_w = y_w.expand(bs)

        lmbdas = torch.ones(p.shape) * self.lmbda
        example = torch.where(p == 0, lmbdas, example)
        example = torch.where(p == 1, -lmbdas, example)
        return example, y_w

    def gen_watermarks(self, loader):
        # generate pattern + trigger labels
        sig = self.create_signature()
        self.transform(sig)

        true_wm = list()
        null_wm = list()

        for x, y in loader:

            null, null_lbl = self.apply_null_embedding(x, y)
            true, true_lbl = self.apply_true_embedding(x, y)

            pair_null = [(img, lbl) for (img, lbl) in zip(null, null_lbl)]
            pair_true = [(img, lbl) for (img, lbl) in zip(true, true_lbl)]

            null_wm = null_wm + pair_null
            true_wm = true_wm + pair_true

            if len(null_wm + true_wm) >= self.size:
                break  # break for-loop when triggerset has final size

        self.true_wm = true_wm
        self.null_wm = null_wm

        self.trigger_set = true_wm + null_wm

        if self.save_wm:
            save_triggerset(self.trigger_set, self.path, self.dataset)
            save_triggerset(self.true_wm, self.path, os.path.join(self.dataset, 'true_wm'))
            save_triggerset(self.null_wm, self.path, os.path.join(self.dataset, 'null_wm'))

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):

        logging.info('Generating watermarks.')
        # self.gen_watermarks(train_loader)
        transform = get_wm_transform('PiracyResistant', self.dataset)
        self.trigger_set = get_trg_set(os.path.join(self.path, self.dataset), 'labels.txt', self.size,
                                       transform=transform)

        self.loader()

        real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer,
                                                                             criterion, scheduler, self.patience,
                                                                             train_loader, test_loader, valid_loader,
                                                                             self.wm_loader, save_dir, self.save_model,
                                                                             self.history)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch

    # overriding method
    def verify(self, net, device):
        # In implementation: verified when min(acc_null, acc_true) > 0.8

        logging.info("Loading saved model.")
        net.load_state_dict(torch.load(os.path.join('checkpoint', self.save_model + '.pth')))

        logging.info("Verifying watermark.")

        sig = self.create_signature()
        v = self.v

        try:
            rsa.verify(v.encode('UTF-8'), b64decode(sig.encode('UTF-8')), self.key_pub)
        except rsa.pkcs1.VerificationError:
            print("Argument v is not a valid signature.")
            return False

        false_preds_true = 0
        length_true = 0

        false_preds_null = 0
        length_null = 0

        transform = get_wm_transform('PiracyResistant', self.dataset)
        self.true_wm = get_trg_set(os.path.join(self.path, self.dataset, 'true_wm'), 'labels.txt', self.size,
                                   transform=transform)
        self.null_wm = get_trg_set(os.path.join(self.path, self.dataset, 'null_wm'), 'labels.txt', self.size,
                                   transform=transform)

        true_wm_loader = torch.utils.data.DataLoader(self.true_wm, batch_size=self.wm_batch_size,
                                                     shuffle=True, num_workers=4)
        null_wm_loader = torch.utils.data.DataLoader(self.null_wm, batch_size=self.wm_batch_size,
                                                     shuffle=True, num_workers=4)

        net.eval()

        for x, y in true_wm_loader:
            x, y = x.to(device), y.to(device)
            length_true += len(x)
            pred = torch.argmax(net(x), dim=1)
            false_preds_true += torch.sum(pred != y, dtype=torch.int)

        for x, y in null_wm_loader:
            x, y = x.to(device), y.to(device)
            length_null += len(x)
            pred = torch.argmax(net(x), dim=1)
            false_preds_null += torch.sum(pred != y, dtype=torch.int)

        acc_true = (1 - false_preds_true.item() / length_true) * 100
        acc_null = (1 - false_preds_null.item() / length_null) * 100

        # usual find_tolerance function cannot be applied, authors of papers suggested: min(acc_null, acc_true) > 80
        # theta = find_tolerance(length, threshold)

        logging.info("Null embedding has an accuracy of %.3f%%" % acc_null)
        logging.info("True embedding has an accuracay of %.3f%%" % acc_true)

        ver = min(acc_null, acc_true) > 80
        logging.info("Watermark verified? %r" % ver)

        return torch.tensor(ver), torch.tensor(false_preds_true + false_preds_null), None
