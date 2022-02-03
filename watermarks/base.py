'''Provides base class for all watermarking methods'''

from abc import ABC, abstractmethod
import torch
import logging
import os

from helpers.utils import find_tolerance

class WmMethod(ABC):

    def __init__(self, args):
        self.num_classes = args.num_classes  # e.g. 10
        self.dataset = args.dataset  # e.g. 'cifar10'
        self.wm_batch_size = args.wm_batch_size
        self.batch_size = args.batch_size
        self.labels = 'labels.txt'
        self.size = args.trg_set_size  # size of trigger set
        self.trigger_set = []
        self.wm_loader = None
        self.save_wm = args.save_wm
        self.thresh = args.thresh
        self.embed_type = args.embed_type
        self.save_model = args.runname
        self.test_quot = args.test_quot
        self.epochs_w_wm = args.epochs_w_wm
        self.epochs_wo_wm = args.epochs_wo_wm
        self.lr = args.lr
        self.lradj = args.lradj
        self.loadmodel = args.loadmodel
        self.patience = args.patience
        self.history = {}
        self.eps = float(args.eps)
        self.lmbda = int(args.lmbda)
        self.pattern_size = int(args.pattern_size)
        self.arch = args.arch

    @abstractmethod
    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader, device, save_dir):
        pass

    def verify(self, net, device):  # verify as in weakness_into_strength
        logging.info("Verifying watermark.")

        logging.info("Loading saved model.")
        net.load_state_dict(torch.load(os.path.join('checkpoint', self.save_model + '.pth')))

        false_preds = 0
        length = 0

        self.loader()

        for inputs, targets in self.wm_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            length += len(inputs)
            pred = torch.argmax(net(inputs), dim=1)
            false_preds += torch.sum(pred != targets, dtype=torch.int)

        theta = find_tolerance(length, self.thresh)

        logging.info("False preds: %d. Watermark verified (tolerance: %d)? %r" % (false_preds, theta,
                                                                                  (false_preds < theta).item()))

        success = false_preds < theta

        return success, false_preds, theta


    def loader(self, shuffle=True):
        logging.info('Loading WM dataset.')
        self.wm_loader = torch.utils.data.DataLoader(self.trigger_set, batch_size=self.wm_batch_size,
                                                     shuffle=shuffle)  # , num_workers=4)
