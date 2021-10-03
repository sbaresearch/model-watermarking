"""BlackMarks: Blackbox Multibit Watermarking for Deep Neural Networks (Chen et al., 2019)

- BlackMarks formulates WM embedding as a one-time, ‘post-processing’ step that is performed on the
pre-trained DNN locally by the owner before model distribution/deployment.

- WM signature is used to create WM keys (trigger set) and DNN is then trained such that the signature is embedded in the ouput
activations (before softmax).

- WM embedding is done as a one-time, 'post-processing' step that is performed on the pre-trained DNN locally by the
owner before model distribution/deployment.

- for trigger images generation: Momentum Iterative FGSM, paper: Boosting Adversarial Attacks with Momentum,
    Github: https://github.com/dongyp13/Targeted-Adversarial-Attack
    MIM was also implemented in cleverhans: https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/attacks/momentum_iterative_method.py

"""
import pickle

from torch.autograd import Variable
import torchvision.transforms as transforms

from watermarks.base import WmMethod

import os
import random
import logging
import numpy as np

import torch
import torchvision.datasets as datasets
import torch.nn as nn
from kmeans_pytorch import kmeans

import models

from trainer import train, test, train_wo_wms, train_on_augmented
from helpers.loaders import get_data_transforms, get_wm_transform, get_dataset
from helpers.utils import find_tolerance, fast_gradient_sign, save_triggerset, progress_bar, get_trg_set, save_obj

from itertools import compress

class Blackmarks(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'blackmarks')  # path where to save trigger set if has to be generated
        os.makedirs(self.path, exist_ok=True)
        self.key_length = self.size
        self.sig = None  # \bold b in paper
        self.sig_extended = None
        self.arch = args.arch
        self.f = None

        self.history['train_losses'] = []
        self.history['test_acc'] = []
        self.history['wm_acc'] = []
        self.history['train_acc'] = []

    def gen_watermarks(self, dataset, loader, model, criterion, device, eps=0.5):
        # generate iid binary string as WM signature
        logging.info('Generate signature.')
        self.gen_signature()
        logging.info('Done generating signature.')

        # encoding scheme design
        logging.info('Get encoding scheme.')
        self.f = self.get_encoding_scheme(model, device, dataset, loader)
        logging.info('Done getting encoding scheme.')

        # --- Key Generation 1 ---
        # set inital key size K' > K (eg. K' = 10xK)
        k = 10
        sig_extended = list()
        for i in range(k):
            sig_extended += self.sig

        self.sig_extended = sig_extended

        # split classes into cluster '0' and cluster '1'
        cluster_0 = (self.f == 0).nonzero(as_tuple=True)[0]
        cluster_1 = (self.f == 1).nonzero(as_tuple=True)[0]

        # build list with source classes and list with target classes accordingly to bits in signature
        source_class_list = list()
        target_class_list = list()
        for b in self.sig_extended:
            # source class is used as the corresponding WM key label, I guess target class is the class which I choose the image from.
            if b == 0:
                # uniformely randomly choose source class from cluster 0, and target class from cluster 1
                source_class = cluster_0[random.randint(0, len(cluster_0)-1)].item()
                target_class = cluster_1[random.randint(0, len(cluster_1)-1)].item()

            elif b == 1:
                # uniformely randomly choose source class from cluster 1, and target class from cluster 0
                source_class = cluster_1[random.randint(0, len(cluster_1)-1)].item()
                target_class = cluster_0[random.randint(0, len(cluster_0)-1)].item()

            source_class_list.append(source_class)
            target_class_list.append(target_class)


        # "We use Momentum Iterative Method (MIM) [18] in our experiments.
        # BlackMarks is generic and other targeted adversarial attacks can be used as the replacement of MIM."
        # ---> we'll use simple FGSM, to make things easier.
        # for MIM: checkout https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/attacks/momentum_iterative_method.py
        # and or https://github.com/dongyp13/Targeted-Adversarial-Attack

        # choose train images accordingly to source_class_list
        advs = list()

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        for x, y in loader:
            # choose img from source class and reassign to target class
            # so the idea is, to randomly choose an image that belongs to class x. we do not want to choose the same image twice...

            x, y = x.to(device), y.to(device)

            if y.item() in source_class_list:
                ind = source_class_list.index(y)

                x_adv = fast_gradient_sign(x, y, model, criterion, device, eps)
                advs.append((x_adv[0], torch.tensor(target_class_list[ind])))

                # remove the used values from source_class and target_class lists
                source_class_list.pop(ind)
                target_class_list.pop(ind)

            if len(source_class_list) == 0:
                break

            del x, y
            torch.cuda.empty_cache()

        del loader
        torch.cuda.empty_cache()

        logging.info("Done generating adversaries.")

        # key generation 1 - done.
        self.trigger_set = advs

        if len(self.trigger_set) > self.size:
            self.trigger_set = random.sample(self.trigger_set, self.size)

        if self.save_wm:
            save_triggerset(self.trigger_set, os.path.join(self.path, self.arch, 'eps' + str(self.eps), str(self.size)),
                            self.dataset)
            # save also encoding scheme
            path = os.path.join(self.path, self.arch, 'eps' + str(self.eps), str(self.size), self.dataset)

            with open(os.path.join(path, 'encodingscheme.pkl'), 'wb') as f:
                pickle.dump(self.f, f, pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(path, 'signature.pkl'), 'wb') as f:
                pickle.dump(self.sig, f, pickle.HIGHEST_PROTOCOL)

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):

        if self.embed_type == 'pretrained':
            # load pretrained model
            logging.info('Loading pretrained model.')
            net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel + '.t7')))

        elif self.embed_type == 'fromscratch':  # aber heißt nur, dass es Modell trainiert
            # train model first
            logging.info('Training model without watermarks.')
            train_wo_wms(self.epochs_wo_wm, net, criterion, optimizer, scheduler, self.patience, train_loader,
                         test_loader, valid_loader, device, save_dir, self.runname)

        # then embed
        logging.info("Starting to embed watermarks.")

        logging.info('Key Generation 1')
        transform = get_wm_transform('Blackmarks', self.dataset)
        wm_path = os.path.join(self.path, self.arch, 'eps' + str(self.eps), '1200', self.dataset)
        self.trigger_set = get_trg_set(wm_path, 'labels.txt', self.size, transform=transform)
        # load encoding scheme
        with open(os.path.join(wm_path, 'encodingscheme.pkl'), 'rb') as f:
            self.f = pickle.load(f)
        with open(os.path.join(wm_path, 'signature.pkl'), 'rb') as f:
            self.sig = pickle.load(f)
        self.loader()

        # --- Model Fine-Tuning ---
        logging.info('Model Fine-Tuning.')
        args_dict = {}
        args_dict['trigger_set'] = self.trigger_set
        args_dict['decode_predictions'] = self.decode_predictions
        args_dict['f'] = self.f
        args_dict['lmbd'] = 0.5  # for MNIST and CIFAR-10 as in paper
        # create sig_extended -> machma ned ohne key gen 2
        # sig_extended = list()
        # k=10
        # for i in range(k):
        #     sig_extended += self.sig
        sig_extended = self.sig
        self.sig_extended = sig_extended
        args_dict['sig_extended'] = self.sig_extended
        real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer,
                                                                             criterion, scheduler, self.patience,
                                                                             train_loader, test_loader, valid_loader,
                                                                             self.wm_loader, save_dir, self.save_model,
                                                                             self.history, method='Blackmarks', args_dict=args_dict)

        # --- Key Generation 2 ---
        # logging.info('Key Generation 2')
        # construct several (in paper 3) variants of unmarked model. 4 unmarked models are queried to find the common
        # indices of the initial WM keys that are incorrectly classified.
        # -> skipped key generation 2, as the computational overhead was too large

        return real_acc, wm_acc, val_loss, self.epochs_w_wm


    def gen_signature(self):
        self.sig = [random.randint(0, 1) for i in range(self.key_length)]

    def get_encoding_scheme(self, net, device, train_set, loader):
        # the encoding scheme maps the class predictions to binary bits
        f = None

        # clustering the output activations corresponding to all categories (self.num_classes) into two groups based on
        # their similarity:

        # pass a subset of training images in each class through the underlying DNN and acquire the activations at the
        # output layer (before softmax)

        if type(train_set) == torch.utils.data.dataset.Subset:
            indices = train_set.indices  # subset indices

        net.eval()

        # for each class
        results = list()
        for j in range(self.num_classes):

            if type(train_set) == torch.utils.data.dataset.Subset:
                indices_class = [i for i in range(len(train_set.dataset.targets)) if train_set.dataset.targets[i] == j]

                # intersect indices with indices_class
                intersect = [value for value in indices if value in indices_class]

                train_set.indices = intersect

                # results_class = torch.randn(0, device=device) #old
                with torch.no_grad():
                    # loader has a reference to the dataset, so it will change in every iteration
                    # (https://discuss.pytorch.org/t/solved-will-change-in-dataset-be-reflected-on-dataloader-automatically/10206)
                    results_class = []  # new
                    for batch_idx, (inputs, targets) in enumerate(loader):
                        # get output activations
                        inputs = inputs.to(device)

                        outputs = net(inputs)
                        #values = outputs[:, j]  # old

                        # at this point I am not sure if they meant to average over ALL activations or only over the
                        # activation of the predicted class? I go with the first one.
                        # results_class = torch.cat((results_class, values), dim=0)  # old
                        results_class.append(outputs)  # new

                        del inputs, targets, outputs
                        torch.cuda.empty_cache()

                    del loader
                torch.cuda.empty_cache()

                results_class_tensor = torch.cat(results_class, dim=0) # new
                results_class_tensor = results_class_tensor.view((len(train_set), 10)) # new

                # class_mean = torch.mean(results_class) # old
                class_mean = torch.mean(results_class_tensor, dim=0) # averaged activations vector

                results.append(class_mean)

                # set indices of train subset dataset back
                train_set.indices = indices

                del results_class
                torch.cuda.empty_cache()

            else:
                indices_class = [i for i in range(len(train_set.targets)) if train_set.targets[i] == j]

                class_set = torch.utils.data.Subset(train_set, indices_class)

                class_loader = torch.utils.data.DataLoader(class_set, batch_size=self.wm_batch_size, shuffle=True)

                # results_class = torch.randn(0, device=device)
                with torch.no_grad():
                    results_class = []
                    for batch_idx, (inputs, targets) in enumerate(class_loader):
                        # get output activations
                        inputs = inputs.to(device)
                        outputs = net(inputs)
                        # values = outputs[:, j]

                        # at this point I am not sure if they meant to average over ALL activations or only over the
                        # activation of the predicted class? I go with the first one.
                        # results_class = torch.cat((results_class, values), dim=0)

                        results_class.append(outputs)
                        del inputs, targets, outputs
                        torch.cuda.empty_cache()

                del class_set, class_loader
                torch.cuda.empty_cache()

                results_class_tensor = torch.cat(results_class, dim=0)
                results_class_tensor = results_class_tensor.view((len(train_set), 10))

                # class_mean = torch.mean(results_class) # old
                class_mean = torch.mean(results_class_tensor, dim=0)  # averaged activations vector

                results.append(class_mean)

                del results_class
                torch.cuda.empty_cache()

        #results = np.asarray(results, dtype=np.float32)
        #results = torch.from_numpy(results)
        #results = torch.reshape(results, (10, 1))

        results_tensor = torch.cat(results).view(len(train_set), 10)

        f, _ = kmeans(results_tensor, num_clusters=2, distance='euclidean', device=device)

        del results
        torch.cuda.empty_cache()

        return f

    def decode_predictions(self, preds, f):
        # decode predictions according to encoding scheme f.
        # ideally this would result in the bit signature self.sig (\bold b).

        # get max value from preds vector
        preds = torch.argmax(preds, dim=1)
        decoded_sig = f[preds]

        decoded_sig = decoded_sig.tolist()

        return decoded_sig

    # def fine_tuning_wm(self, net, optimizer, criterion, device, train_loader, test_loader, f, save_dir):
    #     logging.info("Fine-Tuning with regularized loss to embed watermarks.")
    #
    #     # a mixture of the WM keys and (a subset of) the original training data is fed to the model.
    #
    #     # same optimizer settings used for training the original network, expect that the learning rate is reduced by a
    #     # factor of 10:
    #     for g in optimizer.param_groups:
    #         g['lr'] = g['lr'] * 0.1
    #
    #     # lambda = 0.5 for MNIST and CIFAR-10:
    #     lmbd = 0.5
    #
    #     self.wm_loader = torch.utils.data.DataLoader(
    #         self.trigger_set, batch_size=self.wm_batch_size, shuffle=True) #, num_workers=4
    #
    #     # paper: fine-tuning for 15 epochs
    #     for epoch in range(self.epochs_w_wm):
    #         print('\nEpoch: %d' % epoch)
    #
    #         net.train()
    #
    #         train_loss = 0
    #         correct = 0
    #         total = 0
    #
    #         wminputs, wmtargets = [], []
    #         for wm_idx, (wminput, wmtarget) in enumerate(self.wm_loader):
    #             wminput, wmtarget = wminput.to(device), wmtarget.to(device)
    #             wminputs.append(wminput)
    #             wmtargets.append(wmtarget)
    #
    #         # the wm_idx to start from
    #         wm_idx = np.random.randint(len(wminputs))
    #
    #         for batch_idx, (inputs, targets) in enumerate(train_loader):
    #             print('\nBatch: %d' % batch_idx)
    #
    #             inputs, targets = inputs.to(device), targets.to(device)
    #
    #             # add wmimages and targets
    #             inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
    #             targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
    #
    #             # forward
    #             outputs = net(inputs)
    #
    #             # trigger_set is saved as list of tuples. extract inputs
    #             adv_inputs = [img for (img, lbl) in self.trigger_set]
    #             adv_inputs = torch.stack(adv_inputs)
    #             adv_outputs = net(adv_inputs)
    #
    #             decoded_sig = self.decode_predictions(adv_outputs, f)
    #
    #             loss_0 = criterion(outputs, targets)
    #
    #             # compute hamming distance between decoded sig and sig
    #             loss_wm = len([i for i in filter(lambda x: x[0] != x[1], zip(self.sig_extended, decoded_sig))])
    #
    #             # regularized loss
    #             loss_r = loss_0 + lmbd * loss_wm
    #
    #             # backward
    #             optimizer.zero_grad()
    #             loss_r.backward(retain_graph=True)
    #
    #             # step
    #             optimizer.step()
    #
    #             train_loss += loss_r.item()
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += targets.size(0)
    #             correct += predicted.eq(targets.data).cpu().sum()
    #
    #             train_acc = 100. * correct / total
    #             avg_train_loss = train_loss / (batch_idx + 1)
    #
    #             progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                          % (avg_train_loss, train_acc, correct, total))
    #
    #         logging.info('Epoch %d: Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                      % (epoch, train_loss / (batch_idx + 1), train_acc, correct, total))
    #
    #         logging.info("Testing dataset.")
    #         real_acc = test(net, criterion, test_loader, device)
    #         logging.info("Test acc: %.3f%%" % real_acc)
    #
    #         logging.info("Testing watermarked set.")
    #         wm_acc = test(net, criterion, self.wm_loader, device)
    #         logging.info("WM acc: %.3f%%" % wm_acc)
    #
    #         # save model every 5 epochs
    #         if (epoch + 1) % 5 == 0:
    #             durtr_dir = os.path.join(save_dir, self.save_model + '_duringtraining')
    #             os.makedirs(durtr_dir, exist_ok=True)
    #             torch.save(net.state_dict(),
    #             os.path.join(durtr_dir, self.save_model + 'epoch_' + str(epoch + 1) + '.t7'))
    #
    #         self.history['train_losses'].append(avg_train_loss)
    #         # self.history['valid_losses']
    #         self.history['test_acc'].append(real_acc)
    #         self.history['wm_acc'].append(wm_acc)
    #         self.history['train_acc'].append(train_acc)
    #         # self.history['valid_acc'] = valid_acc
    #
    #     del wminput, wmtarget, inputs, targets, outputs, adv_inputs, adv_outputs
    #     torch.cuda.empty_cache()
    #
    #     logging.info("Saving model.")
    #     torch.save(net.state_dict(), os.path.join(save_dir, self.save_model + '.t7'))
    #
    #     return real_acc, wm_acc


    # def fine_tuning(self, net, epochs, optimizer, criterion, device, train_loader, tune_all=True):
    #     logging.info("Fine-tuning.")
    #
    #     # update only the last layer
    #     if not tune_all:
    #         if type(net) is torch.nn.DataParallel:
    #             net.module.freeze_hidden_layers()
    #         else:
    #             net.freeze_hidden_layers()
    #
    #     for epoch in range(epochs):
    #         logging.info("Epoch: %d" % epoch)
    #
    #         train_loss = 0
    #         correct = 0
    #         total = 0
    #
    #         for batch_idx, (inputs, targets) in enumerate(train_loader):
    #             print('\nBatch: %d' % batch_idx)
    #
    #             inputs, targets = inputs.to(device), targets.to(device)
    #
    #             # clear the gradients of all optimized variables
    #             optimizer.zero_grad()
    #             # forward pass: compute predicted outputs by passing inputs to the model
    #             outputs = net(inputs)
    #             # calculate the loss
    #             loss = criterion(outputs, targets)
    #             # backward pass: compute gradient of the loss with respect to model parameters
    #             loss.backward(retain_graph=True)
    #             # perform a single optimization step (parameter update)
    #             optimizer.step()
    #
    #             train_loss += loss.item()
    #
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += targets.size(0)
    #             correct += predicted.eq(targets.data).cpu().sum()
    #
    #             progress_bar(batch_idx, len(train_loader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)' %
    #                          (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    #
    #     del inputs, targets, outputs
    #     torch.cuda.empty_cache()
    #
    #     logging.info("Fine-Tuning done.")

    # for fine-tuned models

    # def check_class_wm(self, net, device, match_list, match):
    #
    #     net.eval()
    #
    #     match_tensor_1 = torch.tensor([], dtype=torch.bool).to(device)
    #
    #     for wm_img, wm_lbl in self.wm_loader:
    #         wm_img, wm_lbl = wm_img.to(device), wm_lbl.to(device)
    #
    #         outputs = net(wm_img)
    #         _, predicted = torch.max(outputs.data, 1)
    #
    #         match_tensor_2 = torch.eq(wm_lbl, predicted)
    #         match_tensor_1 = torch.cat([match_tensor_1, match_tensor_2], dim=0)
    #
    #         del wm_img
    #         del wm_lbl
    #         del outputs
    #         del predicted
    #
    #         torch.cuda.empty_cache()
    #
    #     if not match:
    #         match_tensor_1 = ~match_tensor_1
    #
    #     match_list.append(match_tensor_1)
    #
    #     del match_tensor_1
    #     del match_tensor_2
    #     torch.cuda.empty_cache()
    #
    #     return match_list



