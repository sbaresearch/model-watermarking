"""How to prove your model belongs to you: a blind-watermark based framework to protect intellectual property of DNN
(Li et al., 2019)

Based on: https://github.com/zhenglisec/Blind-Watermark-for-DNN
"""

import os
import logging

import numpy as np

import torchvision

from helpers.pytorchtools import EarlyStopping
from helpers.utils import make_loss_plot, save_triggerset
from watermarks.base import WmMethod

from models.HidingUNet import UnetGenerator, UnetGenerator_mnist
from models.Discriminator import DiscriminatorNet, DiscriminatorNet_mnist
from models import SSIM

from helpers.loaders import get_data_transforms, get_dataset
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class HowToProve(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'how_to_prove')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

        self.hyper_parameters = [3, 5, 1, 0.1]

        self.train_loss, self.test_loss = [[], []], [[], []]
        self.train_acc, self.test_acc = [[], []], [[], []]

        self.wm_labels = None
        self.wm_inputs, self.wm_cover_labels = [], []
        self.wm_idx = None
        self.valid = None
        self.fake = None

    def gen_watermarks(self, device, test_loader):

        cwd = os.getcwd()

        # load origin sample as trigger_set
        transform, _ = get_data_transforms(self.dataset)
        trigger_set, _, _ = get_dataset(self.dataset, os.path.join(cwd, 'data'), os.path.join(cwd, 'data'), transform,
                                     transform, valid_size=None, testquot=self.test_quot)
        trigger_loader = torch.utils.data.DataLoader(trigger_set, batch_size=self.wm_batch_size, shuffle=False,
                                                     num_workers=2, drop_last=True)

        if self.dataset == "cifar10":
            # load logo
            ieee_logo = torchvision.datasets.ImageFolder(
                root=os.path.join(cwd, 'data', 'IEEE'), transform=transform)
            ieee_loader = torch.utils.data.DataLoader(ieee_logo, batch_size=1)
            for _, (logo, __) in enumerate(ieee_loader):
                self.secret_img = logo.expand(self.wm_batch_size, logo.shape[1], logo.shape[2], logo.shape[3]).to(device) # .cuda()

        elif self.dataset == "mnist":
            # load logo # todo: wtf?
            for _, (logo, l) in enumerate(test_loader):
                for k in range(self.batch_size):
                    if l[k].cpu().numpy() == 1:
                        logo = logo[k:k + 1]
                        break
                        # TODO whazz up with secret_img?
                self.secret_img = logo.expand(self.wm_batch_size, logo.shape[1], logo.shape[2], logo.shape[3]).to(device)
                break

        # get the watermark-cover images for each batch
        for wm_idx, (wm_input, wm_cover_label) in enumerate(trigger_loader):  # TODO this does not shuffle!
            wm_input, wm_cover_label = wm_input.to(device), wm_cover_label.to(device)
            self.wm_inputs.append(wm_input)
            self.wm_cover_labels.append(wm_cover_label)
            # wm_labels.append(SpecifiedLabel(wm_cover_label))

            # choose trigger set only of specified size
            if wm_idx == (int(self.size / self.wm_batch_size) - 1):
                break

        self.wm_idx = wm_idx

        # Adversarial ground truths
        self.valid = torch.FloatTensor(self.wm_batch_size, 1).fill_(1.0).to(device)
        self.fake = torch.FloatTensor(self.wm_batch_size, 1).fill_(0.0).to(device)

        if self.dataset == 'cifar10' or self.dataset == 'mnist':
            np_labels = np.random.randint(10, size=(int(self.size / self.wm_batch_size), self.wm_batch_size))
        else:
            raise NotImplementedError

        self.wm_labels = torch.from_numpy(np_labels).to(device)

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader,  valid_loader, device, save_dir):

        self.gen_watermarks(device, test_loader)

        logging.info("Buidling additional models: Generator and Discriminator.")
        if self.dataset == 'mnist':
            hidnet = UnetGenerator_mnist()
            disnet = DiscriminatorNet_mnist()
        elif self.dataset == 'cifar10':
            hidnet = UnetGenerator()
            disnet = DiscriminatorNet()

        hidnet.to(device)
        disnet.to(device)

        criterionH_mse = nn.MSELoss()
        criterionH_ssim = SSIM()

        optimizerH = optim.Adam(hidnet.parameters(), lr=self.lr, betas=(0.5, 0.999))
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

        criterionD = nn.BCELoss()
        optimizerD = optim.Adam(disnet.parameters(), lr=self.lr, betas=(0.5, 0.999))
        schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=8, verbose=True)

        # scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

        #patience = 20
        #early_stopping = EarlyStopping(patience=patience, verbose=True)

        self.history['train_loss'] = []
        self.history['val_hloss'] = []
        self.history['val_disloss'] = []
        self.history['val_dnnloss'] = []
        self.history['test_acc'] = []
        self.history['wm_acc'] = []
        self.history['wm_input_acc'] = []


        for epoch in range(self.epochs_w_wm):
            logging.info("Training networks.")

            train_loss = self.train(epoch, device, net, optimizer, criterion, hidnet, optimizerH, criterionH_mse, criterionH_ssim, disnet, optimizerD, criterionD, train_loader)

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            # early_stopping(avg_valid_losses[-1], net)

            logging.info("Testing dataset and watermark.")
            val_hloss, val_disloss, val_dnnloss, real_acc, wm_acc, wm_input_acc = self.test(device, test_loader, epoch, net, optimizer, criterion, hidnet, optimizerH, criterionH_mse, criterionH_ssim, disnet, optimizerD, criterionD)
            logging.info("Test acc: %.3f%%" % real_acc)
            logging.info("WM acc: %.3f%%" % wm_acc)
            logging.info("WM Input acc: %.3f%%" % wm_input_acc)

            # if early_stopping.early_stop:
            #     logging.info("Early stopping")
            #     break

            schedulerH.step(val_hloss)
            schedulerD.step(val_disloss)
            scheduler.step()

            logging.info("Saving models.")
            torch.save(net.state_dict(), os.path.join(save_dir, self.save_model + '.t7'))
            torch.save(hidnet.state_dict(), os.path.join(save_dir, self.save_model + 'hidnet.t7'))
            torch.save(disnet.state_dict(), os.path.join(save_dir, self.save_model + 'disnet.t7'))

            self.history['train_loss'].append(train_loss)
            self.history['val_hloss'].append(val_hloss)
            self.history['val_disloss'].append(val_disloss)
            self.history['val_dnnloss'].append(val_dnnloss)
            self.history['test_acc'].append(real_acc)
            self.history['wm_acc'].append(wm_acc)
            self.history['wm_input_acc'].append(wm_input_acc)

        # todo mache dann plot über history
        # make_loss_plot(avg_train_losses, avg_valid_losses, self.save_model)

        self.create_trigger_set(hidnet)

        real_acc = torch.tensor(real_acc)
        wm_acc = torch.tensor(wm_acc)

        return real_acc, wm_acc, val_dnnloss, epoch


    def train(self, epoch, device, Dnnet, optimizer, criterion, Hidnet, optimizerH, criterionH_mse, criterionH_ssim, Disnet, optimizerD, criterionD, trainloader):
        print('\nEpoch: %d' % epoch)
        Dnnet.train()
        Hidnet.train()
        Disnet.train()
        wm_cover_correct, wm_correct, real_correct, wm_total, real_total = 0, 0, 0, 0, 0

        # clear lists to track next epoch
        train_losses = []
        # valid_losses = []

        loss_H_ = AverageMeter()
        loss_D_ = AverageMeter()
        real_acc = AverageMeter()
        wm_acc = AverageMeter()
        for batch_idx, (input, label) in enumerate(trainloader):
            input, label = input.to(device), label.to(device)
            wm_input = self.wm_inputs[(self.wm_idx + batch_idx) % len(self.wm_inputs)]
            wm_label = self.wm_labels[(self.wm_idx + batch_idx) % len(self.wm_inputs)]
            wm_cover_label = self.wm_cover_labels[(self.wm_idx + batch_idx) % len(self.wm_inputs)]

            #############Discriminator##############
            optimizerD.zero_grad()
            wm_img = Hidnet(wm_input, self.secret_img)
            wm_dis_output = Disnet(wm_img.detach())
            real_dis_output = Disnet(wm_input)
            loss_D_wm = criterionD(wm_dis_output, self.fake)
            loss_D_real = criterionD(real_dis_output, self.valid)
            loss_D = loss_D_wm + loss_D_real
            loss_D.backward()
            optimizerD.step()

            ################Hidding Net#############
            optimizerH.zero_grad()
            optimizerD.zero_grad()
            optimizer.zero_grad()
            wm_dis_output = Disnet(wm_img)
            wm_dnn_output = Dnnet(wm_img)
            loss_mse = criterionH_mse(wm_input, wm_img)
            loss_ssim = criterionH_ssim(wm_input, wm_img)
            loss_adv = criterionD(wm_dis_output, self.valid)

            loss_dnn = criterion(wm_dnn_output, wm_label)
            loss_H = self.hyper_parameters[0] * loss_mse + self.hyper_parameters[1] * (1 - loss_ssim) + \
                     self.hyper_parameters[2] * loss_adv + self.hyper_parameters[3] * loss_dnn
            loss_H.backward()
            optimizerH.step()

            ################DNNet#############
            optimizer.zero_grad()
            inputs = torch.cat([input, wm_img.detach()], dim=0)
            labels = torch.cat([label, wm_label], dim=0)
            dnn_output = Dnnet(inputs)

            loss_DNN = criterion(dnn_output, labels)
            loss_DNN.backward()
            optimizer.step()

            # calculate the accuracy
            wm_cover_output = Dnnet(wm_input)
            _, wm_cover_predicted = wm_cover_output.max(1)
            wm_cover_correct += wm_cover_predicted.eq(wm_cover_label).sum().item()

            _, wm_predicted = dnn_output[self.batch_size: self.batch_size + self.wm_batch_size].max(1)
            wm_correct += wm_predicted.eq(wm_label).sum().item()
            wm_total += self.wm_batch_size

            _, real_predicted = dnn_output[0:self.batch_size].max(1)
            real_correct += real_predicted.eq(
                labels[0:self.batch_size]).sum().item()
            real_total += self.batch_size

            print_msg = (
                '[%d/%d][%d/%d]  Loss D: %.4f Loss_H: %.4f (mse: %.4f ssim: %.4f adv: %.4f)  Loss_real_DNN: %.4f Real acc: %.3f  wm acc: %.3f' % (
                    epoch, self.epochs_w_wm, batch_idx, len(trainloader),
                    loss_D.item(), loss_H.item(), loss_mse.item(
                    ), loss_ssim.item(), loss_adv.item(), loss_DNN.item(),
                    100. * real_correct / real_total, 100. * wm_correct / wm_total))

            loss_H_.update(loss_H.item(), int(input.size()[0]))
            loss_D_.update(loss_D.item(), int(input.size()[0]))
            real_acc.update(100. * real_correct / real_total)
            wm_acc.update(100. * wm_correct / wm_total)

            train_losses.append(loss_DNN.item())

        self.train_loss[0].append(loss_H_.avg)
        self.train_loss[1].append(loss_D_.avg)
        self.train_acc[0].append(real_acc.avg)
        self.train_acc[1].append(wm_acc.avg)
        # save_loss_acc(epoch, train_loss, train_acc, True)  # todo ist in utils jetzt

        train_loss = np.average(train_losses)
        # valid_loss = np.average(valid_losses)

        # print_msg = ('Epoch %d: Train loss: %.3f | Valid loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (epoch, train_loss, valid_loss, 100. * correct / total, correct, total))

        logging.info(print_msg)

        return train_loss

    def test(self, device, test_loader, epoch, Dnnet, optimizer, criterion, Hidnet, optimizerH, criterionH_mse, criterionH_ssim, Disnet, optimizerD, criterionD):
        Dnnet.eval()
        Hidnet.eval()
        Disnet.eval()

        wm_cover_correct, wm_correct, real_correct, real_total, wm_total = 0, 0, 0, 0, 0
        Hlosses = AverageMeter()
        Dislosses = AverageMeter()
        real_acc = AverageMeter()
        wm_acc = AverageMeter()
        DNNlosses = AverageMeter()
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(test_loader):
                input, label = input.to(device), label.to(device)
                wm_input = self.wm_inputs[(self.wm_idx + batch_idx) % len(self.wm_inputs)]
                wm_label = self.wm_labels[(self.wm_idx + batch_idx) % len(self.wm_inputs)]
                wm_cover_label = self.wm_cover_labels[(self.wm_idx + batch_idx) % len(self.wm_inputs)]

                #############Discriminator###############
                wm_img = Hidnet(wm_input, self.secret_img)
                wm_dis_output = Disnet(wm_img.detach())
                real_dis_output = Disnet(wm_input)
                loss_D_wm = criterionD(wm_dis_output, self.fake)
                loss_D_real = criterionD(real_dis_output, self.valid)
                loss_D = loss_D_wm + loss_D_real
                Dislosses.update(loss_D.item(), int(wm_input.size()[0]))

                ################Hidding Net#############
                wm_dnn_outputs = Dnnet(wm_img)
                loss_mse = criterionH_mse(wm_input, wm_img)
                loss_ssim = criterionH_ssim(wm_input, wm_img)
                loss_adv = criterionD(wm_dis_output, self.valid)

                loss_dnn = criterion(wm_dnn_outputs, wm_label)
                loss_H = self.hyper_parameters[0] * loss_mse + self.hyper_parameters[1] * (1 - loss_ssim) + \
                         self.hyper_parameters[2] * loss_adv + self.hyper_parameters[3] * loss_dnn
                Hlosses.update(loss_H.item(), int(input.size()[0]))

                ################DNNet#############
                inputs = torch.cat([input, wm_img.detach()], dim=0)
                labels = torch.cat([label, wm_label], dim=0)
                dnn_outputs = Dnnet(inputs)

                loss_DNN = criterion(dnn_outputs, labels)
                DNNlosses.update(loss_DNN.item(), int(inputs.size()[0]))

                wm_cover_output = Dnnet(wm_input)
                _, wm_cover_predicted = wm_cover_output.max(1)
                wm_cover_correct += wm_cover_predicted.eq(
                    wm_cover_label).sum().item()

                # wm_dnn_output = Dnnet(wm_img)
                # _, wm_predicted = wm_dnn_output.max(1)
                _, wm_predicted = dnn_outputs[self.batch_size:
                                              self.batch_size + self.wm_batch_size].max(1)

                wm_correct += wm_predicted.eq(wm_label).sum().item()
                wm_total += self.wm_batch_size

                _, real_predicted = dnn_outputs[0:self.batch_size].max(1)
                real_correct += real_predicted.eq(
                    labels[0:self.batch_size]).sum().item()
                real_total += self.batch_size

        val_hloss = Hlosses.avg
        val_disloss = Dislosses.avg
        val_dnnloss = DNNlosses.avg
        real_acc.update(100. * real_correct / real_total)
        wm_acc.update(100. * wm_correct / wm_total)
        self.test_acc[0].append(real_acc.avg)
        self.test_acc[1].append(wm_acc.avg)
        print('Real acc: %.3f  wm acc: %.3f wm cover acc: %.3f ' % (
            100. * real_correct / real_total, 100. * wm_correct / wm_total, 100. * wm_cover_correct / wm_total))

        resultImg = torch.cat([wm_input, wm_img, self.secret_img], 0)
        torchvision.utils.save_image(resultImg, os.path.join(self.path, 'Epoch_' + str(epoch) + '_img.png'),
                                     nrow=self.wm_batch_size,
                                     padding=1, normalize=True)
        self.test_loss[0].append(val_hloss)
        self.test_loss[1].append(val_disloss)

        # save_loss_acc(epoch, test_loss, test_acc, False)  # ist in utils jetzt
        # save
        real_acc = 100. * real_correct / real_total
        wm_acc = 100. * wm_correct / wm_total
        wm_input_acc = 100. * wm_cover_correct / wm_total

        logging.info("Saving models, hidnet, disnet and dnnet.")
        torch.save(Hidnet.state_dict(), os.path.join('checkpoint', self.save_model + '_hidnet.pt'))
        torch.save(Disnet.state_dict(), os.path.join('checkpoint', self.save_model + '_disnet.pt'))
        torch.save(Dnnet.state_dict(), os.path.join('checkpoint', self.save_model + '_dnnet.pt'))

        return val_hloss, val_disloss, val_dnnloss, real_acc, wm_acc, wm_input_acc

    def create_trigger_set(self, hidnet):

        # wm_cover_lbls sind die wahren labels von den images, wm_lbls sind die fake labels.

        for i in range(len(self.wm_inputs)):
            wm_input = self.wm_inputs[i]
            wm_label = self.wm_labels[i]
            wm_cover_label = self.wm_cover_labels[i]

            wm_img = hidnet(wm_input, self.secret_img)

            for j in range(wm_img.shape[0]):
                self.trigger_set.append((wm_img[j], wm_label[j]))

                if len(self.trigger_set) >= self.size:
                    break

        if self.save_wm:
            save_triggerset(self.trigger_set, self.path, self.dataset)
