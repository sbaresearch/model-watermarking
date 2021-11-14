"""Some helper functions for PyTorch, including:
    - count_parameters: calculate parameters of network and display as a pretty table.
    - progress_bar: progress bar mimic xlua.progress.
"""
import csv
import os
import sys
import re
import time
import logging
import pickle
import shutil

#from smtplib import SMTP_SSL as SMTP  # this invokes the secure SMTP protocol (port 465, uses SSL)
# from smtplib import SMTP                  # use this for standard SMTP protocol   (port 25, no encryption)


# old version
# from email.MIMEText import MIMEText
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import smtplib

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.autograd import Variable
from torchvision.utils import save_image

from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt

from helpers.image_folder_custom_class import ImageFolderCustomClass

term_width = 80
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    """ creates progress bar for training"""
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    l = list()
    l.append('  Step: %s' % format_time(step_time))
    l.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        l.append(' | ' + msg)

    msg = ''.join(l)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def adjust_learning_rate(init_lr, optimizer, epoch, lradj):
    """Sets the learning rate to the initial LR decayed by 10 every /epoch/ epochs"""
    lr = init_lr * (0.1 ** (epoch // lradj))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# finds a value for theta (maximum number of errors tolerated for verification) (frontier-stitching)
def find_tolerance(key_length, threshold):
    theta = 0
    factor = 2 ** (-key_length)
    s = 0
    # while True:
    #     # for z in range(theta + 1):
    #     s += binomial(key_length, theta)
    #     if factor * s >= threshold:
    #         return theta
    #     theta += 1

    while factor * s < threshold:
        s += binomial(key_length, theta)
        theta += 1

    return theta


# (frontier-stitching)
def binomial(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n-k)):
        b *= n
        b //= t+1
        n -= 1
    return b


def fast_gradient_sign(x, y, model, criterion, device, eps):

    x.requires_grad = True

    outputs = model(x)

    loss = criterion(outputs, y)

    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = x.grad.data

    sign_data_grad = data_grad.sign()

    eps = float(eps)

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = x + eps * sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image


def set_up_logger(file):
    # create custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # format for our loglines
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # setup console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # setup file logging as well
    fh = logging.FileHandler(file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def save_triggerset(trigger_set, path, dataset, wm_type=None):
    logging.info('Saving watermarks.')
    os.makedirs(path, exist_ok=True)

    if wm_type:
        path = os.path.join(path, wm_type)
    path = os.path.join(path, dataset)

    os.makedirs(path, exist_ok=True)

    labels = os.path.join(path, 'labels.txt')
    images = os.path.join(path, 'pics')

    if not os.path.isdir(images):
        os.mkdir(images)

    # maybe: https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty

    if os.path.exists(labels):
        os.remove(labels)

    for idx, (img, lbl) in enumerate(trigger_set):
        save_image(img, os.path.join(images, str(idx+1) + '.jpg'))
        with open(labels, 'a') as f:
            if torch.is_tensor(lbl):
                f.write(str(lbl.item()) + '\n')
            else:
                f.write(str(lbl) + '\n')



def get_channels(dataset):
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'imagenet' or dataset == 'stl10':
        return 3
    elif dataset == 'mnist' or dataset == 'svhn':
        return 1


def get_size(dataset):
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'stl10':
        return 32, 32

    elif dataset == 'mnist' or dataset == 'svhn':
        return 28, 28

    elif dataset == 'imagenet':
        return 256, 256

    else:
        raise NotImplementedError


def image_char(char, image_size, font_size, type):
    if type == "RGB":
        img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    elif type == "BW":
        img = Image.new("L", (image_size, image_size), (255))
    else:
        raise NotImplementedError

    img.getpixel((0, 0)) # brauch ma ned oder?
    draw = ImageDraw.Draw(img)
    font_path = os.path.join(os.getcwd(), "font", "sans_serif.ttf")
    font = ImageFont.truetype(font_path, font_size)

    draw.text((1, 18), char, fill='grey', font=font)
    return img

# WM for embedded systems
def add_watermark(tensor, watermark):
    """Normalize a tensor image with mean and standard deviation.
    See ``Normalize`` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m in zip(tensor, watermark):
        t.add_(m)
    return tensor


# from how_to_prove
def save_loss_acc(epoch, loss, acc, train, save_path, args):

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(np.arange(epoch + 1), loss[0], '-y', label='ste-model loss')
    ax1.plot(np.arange(epoch + 1), loss[1], '-r', label='discriminator loss')
    ax2.plot(np.arange(epoch + 1), acc[0], '-g', label='real_acc')
    ax2.plot(np.arange(epoch + 1), acc[1], '-b', label='wm_acc')

    ax1.set_xlabel('Epoch(' + ",".join(str(l) for l in args.hyper_parameters) + ')')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')

    ax1.set_ylim(0, 5)
    ax2.set_ylim(0, 100)

    ax1.legend(loc=1)
    ax2.legend(loc=2)
    if train:
        plt.savefig(save_path + 'results_train.png')
    else:
        plt.savefig(save_path + 'results_test.png')
    plt.close()


def make_loss_plot(avg_train_losses, avg_valid_losses, runname):
    logging.info("Make loss plot.")
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    # minposs = avg_valid_losses.index(min(avg_valid_losses))+1
    # plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(avg_train_losses)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig('loss_plots/loss_plot_' + runname + '.png', bbox_inches='tight')


def get_trg_set_sizes(dataset):
    if dataset == "cifar10":
        return [50, 100, 200, 400, 500]
    elif dataset == "mnist":
        return [60, 120, 240, 480, 600]


def get_trg_set(path, labels, size, transform=None):
    wm_set = ImageFolderCustomClass(
        path,
        transform)
    img_nlbl = list()
    wm_targets = np.loadtxt(os.path.join(path, labels))
    for idx, (path, target) in enumerate(wm_set.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))

        if idx == (size - 1):
            break

    wm_set.imgs = img_nlbl

    return wm_set


def save_results(csv_args, csv_file):
    logging.info("Saving results.")
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(csv_args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_up_optim_sched(args, net):

    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if args.sched == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120, 140, 160, 180], gamma=args.lradj)
    elif args.sched == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=200)

    return optimizer, scheduler


def save_obj(obj, name):
    with open(os.path.join('results', name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(os.path.join('results', name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def zip_checkpoint_dir(save_dir, save_model):
    dir = os.path.join(save_dir, save_model + '_duringtraining')

    # zip dir
    shutil.make_archive(dir, 'zip', dir)

def re_initializer_layer(model, num_classes, layer=None):
    """remove the last layer and add a new one"""
    indim = model.module.linear.in_features
    private_key = model.module.linear
    if layer:
        model.module.linear = layer
    else:
        model.module.linear = nn.Linear(indim, num_classes).cuda()
    return model, private_key
