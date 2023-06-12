""" generate watermarks """

import argparse
import time
import traceback

from babel.numbers import format_decimal

# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.backends import cudnn

import models
import watermarks

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *

from trainer import train_wo_wms

# possible models to use
model_names = sorted(name for name in models.__dict__ if name.islower() and callable(models.__dict__[name]))
print('models : ', model_names)

# possible watermarking methods to use
watermarking_methods = sorted(
    watermark for watermark in watermarks.__dict__ if callable(watermarks.__dict__[watermark]))
print('watermarks: ', watermarking_methods)

# set up argument parser
parser = argparse.ArgumentParser(description='Train models with watermarks.')

# model and dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--arch', metavar='ARCH', default='cnn_cifar10', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cnn_cifar10)')

# watermark related
parser.add_argument('--method', default=None, choices=watermarking_methods,
                    help='watermarking method: ' + ' | '.join(
                        watermarking_methods) + ' (default: weakness_into_strength)')
parser.add_argument('--wm_type', default=None, help='wm type for ProtectingIP: content, unrelated, noise')
parser.add_argument('--save_wm', action='store_true', help='save generated watermarks?')
parser.add_argument('--runname', default='train', help='the exp name')
parser.add_argument('--trg_set_size', default=100, type=int, help='the size of the trigger set (default: 100)')
parser.add_argument('--thresh', default=0.05, type=float, help='threshold for watermark verification')
parser.add_argument('--embed_type', default='', choices=['', 'fromscratch', 'pretrained', 'only_wm', 'augmented'],
                    help='either fromscratch or pretrained or only_wm or augmented')
parser.add_argument('--loadmodel', default='', help='path which model should be load for pretrained embed type')
parser.add_argument('--eps', default=0.1, help='epsilon for frontier stitching')
parser.add_argument('--lmbda', default=100, help='lambda for piracy resistant')
parser.add_argument('--pattern_size', default=6, help='patternsize for piracy resistant')

parser.add_argument('--test_quot', default=None, type=int,
                    help='the quotient of data subset (for testing reasons; default: None)')


parser.add_argument('--backdoor', default=None, help='specify wether watermark or backdoor trigger-set should be used')

# hyperparameters
parser.add_argument('--epochs_w_wm', default=0, type=int, help='number of epochs trained with watermarks')
parser.add_argument('--epochs_wo_wm', default=2, type=int, help='number of epochs trained without watermarks')
parser.add_argument('--batch_size', default=32, type=int, help='the batch size')
parser.add_argument('--wm_batch_size', default=16, type=int, help='the wm batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')
parser.add_argument('--patience', default=20, help='early stopping patience (default 20)')

# cuda
parser.add_argument('--cuda', default=None, help='set cuda (e.g. cuda:0)')

parser.add_argument('--save_file', default="save_results.csv", help='file for saving results')

args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd()

    # set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

except Exception as e:
    msg = 'An error occured during setup: ' + str(e)

try:
    generation_time = 0

    wm_method = watermarks.__dict__[args.method](args)

    if args.method == 'ProtectingIP':

        start_time = time.time()
        wm_method.gen_watermarks(device)
        generation_time = time.time() - start_time

    elif args.method == 'PiracyResistant':

        transform = get_wm_transform('PiracyResistant', args.dataset)
        dataset, _, _ = get_dataset(args.dataset, os.path.join(cwd, 'data'), os.path.join(cwd, 'data'),
                                    transform, transform, valid_size=None,
                                    testquot=None, size_train=args.trg_set_size, size_test=args.trg_set_size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.wm_batch_size, num_workers=0, shuffle=False,
                                             drop_last=True)

        start_time = time.time()
        wm_method.gen_watermarks(loader)
        generation_time = time.time() - start_time

    elif args.method == 'ExponentialWeighting':

        start_time = time.time()
        wm_method.gen_watermarks()
        generation_time = time.time() - start_time

    elif args.method == 'FrontierStitching':
        net = models.__dict__[args.arch](num_classes=args.num_classes)
        net.load_state_dict(torch.load(os.path.join('checkpoint', args.loadmodel + '.pth')))
        net.to(device)

        criterion = nn.CrossEntropyLoss()

        _, transform = get_data_transforms(args.dataset)
        dataset, _, _ = get_dataset(args.dataset, os.path.join(cwd, 'data'), os.path.join(cwd, 'data'),
                                    transform, transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.wm_batch_size, num_workers=0, shuffle=False,
                                             drop_last=True)

        start_time = time.time()
        wm_method.gen_watermarks(net, criterion, device, loader, args.eps)
        generation_time = time.time() - start_time

    elif args.method == 'WMEmbeddedSystems':
        transform = get_wm_transform('WMEmbeddedSystems', args.dataset)
        dataset, _, _ = get_dataset(args.dataset, os.path.join(cwd, 'data'), os.path.join(cwd, 'data'),
                                    transform, transform)

        start_time = time.time()
        wm_method.gen_watermarks(dataset, device)
        generation_time = time.time() - start_time

    csv_args = [args.method, args.wm_type, args.dataset, args.arch, generation_time]

    with open(args.save_file, 'a') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(csv_args)

except Exception as e:
    msg = 'An error occured during watermark generation in ' + args.runname + ': ' + str(e)

    traceback.print_tb(e.__traceback__)
