"""implementing watermarking methods for"""

import argparse
import time
import traceback

from babel.numbers import format_decimal

from torch.backends import cudnn

import models
import watermarks

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *


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
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cnn_cifar)')

# watermark related
parser.add_argument('--method', default=None, choices=watermarking_methods,
                    help='watermarking method: ' + ' | '.join(
                        watermarking_methods) + ' (default: weakness_into_strength)')
parser.add_argument('--wm_type', default=None, help='wm type for ProtectingIPP: content, unrelated, noise')
parser.add_argument('--save_wm', action='store_true', help='save generated watermarks?')
parser.add_argument('--runname', default='train', help='the exp name')
parser.add_argument('--trg_set_size', default=100, type=int, help='the size of the trigger set (default: 100)')
parser.add_argument('--thresh', default=0.05, type=float, help='threshold for watermark verification')
parser.add_argument('--embed_type', default='', choices=['', 'fromscratch', 'pretrained', 'only_wm', 'augmented'],
                    help='either fromscratch or pretrained or only_wm or augmented')
parser.add_argument('--loadmodel', default='', help='path which model should be load for pretrained embed type')
parser.add_argument('--eps', default=0.1, help='epsilon for FrontierStitching or WMEmbeddedSystems')
parser.add_argument('--lmbda', default=100, help='lambda for PiracyResistant')
parser.add_argument('--pattern_size', default=6, help='patternsize for PiracyResistant and WMEMbeddedSystems')

# hyperparameters
parser.add_argument('--epochs_w_wm', default=0, type=int, help='number of epochs trained with watermarks')
parser.add_argument('--epochs_wo_wm', default=2, type=int, help='number of epochs trained without watermarks')
parser.add_argument('--batch_size', default=64, type=int, help='the batch size')
parser.add_argument('--wm_batch_size', default=32, type=int, help='the wm batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')
parser.add_argument('--patience', default=20, help='early stopping patience (default 20)')

# cuda
parser.add_argument('--cuda', default=None, help='set cuda (e.g. cuda:0)')

# for testing with a smaller subset
parser.add_argument('--test_quot', default=None, type=int,
                    help='the quotient of data subset (for testing reasons; default: None)')

# experiments
parser.add_argument('--trg_set_sizes_list', nargs='+', default=[100], type=int, help='trgset sizes for experiments')
parser.add_argument('--save_file', default="save_results.csv", help='file for saving results')


args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd()

    # create log and config file
    log_dir = os.path.join(cwd, 'log')
    os.makedirs(log_dir, exist_ok=True)
    configfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'conf_' + str(args.runname) + '.txt')
    logfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'log_' + str(args.runname) + '.txt')
    set_up_logger(logfile)

    # create save_dir, results_dir and loss_plots_dir
    save_dir = os.path.join(cwd, 'checkpoint')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(cwd, 'results'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'loss_plots'), exist_ok=True)

    # set up paths for dataset
    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')

    # save configuration parameters
    with open(configfile, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    # set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True

    # load train, valid and test set
    valid_size = 0.1  # https://arxiv.org/abs/1512.03385 uses 0.1 for resnet
    transform_train, transform_test = get_data_transforms(args.dataset)
    train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test,
                                                 valid_size, testquot=args.test_quot)
    train_loader, test_loader, valid_loader = get_dataloader(train_set, test_set, args.batch_size, valid_set, shuffle=True)

    # set up loss
    criterion = nn.CrossEntropyLoss()

except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)

try:
    generation_time = 0
    embedding_time = 0

    runname = args.runname

    for trg_set_size in args.trg_set_sizes_list:

        real_acc, wm_acc, success, false_preds, theta = None, None, None, None, None

        # create new model
        logging.info('Building model. new Model: ' + args.arch)
        net = models.__dict__[args.arch](num_classes=args.num_classes)
        net = net.to(device)

        # set up optimizer and scheduler
        optimizer, scheduler = set_up_optim_sched(args, net)

        args.trg_set_size = trg_set_size
        args.runname = runname + "_" + args.optim + "_" + args.sched + "_" + str(args.trg_set_size)

        logging.info('Training model with watermarks. Method: ' + args.method)
        logging.info('Trigger set size: ' + str(args.trg_set_size))
        # initialize method
        wm_method = watermarks.__dict__[args.method](args)

        # embed watermark
        start_time = time.time()
        real_acc, wm_acc, val_loss, epoch = wm_method.embed(net, criterion, optimizer, scheduler, train_set, test_set,
                                                            train_loader, test_loader, valid_loader, device, save_dir)
        embedding_time = time.time() - start_time
        logging.info("Time for embedding watermarks: %s" % embedding_time)

        # verify watermark
        success, false_preds, theta = wm_method.verify(net, device)

        # save results to csv
        csv_args = [getattr(args, arg) for arg in vars(args)] + [format_decimal(real_acc.item(), locale='de_DE'),
                                                                 format_decimal(wm_acc.item(), locale='de_DE'),
                                                                 bool(success.item()), false_preds.item(), theta,
                                                                 val_loss, epoch+1, generation_time, embedding_time]

        save_results(csv_args, os.path.join(cwd, args.save_file))
        save_obj(wm_method.history, wm_method.save_model)
        zip_checkpoint_dir(save_dir, wm_method.save_model)

        del net
        del optimizer
        del scheduler
        del wm_method


except Exception as e:
    msg = 'An error occurred during training in ' + args.runname + ': ' + str(e)
    logging.error(msg)

    traceback.print_tb(e.__traceback__)