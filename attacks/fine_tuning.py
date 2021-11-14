
from helpers.loaders import get_data_transforms, get_dataset, get_dataloader
from helpers.pytorchtools import EarlyStopping
from trainer import train, test

import os
import logging


def fine_tune(net, device, criterion, optimizer, scheduler, test_loader_orig, dataset, batch_size, num_epochs, patience,
              savefile, wm_loader=None, tune_all=True):
    """
    Run Fine-Tuning Attack on model.
    """

    if dataset == 'cifar10':
        # target_domain = 'cinic10'
        target_domain = 'cinic10-imagenet'
        size_train = 50000  # or half
        size_test = 10000  # or half
    elif dataset == 'mnist':
        target_domain = 'emnist'
        size_train = 60000  # or half
        size_test = 10000  # or half

    # set up paths for dataset
    cwd = os.getcwd()
    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')

    transform_train, transform_test = get_data_transforms(target_domain)
    train_set, test_set, valid_set = get_dataset(target_domain, train_db_path, test_db_path, transform_train,
                                                 transform_test, valid_size=0.1, testquot=None, size_train=size_train, size_test=size_test)
    train_loader, test_loader, valid_loader = get_dataloader(train_set, test_set, batch_size, valid_set)


    history = {}
    avg_train_losses = []
    avg_valid_losses = []
    test_acc_list = []
    wm_acc_list = []
    train_acc_list = []
    valid_acc_list = []
    test_acc_orig_list = []
    best_test_acc, best_epoch, best_wm_acc, best_test_acc_orig, wm_acc = 0, 0, 0, 0, 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=int(patience), verbose=True, path=savefile, trace_func=logging.info)

    # test before train
    logging.info("Testing dataset.")
    test_acc = test(net, criterion, test_loader, device)
    logging.info("Test acc: %.3f%%" % test_acc)
    test_acc_list.append(test_acc)

    logging.info("Testing original dataset.")
    test_acc_orig = test(net, criterion, test_loader_orig, device)
    logging.info("Test acc: %.3f%%" % test_acc_orig)
    test_acc_orig_list.append(test_acc_orig)

    if wm_loader:
        logging.info("Testing trigger set.")
        wm_acc = test(net, criterion, wm_loader, device)
        logging.info("Test acc: %.3f%%" % wm_acc)
        wm_acc_list.append(wm_acc)

    train_acc_list.append(None)
    valid_acc_list.append(None)
    avg_train_losses.append(None)
    avg_valid_losses.append(None)

    # weakness into strength macht 20 epochs
    for epoch in range(num_epochs):

        avg_train_losses, avg_valid_losses, train_acc, valid_acc = train(epoch, net, criterion, optimizer, train_loader,
                                                                         device,
                                                                         avg_train_losses, avg_valid_losses,
                                                                         valid_loader, wmloader=False, tune_all=tune_all)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        logging.info("Testing dataset.")
        test_acc = test(net, criterion, test_loader, device)
        logging.info("Test acc: %.3f%%" % test_acc)
        test_acc_list.append(test_acc)

        logging.info("Testing original dataset.")
        test_acc_orig = test(net, criterion, test_loader_orig, device)
        logging.info("Test acc: %.3f%%" % test_acc_orig)
        test_acc_orig_list.append(test_acc_orig)

        # # replacing the last layer to check the wm resistance
        # new_layer = net.module.linear
        # net, _ = re_initializer_layer(net, 0, private_key)
        # print("WM acc:")
        # test(net, criterion, logfile, wmloader, device)
        #
        # # plugging the new layer back
        # net, _ = re_initializer_layer(net, 0, new_layer)

        if wm_loader:
            logging.info("Testing trigger set.")
            wm_acc = test(net, criterion, wm_loader, device)
            logging.info("Test acc: %.3f%%" % wm_acc)
            wm_acc_list.append(wm_acc)

        if avg_valid_losses[-1] < early_stopping.val_loss_min:  # bc this model will be saved
            best_test_acc = test_acc
            best_wm_acc = wm_acc
            best_test_acc_orig = test_acc_orig
            best_epoch = epoch


        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_losses[-1], net)

        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        scheduler.step()

    history['train_losses'] = avg_train_losses
    history['valid_losses'] = avg_valid_losses
    history['test_acc'] = test_acc_list
    history['train_acc'] = train_acc_list
    history['valid_acc'] = valid_acc_list
    history['test_acc_orig'] = test_acc_orig_list
    history['wm_acc'] = wm_acc_list

    return best_test_acc, early_stopping.val_loss_min, best_wm_acc, best_test_acc_orig, best_epoch, history
