'''Provides train and test function'''

import os
import logging

import torch
import numpy as np

from helpers.utils import progress_bar, make_loss_plot, save_obj

from helpers.pytorchtools import EarlyStopping


def train(epoch, net, criterion, optimizer, train_loader, device, avg_train_losses, avg_valid_losses,
          valid_loader=False, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    net.train()

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    train_acc = 0
    valid_acc = 0

    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print('\nBatch: %d' % batch_idx)
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = net(inputs)

        # calculate the loss
        loss = criterion(outputs, targets)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward(retain_graph=True)
        # perform a single optimization step (parameter update)
        optimizer.step()
        # record training loss
        train_losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        train_acc = 100. * correct / total

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                     (np.average(train_losses), train_acc, correct, total))

    ######################
    # validate the model #
    ######################
    if valid_loader:
        correct = 0
        total = 0
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = net(inputs)
                # calculate the loss
                loss = criterion(outputs, targets)
                # record validation loss
                valid_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                valid_acc = 100. * correct / total

    # print training / validation statistics
    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    logging.info(('Epoch %d: Train loss: %.3f | Valid loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (epoch, train_loss, valid_loss, train_acc, correct, total)))

    return avg_train_losses, avg_valid_losses, train_acc, valid_acc


def test(net, criterion, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    logging.info('Test results: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total


def train_wo_wms(epochs, net, criterion, optimizer, scheduler, patience, train_loader, test_loader, valid_loader,
                 device,
                 save_dir, save_model, history=dict()):
    logging.info("Training model without watermarks.")

    avg_train_losses = []
    avg_valid_losses = []
    test_acc_list = []
    best_test_acc, best_epoch = 0, 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=int(patience), verbose=True,
                                   path=os.path.join(save_dir, save_model + '.pth'),
                                   trace_func=logging.info)

    for epoch in range(epochs):

        avg_train_losses, avg_valid_losses, train_acc, valid_acc = train(epoch, net, criterion, optimizer, train_loader,
                                                                         device,
                                                                         avg_train_losses, avg_valid_losses,
                                                                         valid_loader)

        logging.info("Testing dataset.")
        test_acc = test(net, criterion, test_loader, device)
        logging.info("Test acc: %.3f%%" % test_acc)

        test_acc_list.append(test_acc)

        if avg_valid_losses[-1] < early_stopping.val_loss_min:  # bc this model will be saved
            best_test_acc = test_acc
            best_epoch = epoch

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_losses[-1], net)

        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        scheduler.step()

    # make_loss_plot(avg_train_losses, avg_valid_losses, save_model)

    history['train_losses'] = avg_train_losses
    history['valid_losses'] = avg_valid_losses
    history['test_acc'] = test_acc_list
    history['train_acc'] = train_acc
    history['valid_acc'] = valid_acc

    # torch.save(torch.tensor([avg_train_losses, avg_valid_losses, test_acc_list]),
    #            os.path.join('results', save_model + '.pth'))

    return best_test_acc, early_stopping.val_loss_min, best_epoch, history


def train_on_wms(epochs, device, net, optimizer, criterion, scheduler, wm_loader, test_loader, save_dir, save_model,
                 history):
    logging.info("Training model only on trigger set.")

    avg_train_losses = []
    avg_valid_losses = []
    test_acc_list = []
    wm_acc_list = []

    durtr_dir = os.path.join(save_dir, save_model + '_duringtraining')
    os.makedirs(durtr_dir, exist_ok=True)

    for epoch in range(epochs):

        # no valid_loader for wm_loader
        avg_train_losses, avg_valid_losses, train_acc, valid_acc = train(epoch, net, criterion, optimizer, wm_loader,
                                                                         device,
                                                                         avg_train_losses, avg_valid_losses)

        logging.info("Testing dataset.")
        test_acc = test(net, criterion, test_loader, device)
        logging.info("Test acc: %.3f%%" % test_acc)

        test_acc_list.append(test_acc)

        logging.info("Testing triggerset (no train, test split).")
        wm_acc = test(net, criterion, wm_loader, device)
        logging.info("WM acc: %.3f%%" % wm_acc)

        wm_acc_list.append(wm_acc)

        # save model every 5 epochs # could be variable
        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join(durtr_dir, save_model + 'epoch_' + str(epoch + 1) + '.pth'))

        scheduler.step()

    logging.info("Saving model.")
    torch.save(net.state_dict(), os.path.join(save_dir, save_model + '.pth'))

    # history = {'train_losses': avg_train_losses,
    #            'valid_losses': avg_valid_losses,
    #            'test_acc': test_acc_list,
    #            'wm_acc': wm_acc_list}

    history['train_losses'] = avg_train_losses
    history['valid_losses'] = avg_valid_losses
    history['test_acc'] = test_acc_list
    history['wm_acc'] = wm_acc_list
    history['train_acc'] = train_acc
    history['valid_acc'] = valid_acc

    # save_obj(history, save_model)
    # torch.save(torch.tensor([avg_train_losses, avg_valid_losses, test_acc_list, wm_acc_list]),
    #            os.path.join('results', save_model + '.pth'))

    return test_acc, wm_acc, None, epoch, history


def train_on_augmented(epochs, device, net, optimizer, criterion, scheduler, patience, train_loader, test_loader,
                       valid_loader,
                       wm_loader, save_dir, save_model, history):
    logging.info("Training on dataset augmented with trigger set.")

    avg_train_losses = []
    avg_valid_losses = []
    test_acc_list = []
    wm_acc_list = []

    best_test_acc, best_wm_acc, best_epoch = 0, 0, 0

    early_stopping = EarlyStopping(patience=int(patience), verbose=True,
                                   path=os.path.join(save_dir, save_model + '.pth'),
                                   trace_func=logging.info)

    durtr_dir = os.path.join(save_dir, save_model + '_duringtraining')
    os.makedirs(durtr_dir, exist_ok=True)

    for epoch in range(epochs):
        avg_train_losses, avg_valid_losses, train_acc, valid_acc = train(epoch, net, criterion, optimizer,
                                                                         train_loader, device,
                                                                         avg_train_losses, avg_valid_losses,
                                                                         valid_loader, wm_loader)

        logging.info("Testing dataset.")
        test_acc = test(net, criterion, test_loader, device)
        logging.info("Test acc: %.3f%%" % test_acc)

        test_acc_list.append(test_acc)

        logging.info("Testing triggerset (no train, test split).")
        wm_acc = test(net, criterion, wm_loader, device)
        logging.info("WM acc: %.3f%%" % wm_acc)

        wm_acc_list.append(wm_acc)

        if avg_valid_losses[-1] < early_stopping.val_loss_min:  # bc this model will be saved
            best_test_acc = test_acc
            best_wm_acc = wm_acc
            best_epoch = epoch

        # save model every 5 epochs # could be variable
        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join(durtr_dir, save_model + 'epoch_' + str(epoch + 1) + '.pth'))

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_losses[-1], net)

        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        scheduler.step()

    # make_loss_plot(avg_train_losses, avg_valid_losses, save_model)

    history['train_losses'] = avg_train_losses
    history['valid_losses'] = avg_valid_losses
    history['test_acc'] = test_acc_list
    history['wm_acc'] = wm_acc_list
    history['train_acc'] = train_acc
    history['valid_acc'] = valid_acc

    # save_obj(history, save_model)

    # torch.save(torch.tensor([avg_train_losses, avg_valid_losses, test_acc_list, wm_acc_list]),
    #           os.path.join('results', save_model + '.pth'))

    return best_test_acc, best_wm_acc, early_stopping.val_loss_min, best_epoch, history
