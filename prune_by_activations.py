import argparse
import csv
import os
import sys

import torch
from torch import nn
import torch.nn.utils.prune
from torchvision import datasets

import models

from GenericPruningMethod import GenericUnstructured
from activations.calc_avg_activations import ActivationCalculation
from helpers.loaders import get_wm_transform
from helpers.utils import get_trg_set
from trainer import test

parser = argparse.ArgumentParser(description='Load model and visualize activations')

parser.add_argument('--dataset', default='mnist', help='the dataset to test')
parser.add_argument('--arch', metavar='ARCH', default='cnn_mnist')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--loadmodel', default=None, help='the model which should be loaded')
parser.add_argument('--wm_type', default='content', help='e.g. content, noise, unrelated')
parser.add_argument('--model_type', default='clean', help='e.g. clean, backdoor, watermarked')
parser.add_argument('--test_size', default=1000, help='how many trigger images should be tested (default= 1000)')
parser.add_argument('--save_file', default="automated_pruning_results.csv", help='file for saving results')

parser.add_argument('--pruning_rates', nargs='+', default=[0.1], type=float,
                    help='percentages (list) of how many weights to prune')
parser.add_argument('--prune_by_trigger', action='store_true',
                    help="if given, the top-used trigger neurons are pruned instead of the least used clean activation neurons")
parser.add_argument('--save_model', action='store_true', help='save attacked model?')
parser.add_argument('--cuda')

args = parser.parse_args()
device = torch.cuda.current_device() if (torch.cuda.is_available() and torch.cuda.device_count() == 1) else 'cpu'
torch.set_default_device(device)

dataset = args.dataset
criterion = nn.CrossEntropyLoss()

cwd = os.getcwd()
datasets_dict = {'cifar10': datasets.CIFAR10, 'mnist': datasets.MNIST}


def save_results(csv_results, csv_file):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for row in csv_results:
            writer.writerow(row)

        writer.writerow("\n")


def load_net(net, device):
    if device == 'cpu':
        net.load_state_dict(
            torch.load(os.path.join(cwd, 'checkpoint', args.loadmodel + '.pth'), map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(os.path.join(cwd, 'checkpoint', args.loadmodel + '.pth')))

    return net


def prune_weights(layer_to_prune, activations, pruning_rate, prune_by_incoming=False):
    try:
        layer_to_prune
    except NameError as e:
        print(e)
        return

    weights_to_prune = [(layer_to_prune, 'weight')]
    flat_weights = layer_to_prune.weight.flatten()
    weight_amount = int(pruning_rate * len(flat_weights))

    flat_activations = activations.flatten()
    if prune_by_incoming:
        interleaving_factor = len(layer_to_prune.weight)
    else:
        # the input parameters of the layer are the second dimension of the weights
        interleaving_factor = len(layer_to_prune.weight[0])

    flat_activs_reapeated = flat_activations.repeat_interleave(interleaving_factor)

    if args.prune_by_trigger:
        # get the 'amount' highest activations
        weight_topk = torch.topk(torch.abs(flat_activs_reapeated), k=weight_amount, largest=True)
    else:
        # get the 'amount' lowest activations
        weight_topk = torch.topk(torch.abs(flat_activs_reapeated), k=weight_amount, largest=False)

    weight_mask = torch.ones(len(flat_weights))
    weight_mask[weight_topk.indices] = 0

    torch.nn.utils.prune.global_unstructured(weights_to_prune, pruning_method=GenericUnstructured, mask=weight_mask)
    torch.nn.utils.prune.remove(layer_to_prune, "weight")


def prune_bias(layer_to_prune, activations, pruning_rate):
    try:
        layer_to_prune
    except NameError as e:
        print(e)
        return

    bias_to_prune = [(layer_to_prune, 'bias')]
    bias_amount = int(pruning_rate * len(layer_to_prune.bias.flatten()))
    flat_activations = activations.flatten()

    if args.prune_by_trigger:
        # get the 'amount' highest activations
        bias_topk = torch.topk(torch.abs(flat_activations), k=bias_amount, largest=True)
    else:
        # get the 'amount' lowest activations
        bias_topk = torch.topk(torch.abs(flat_activations), k=bias_amount, largest=False)

    bias_mask = torch.ones(len(flat_activations))
    bias_mask[bias_topk.indices] = 0

    # prune bias
    torch.nn.utils.prune.global_unstructured(bias_to_prune, pruning_method=GenericUnstructured, mask=bias_mask)
    torch.nn.utils.prune.remove(layer_to_prune, "bias")


# load already trained model
net = models.__dict__[args.arch](num_classes=args.num_classes)

net = load_net(net, device)

# print(net)

net = net.to(device)
net.eval()

transform = get_wm_transform('ProtectingIP', args.dataset)

# load test set
clean_test_set = datasets_dict[args.dataset](root=os.path.join(cwd, 'data'), train=False, transform=transform,
                                             download=True)
clean_test_loader = torch.utils.data.DataLoader(clean_test_set, batch_size=64, shuffle=False)

# load test set with triggers
trigger_test_set_path = os.path.join(cwd, 'data', 'test_set', 'protecting_ip', args.model_type, args.wm_type,
                                     args.dataset)
trigger_test_set = get_trg_set(trigger_test_set_path, 'labels.txt', int(args.test_size), transform=transform)
trigger_test_loader = torch.utils.data.DataLoader(trigger_test_set, batch_size=64, shuffle=False)

# if args.model_type == 'watermarked':
#     # load test set with triggers
#     extrapol_trigger_test_set_path = os.path.join(cwd, 'data', 'test_set', 'protecting_ip', args.model_type,
#                                                   args.wm_type,
#                                                   args.dataset + '_extrapol')
#     extrapol_trigger_test_set = get_trg_set(extrapol_trigger_test_set_path, 'labels.txt', int(args.test_size),
#                                             transform=transform)
#     extrapol_trigger_test_loader = torch.utils.data.DataLoader(extrapol_trigger_test_set, batch_size=64, shuffle=False)

# specify which layers should be given to activation calculation
if args.arch == 'lenet5':
    layers_to_monitor = [('last_conv', net.conv_layer[4]), ('first_fc', net.fc_layer[0]),
                         ('second_fc', net.fc_layer[2])]
elif args.arch == 'resnet34':
    layers_to_monitor = [('first_fc', net.linear), ('last_conv', net.layer4[2].conv2)]
elif args.arch == "densenet":
    layers_to_monitor = [('first_fc', net.fc), ('last_conv', net.gap)]

activ_calculator = ActivationCalculation(arch=args.arch, device=device)
print('Calculating activations over test set')
clean_activs = activ_calculator.calc_avg_activations(net, layers_to_monitor, clean_test_loader)

print('Calculating activations over trigger set')
trigger_activs = activ_calculator.calc_avg_activations(net, layers_to_monitor, trigger_test_loader)

print('Finished with activation calculation')

print('Calculate difference between trigger images and clean test set')
diff_activs_fst_fc = trigger_activs['first_fc'] - clean_activs['first_fc']
# diff_activs_snd_fc = trigger_activs['second_fc'] - clean_activs['second_fc']

if args.prune_by_trigger:
    activs_for_pruning_fst_fc = diff_activs_fst_fc
    activs_for_pruning_last_conv = clean_activs['last_conv']
    # activs_for_pruning_snd_fc = diff_activs_snd_fc
    path_specification = "_pruned_byTrigger_"
    used_activations = "trigger"
else:
    activs_for_pruning_fst_fc = clean_activs['first_fc']
    activs_for_pruning_last_conv = clean_activs['last_conv']
    # activs_for_pruning_snd_fc = clean_activs['second_fc']
    path_specification = "_pruned_"
    used_activations = "clean"

csv_results = []

for pruning_rate in args.pruning_rates:
    print("Current pruning rate is: " + str(pruning_rate))

    # reload original model
    net = load_net(net, device)
    if args.arch == "lenet5":
        first_layer_to_prune = net.fc_layer[0]
        second_layer_to_prune = net.fc_layer[2]

        # prune second fc layer first, then first fc
        # prune(second_layer_to_prune, activs_for_pruning_snd_fc, pruning_rate)
        prune_weights(first_layer_to_prune, activs_for_pruning_fst_fc, pruning_rate)
        prune_bias(first_layer_to_prune, activs_for_pruning_fst_fc, pruning_rate)

    elif args.arch == "resnet34":
        # TODO select the wanted layer to prune
        first_layer_to_prune = net.linear

        prune_weights(first_layer_to_prune, activs_for_pruning_last_conv, pruning_rate, prune_by_incoming=True)

    elif args.arch == "densenet":
        # TODO select the wanted layer to prune
        first_layer_to_prune = net.fc
        last_conv_to_prune = net.dense4[15].conv1

        prune_weights(first_layer_to_prune, activs_for_pruning_last_conv, pruning_rate, prune_by_incoming=True)

    else:
        sys.exit("given net architecture is not supported")

    # check test_acc
    print('Testing on clean test set.')
    test_acc = test(net, criterion, clean_test_loader, device)
    print("clean acc: %.3f%%" % test_acc)
    # check wm_acc
    print('Testing on original trigger set.')
    trigger_acc = test(net, criterion, trigger_test_loader, device)
    print("trigger acc: %.3f%%" % trigger_acc)

    extrapol_trigger_acc = torch.zeros(1, 1)
    # if args.model_type == 'watermarked':
    #     print('Testing on extrapolated trigger set.')
    #     extrapol_trigger_acc = test(net, criterion, extrapol_trigger_test_loader, device)
    #     print("extrapolated trigger acc: %.3f%%" % extrapol_trigger_acc)

    new_model_name = args.loadmodel + path_specification + str(pruning_rate)
    csv_result_row = [new_model_name, args.dataset, args.model_type, args.wm_type, used_activations, pruning_rate,
                      str(test_acc.detach().numpy()) + " %", str(trigger_acc.detach().numpy()) + " %",
                      str(extrapol_trigger_acc.detach().numpy()) + " %"]
    csv_results.append(csv_result_row)

    # save model
    if args.save_model:
        torch.save(net.state_dict(),
                   os.path.join(cwd, 'checkpoint', args.loadmodel + path_specification + str(pruning_rate) + '.pth'))

print('Saving results.')
save_results(csv_results, os.path.join(cwd, args.save_file))
