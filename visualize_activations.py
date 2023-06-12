import argparse
import math
import os

from matplotlib import pyplot as plt, gridspec
from matplotlib.figure import figaspect
import torch.nn.utils.prune
from torchvision import datasets

import models
import torchvision.transforms as transforms

from activations.calc_avg_activations import ActivationCalculation
from helpers.loaders import get_wm_transform
from helpers.utils import get_trg_set

parser = argparse.ArgumentParser(description='Load model and visualize activations')

parser.add_argument('--dataset', default='mnist', help='the dataset to test')
parser.add_argument('--arch', metavar='ARCH', default='cnn_mnist')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--loadmodel', default=None, help='the model which should be loaded')
parser.add_argument('--wm_type', default='content', help='e.g. content, noise, unrelated')
parser.add_argument('--model_type', default='clean', help='e.g. clean, backdoor, watermarked')
parser.add_argument('--test_size', default=1000, help='how many trigger images should be tested (default= 1000)')

parser.add_argument('--cuda')

args = parser.parse_args()
device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'
dataset = args.dataset

cwd = os.getcwd()
datasets_dict = {'cifar10': datasets.CIFAR10, 'mnist': datasets.MNIST}


def plot_avg_activations(activations, plot_title):
    w, h = figaspect(1)
    fig = plt.figure(figsize=(w, h))
    fig.tight_layout()
    # e.g 256 neurons -> 16*16 grid
    dims = int(math.sqrt(len(activations)))
    gs = gridspec.GridSpec(dims, dims)
    # gs = gridspec.GridSpec(2, 3)
    # gs.update(wspace=0.0, hspace=0.0, left=0.06, right=0.52, top=0.88, bottom=0.1)

    for i in range(len(activations)):
        a = fig.add_subplot(gs[i])
        plt.imshow(activations[i].detach().numpy(), cmap='gray')
        # plt.imshow(activations[i].detach().numpy())
        a.axis('off')
        a.set_xticklabels([])
        a.set_yticklabels([])

    cax = plt.axes([0.9, 0.1, 0.015, 0.78])
    plt.colorbar(cax=cax)
    fig.suptitle(plot_title)
    plt.show()


# load already trained model
net = models.__dict__[args.arch](num_classes=args.num_classes)
if device == 'cpu':
    net.load_state_dict(
        torch.load(os.path.join(cwd, 'checkpoint', args.loadmodel + '.pth'), map_location=torch.device('cpu')))
else:
    net.load_state_dict(torch.load(os.path.join(cwd, 'checkpoint', args.loadmodel + '.pth')))

net = net.to(device)
net.eval()

transform = get_wm_transform('ProtectingIP', args.dataset)

# load test set
clean_test_set = datasets_dict[args.dataset](root=os.path.join(cwd, 'data'), train=False, transform=transform, download=True)
clean_test_loader = torch.utils.data.DataLoader(clean_test_set, batch_size=64, shuffle=False)

# load test set with triggers
trigger_test_set_path = os.path.join(cwd, 'data', 'test_set', 'protecting_ipp', args.model_type, args.wm_type,
                                     args.dataset)
trigger_test_set = get_trg_set(trigger_test_set_path, 'labels.txt', int(args.test_size), transform=transform)
trigger_test_loader = torch.utils.data.DataLoader(trigger_test_set, batch_size=64, shuffle=False)

print('Calculating activations over test set')
activ_calculator = ActivationCalculation(arch=args.arch, device=device)

if args.arch == 'lenet5':
    layers_to_monitor = [('last_conv', net.conv_layer[4])]

clean_activs = activ_calculator.calc_avg_activations(net, layers_to_monitor, clean_test_loader)

print('Calculating activations over trigger set')
trigger_activs = activ_calculator.calc_avg_activations(net, layers_to_monitor, trigger_test_loader)
print('Finished with activation calculation')

plot_avg_activations(clean_activs['last_conv'], plot_title="Average activations of last conv layer\n on clean test data")
plot_avg_activations(trigger_activs['last_conv'], plot_title="Average activations of last conv layer\n on trigger test data")

print('Calculate difference between trigger images and clean test set')
diff_activs_last_conv = trigger_activs['last_conv'] - clean_activs['last_conv']
# only for plotting purposes
diff_plot_last_conv = diff_activs_last_conv.apply_(lambda x: 0 if x < 0 else x)

plot_avg_activations(diff_plot_last_conv,
                     plot_title="Average activations of neurons that\n only activate on trigger test data")
