import sys

import torch

class ActivationCalculation:

    def __init__(self, arch, device):
        self.arch = arch
        self.device = device
        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    """ layers_to_monitor is an array of tupels, e.g.:
    [('last_conv', net.conv_layer[4])]
    """
    def calc_avg_activations(self, net, layers_to_monitor, img_loader):
        # reset activations and register forward hook for each given layer
        for name, layer in layers_to_monitor:
            self.activation[name] = []
            layer.register_forward_hook(self.get_activation(name))

        with torch.no_grad():
            layer_outputs = {}
            for name, layer in layers_to_monitor:
                layer_outputs[name] = []

            for batch_idx, (inputs, targets) in enumerate(img_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # the real outputs are not used, next command is needed to get the activation
                # from the before specified hooks
                net(inputs)

                batch_activations = {}
                for name, layer in layers_to_monitor:
                    batch_activations[name] = self.activation[name]
                    layer_outputs[name].append(torch.mean(batch_activations[name], 0))

            avg_activations = {}
            for name, layer in layers_to_monitor:
                avg_activations[name] = torch.mean(torch.stack(layer_outputs[name]), 0)

            return avg_activations

