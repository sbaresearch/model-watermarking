""" Evolutionary Trigger Set Generation for DNN Black-Box Watermarking (Guo and Potkonjak, 2019)

- based on trigger patterns, we use DE algorithm to find the "best" place for the pattern in the image (see paper)

- take two different trigger patterns: KEY - from Guo et al: WM embedded systems and LOGO - from Zhang et al.

--> paper extrem schlecht beschrieben!
"""
import copy
import pickle

from watermarks.base import WmMethod

import os
import random
import logging
import numpy as np
import heapq
import time

from copy import deepcopy  # , copy
# import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
# import torchvision
# from kmeans_pytorch import kmeans
# from torch.utils.data.dataset import Subset
# from torch.utils.data import DataLoader
# from torch.autograd import Variable

from helpers.loaders import get_dataset, get_dataloader, get_data_transforms, get_wm_transform

from trainer import test, train_on_augmented
# from helpers.loaders import get_data_transforms
from helpers.utils import add_watermark, save_triggerset, find_tolerance, image_char, get_size, get_channels, \
    get_trg_set
from helpers.transforms import RandomWatermark, EmbedText


# from project_datasets import Cifar10Meta, Cifar100Meta, MnistMeta


class EvolutionaryGen(WmMethod):
    def __init__(self, args):
        super().__init__(args)
        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'evolutionary_gen')  # path where to save
        os.makedirs(self.path, exist_ok=True)
        # trigger set if has to be generated
        self.wm_type = args.wm_type  # logo or key
        self.watermarks = []  # list of dicts containing information for every single watermark candidate in the
        self.watermark = None  # final watermark, best solution in gen_watermarks
        # evolving process
        self.evolve = closest_point_evolve if self.wm_type == "key" else straightforward_evolve
        self.num_bits = self.pattern_size

    def gen_watermarks(self, model, criterion, train_set, device, iters, threshold=0.999):

        logging.info('Generating watermarks. Type = ' + self.wm_type)
        # randomly initialize candidates
        if self.wm_type == 'key':
            # this is the pattern that was implemented in >>WMEmbeddedSystems<<
            # create n-bit signature
            if self.dataset == "cifar10" or self.dataset == "cifar100":  # TODO check ob wirklich für cifar100 auch so
                # the CIFAR10 variant includes 64 pixels and encodes 128 bits of information. # TODO: 64 pixels??
                # every pixel in the RGB color space with v_k= +/- 100 encodes 2 bits, and the message can be decoded by
                # reading the pixels from left to right, top to bottom.

                # create 4 candidates
                num_candidates = 4  # TODO make variable number of candidates
                for i in range(num_candidates):
                    raw_watermark = np.zeros([32 * 32], dtype=np.float32)
                    raw_watermark[random.sample(range(32 * 32), self.num_bits)] = 1.
                    raw_watermark = raw_watermark.reshape([32, 32])

                    strength = (1. / num_candidates) * (i + 1)
                    # create message mark of magnitude strength. # np.array([1.221, 1.221, 1.301])
                    watermark = np.array([1, 1, 1])[:, np.newaxis, np.newaxis] * raw_watermark[np.newaxis,
                                                                                             :, :] * strength

                    msg = np.reshape(watermark, [-1, 2])
                    pos = msg.nonzero()
                    pos = [np.asarray([pos[0][i], pos[1][i]]) for i in range(len(pos[0]))]

                    watermark_dict = {"pattern": watermark,
                                      "pos": pos,
                                      "shape": watermark.shape,
                                      "type": self.wm_type,
                                      "strength": strength,
                                      "evolve_size": 10,
                                      "fitness": None}

                    self.watermarks.append(watermark_dict)

            elif self.dataset == "mnist":
                # the MNIST variant includes 192 pixels and encodes 192-bit information, with v_k = 100,200 to encode 0
                # and 1 respectively.

                # this is the pattern that was implemented in >>WMEmbeddedSystems<<
                # create n-bit signature

                # create n candidates -> go with 4
                num_candidates = 4
                for i in range(num_candidates):
                    raw_watermark = np.zeros([28 * 28], dtype=np.float32)
                    raw_watermark[random.sample(range(28 * 28), self.num_bits)] = 1.
                    raw_watermark = raw_watermark.reshape([28, 28])

                    strength = (1. / num_candidates) * (i + 1)

                    # create message mark of magnitude strength.
                    watermark = raw_watermark * strength
                    watermark = watermark.reshape((1, watermark.shape[0], watermark.shape[1]))
                    pos = watermark.nonzero()
                    pos = [np.asarray([pos[1][i], pos[2][i]]) for i in range(len(pos[0]))]

                    watermark_dict = {"pattern": watermark,
                                      "pos": pos,
                                      "shape": watermark.shape,
                                      "type": self.wm_type,
                                      "strength": strength,
                                      "evolve_size": 10,
                                      "fitness": None}

                    self.watermarks.append(watermark_dict)

        elif self.wm_type == 'logo':
            # this is the content pattern that was implemented in ProtectingIPP content

            if self.dataset == "cifar10" or self.dataset == "cifar100":
                # create 4 candidates
                num_candidates = 4  # TODO make variable number of candidates
                positions = [(0, -2), (6, -2), (6, 22), (0, 22)]
                for i in range(num_candidates):
                    strength = (1. / num_candidates) * (i + 1)

                    ch = get_channels(self.dataset)
                    w, h = get_size(self.dataset)
                    watermark_dict = {"pattern": "TEST",  # text that will be embedded on the img, alternatively image
                                      "pos": positions[i],  # position of upper left corner for image
                                      "shape": (ch, w, h),
                                      "type": self.wm_type,
                                      "strength": strength,
                                      "evolve_size": 10,  # TODO check this, ist das iters?
                                      "fitness": None}
                    self.watermarks.append(watermark_dict)

            elif self.dataset == "mnist":
                # create 4 candidates
                num_candidates = 5  # TODO make variable number of candidates
                positions = [(0, -2), (2, -2), (2, 18), (0, 18)]
                for i in range(num_candidates):
                    # raw_image = image_char("TEST", get_size(self.dataset)[0], 10, "BW")  # maybe: change to "simpler" text, see paper.
                    # raw_watermark = transforms.ToTensor()(raw_image)

                    strength = (1. / num_candidates) * (i + 1)
                    # create message mark of magnitude strength.
                    # watermark = raw_watermark * strength

                    # msg = np.reshape(watermark, [-1])
                    # pos = watermark.nonzero()
                    # pos = [np.asarray([pos[0][i], pos[1][i]]) for i in range(len(pos[0]))]

                    ch = get_channels(self.dataset)
                    w, h = get_size(self.dataset)
                    watermark_dict = {"pattern": "TEST",  # text that will be embedded on the img, alternatively image
                                      "pos": positions[i],  # position of upper left corner for image
                                      "shape": (ch, w, h),
                                      "type": self.wm_type,
                                      "strength": strength,
                                      "evolve_size": 10,  # TODO check this, ist das iters?
                                      "fitness": None}
                    self.watermarks.append(watermark_dict)

        path_watermarked = os.path.join(self.path, 'watermarked')
        if not os.path.isdir(path_watermarked):
            os.mkdir(path_watermarked)

        fitness = []
        meta = []
        for wm in self.watermarks:
            wm_fitness, wm_meta = evaluate(wm, self.dataset, train_set, path_watermarked, self.test_quot,
                                           self.wm_batch_size, model,
                                           criterion, device)
            wm["fitness"] = wm_fitness
            fitness.append(wm_fitness)
            meta.append(wm_meta)

        fitness = torch.tensor(fitness)
        fitness.to(device)
        meta = torch.tensor(meta)
        meta.to(device)

        logging.info("Starting Differential Evolution.")
        # for each generation
        for iteration in range(iters):
            logging.info("Iteration: " + str(iteration))

            # Early Stopping
            maximum = torch.max(fitness).item()

            if maximum > threshold:
                break

            num_watermarks = len(self.watermarks)
            # for each candidate
            for i in range(num_watermarks):
                # randomly pick j, k, l
                j, k, l = np.random.choice(num_watermarks, 3, replace=False)
                new_candidate = self.evolve(self.watermarks[j], self.watermarks[k], self.watermarks[l])
                new_fitness, new_meta = evaluate(new_candidate, self.dataset, train_set, path_watermarked,
                                                 self.test_quot,
                                                 self.wm_batch_size,
                                                 model, criterion, device)

                if new_fitness.item() > fitness[i]:
                    self.watermarks[i] = new_candidate
                    fitness[i] = new_fitness.item()

        # key: candidate will be an array of K tuples {c_k^x, c_k^y}^K
        best_idx = fitness.argmax()
        best_solution = self.watermarks[best_idx]
        best_score = fitness[best_idx]

        self.watermark = best_solution

        transform = get_wm_transform('EvolutionaryGen', self.dataset)

        train_set, _, _ = get_dataset(self.dataset, self.path, self.path, transform, transform,
                                      valid_size=None, testquot=None, size=self.size)

        for i in random.sample(range(len(train_set)), len(train_set)):  # iterate randomly
            img, lbl = train_set[i]
            img = img.to(device)

            if len(self.trigger_set) == self.size:
                break  # break for-loop when triggerset has final size

            if self.wm_type == "logo":
                img = EmbedText(wm["pattern"], wm["pos"], wm["strength"])(img)
            elif self.wm_type == "key":
                img.add_(torch.from_numpy(self.watermark["pattern"]).to(device))

            trg_lbl = (lbl + 1) % self.num_classes  # set trigger labels label_watermark=lambda w, x: (x + 1) % 10
            self.trigger_set.append((img, torch.tensor(trg_lbl).to(device)))

        if self.save_wm:
            save_triggerset(self.trigger_set, os.path.join(self.path, self.arch, str(iters)), self.dataset,
                            self.wm_type)
            # save also pattern
            path = os.path.join(self.path, self.arch, str(iters), self.wm_type, self.dataset)
            with open(os.path.join(path, 'pattern.pkl'), 'wb') as f:
                pickle.dump(wm["pattern"], f, pickle.HIGHEST_PROTOCOL)

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):
        logging.info("Embedding watermarks.")

        logging.info('Loading pretrained model.')
        # load model
        net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel + '.t7')))

        # self.gen_watermarks(net, criterion, train_set, device, 2, 0.999)  # todo iters and threshold variable?
        transform = get_wm_transform('EvolutionaryGen', self.dataset)
        self.trigger_set = get_trg_set(os.path.join(self.path, self.arch, self.wm_type, self.dataset), 'labels.txt',
                                       self.size, transform=transform)

        self.loader()

        real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer,
                                                                             criterion,
                                                                             scheduler, self.patience, train_loader,
                                                                             test_loader,
                                                                             valid_loader, self.wm_loader, save_dir,
                                                                             self.save_model,
                                                                             self.history)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch


def evaluate(wm, dataset, train_set, path_watermarked, test_quot, batch_size, net, criterion, device):
    # evaluate the accuracy of candidates on a random set of 640 training images
    watermarked_set = torch.utils.data.Subset(train_set, random.sample(range(len(train_set)), 640))

    if wm["type"] == "key":
        if dataset == "cifar10" or dataset == "cifar100":
            transform_watermarked = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                RandomWatermark(wm["pattern"].astype(np.float32), probability=1.0),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == "mnist":
            transform_watermarked = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                RandomWatermark(wm["pattern"].astype(np.float32), probability=1.0),
            ])
    elif wm["type"] == "logo":
        if dataset == "cifar10" or dataset == "cifar100":
            transform_watermarked = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                EmbedText(wm["pattern"], wm["pos"], wm["strength"]),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif dataset == "mnist":
            transform_watermarked = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                EmbedText(wm["pattern"], wm["pos"], wm["strength"]),
            ])
    else:
        raise NotImplementedError

    if test_quot:
        watermarked_set.dataset.transform = transform_watermarked
    else:
        watermarked_set.transform = transform_watermarked

    num_classes = 10 if (dataset == 'mnist' or dataset == 'cifar10') else 100 if (dataset == 'cifar100') else None
    # if type(watermarked_set) == 'torch.utils.data.dataset.Subset':
    watermarked_set.dataset.targets = [(lbl + 1) % num_classes for lbl in watermarked_set.dataset.targets]
    # else:
    #     watermarked_set.targets = [(lbl + 1) % num_classes for lbl in watermarked_set.targets]

    watermarked_loader = torch.utils.data.DataLoader(watermarked_set, batch_size=batch_size, shuffle=True)

    acc = test(net, criterion, watermarked_loader, device) / 100.
    str_coef = 0.1
    # x0.2 because candidate strength is between 0 and 5 (normalized by std of around 0.2)
    fitness = acc * (1 - str_coef) + wm["strength"] * 0.2 * str_coef
    meta = (acc, wm["strength"])
    return fitness, meta


def analyze(iteration, meta, timestamp, fitness):
    max_idx = np.argmax(fitness)
    msg = "[Iteration %.3d], Time Elapsed %.3fs, Fitness %.7f, " % (
        iteration, time.time() - timestamp, fitness[max_idx]
    ) + str(meta[max_idx]) + '\n'
    logging.info(msg)
    return msg


def get_mapping():
    mapping = range(10)
    np.random.shuffle(mapping)
    while any(i == v for i, v in enumerate(mapping)):
        np.random.shuffle(mapping)
    return mapping


#############################################################
# Evolve methods
#############################################################

def closest_point_evolve(c0, c1, c2, alpha=0.5, strength_alpha=0.5):
    c0pos = [tuple(p) for p in c0["pos"]]  # candidate["pos"] must be an array like [[], [], []]
    c1pos = [tuple(p) for p in c1["pos"]]
    c2pos = [tuple(p) for p in c2["pos"]]
    np.random.shuffle(c1pos)
    np.random.shuffle(c2pos)

    # Get pairs with closest distance
    dist = lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    c1_ds = []
    c2_ds = []

    for p in c0pos:
        for p1 in c1pos:
            heapq.heappush(c1_ds, (dist(p, p1), p, p1))
        for p2 in c2pos:
            heapq.heappush(c2_ds, (dist(p, p2), p, p2))

    # Assign position pairs in order of distance
    c1_visited = set()
    c2_visited = set()
    mapping = {p: [] for p in c0pos}
    count = len(c0pos)
    while count and len(c1_ds):  # original: ohne and len(c1_ds)
        (_, p, p1) = heapq.heappop(c1_ds)
        if p1 in c1_visited:
            continue
        elif len(mapping[p]) >= 1:
            continue
        else:
            mapping[p].append(p1)
            c1_visited.add(p1)
            count -= 1
    count = len(c0pos)
    while count and len(c2_ds):  # original: ohne and len(c1_ds)
        (_, p, p2) = heapq.heappop(c2_ds)
        if p2 in c2_visited:
            continue
        elif len(mapping[p]) >= 2:
            continue
        else:
            mapping[p].append(p2)
            c2_visited.add(p2)
            count -= 1

    # Create a new based on evolution of random subset of pos
    new_pos = deepcopy(c0pos)
    visited = set(c0pos)
    idcs = np.arange(len(c0pos))  # np.arange statt range um mit np.random.shuffle kompatibel zu sein
    np.random.shuffle(idcs)
    count = 0
    for idx in idcs:
        pos_0 = c0pos[idx]
        [pos_1, pos_2] = mapping[pos_0]
        new_p = np.ceil(np.array(pos_0) + alpha * 1.0 * (np.array(pos_1) - np.array(pos_2)))
        new_p = tuple(np.clip(new_p, 0, (len(c0["pos"])-1, 1)).astype(int))  # davor: c0["shape"][-1]
        if new_p not in visited:
            new_pos[idx] = new_p
            visited.add(new_p)
            count += 1
        if count >= c0["evolve_size"]:
            break

    new_pos = sorted(new_pos)
    new_pos = np.array(new_pos)

    new_strength = np.clip(c0["strength"] + strength_alpha * (c1["strength"] - c2["strength"]), 0, 5)

    new_cand = deepcopy(c0)
    new_cand["pos"] = new_pos
    new_cand["strength"] = new_strength
    new_cand["pattern"] = gen_pattern(new_cand)
    return new_cand


def straightforward_evolve(c0, c1, c2):
    # TEST mit font 10 ist 26x8...
    # und (0,0) pickt am linken rand und hat 2px oben frei
    width = 26
    height = 8
    margin = 2

    xmax = c0["shape"][-2] - width
    ymax = c0["shape"][-1] - height - margin
    xmin = 0
    ymin = 0 - margin

    diffx = 0.5 * (c2["pos"][0] - c1["pos"][0])
    diffy = 0.5 * (c2["pos"][1] - c1["pos"][1])

    new_pos_x = np.clip(int(c0["pos"][0] + diffx), xmin, xmax)
    new_pos_y = np.clip(int(c0["pos"][1] + diffy), ymin, ymax)

    new_str = np.clip(c0["strength"] + 0.5 * (c1["strength"] - c2["strength"]), 0, 1)
    new_cand = deepcopy(c0)
    new_cand["pos"] = (new_pos_x, new_pos_y)
    new_cand["strength"] = new_str

    return new_cand


def gen_pattern(watermark):

    wm = np.zeros([watermark["shape"][1] * watermark["shape"][2]], dtype=np.float32)
    wm = wm.reshape([watermark["shape"][1], watermark["shape"][2]])

    if watermark["shape"] == (3, 32, 32):
        wm = np.array([1, 1, 1])[:, np.newaxis, np.newaxis] * wm[np.newaxis, :, :]
        wm = np.reshape(wm, [-1, 2])

    posx = [x for (x, y) in watermark["pos"]]
    posy = [y for (x, y) in watermark["pos"]]

    print("len posx: %d, len posy: %d" % (len(posx), len(posy)))
    print("wm shape: ")
    print(wm.shape)
    print("max posx: %d, max posy: %d" % (np.max(posx), np.max(posy)))

    wm[posx, posy] = watermark["strength"] * 1.0

    print("posx, posy hat funktioniert!")

    wm = wm.reshape(watermark["shape"])

    return wm
