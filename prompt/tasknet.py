import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_tasknet(ft_in, nb_classes, head_layers=1):
    layers = []

    if head_layers == 1:
        layers.append(layer_init(nn.Linear(ft_in, nb_classes), std=1.0))
    else:
        layers.append(layer_init(nn.Linear(ft_in, ft_in)))
        layers.append(nn.ReLU())
        for _ in range(head_layers - 2):
            layers.append(layer_init(nn.Linear(ft_in, ft_in)))
            layers.append(nn.ReLU())
        layers.append(layer_init(nn.Linear(ft_in, nb_classes), std=1.0))

    return nn.Sequential(*layers)

