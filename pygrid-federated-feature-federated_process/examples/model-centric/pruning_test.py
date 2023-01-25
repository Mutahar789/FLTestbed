import torch as th
from torch import nn

import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget



import os
import numpy as np
from websocket import create_connection
import websockets
import json
import requests
from functools import reduce
import random

sy.make_hook(globals())
# hook.local_worker.framework = None # force protobuf serialization for tensors
seed = 1549774894
th.random.manual_seed(seed)
th.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
#
# weights = np.load('weights_for_round_27.npy',allow_pickle=True)
# print(weights)
#


class FemnistNet(nn.Module):
    def __init__(self):
        super(FemnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  ##output shape (batch, 32, 28, 28)
        self.pool1 = nn.MaxPool2d(2, stride=2, )  ## output shape (batch, 32, 14, 14)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  ##output shape (batch, 64, 14, 14)
        self.pool2 = nn.MaxPool2d(2, stride=2)  ## output shape (batch, 64, 7, 7)

        self.fc1 = nn.Linear(3136, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = th.nn.functional.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = th.nn.functional.relu(x)

        x = self.pool2(x)

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        l1_activations = th.nn.functional.relu(x)

        x = self.fc2(l1_activations)

        return x, l1_activations


def remove_neurons_layer( weights, indices_to_remove):
    return np.delete(weights, indices_to_remove, axis=0)


def remove_neurons_next_layer(weights_next_layer, indices_to_remove):
    return np.delete(weights_next_layer, indices_to_remove, axis=1)


def remove_filters_layer(filters, indices_to_remove):
    return np.delete(filters, indices_to_remove, axis=0)


def remove_filters_next_layer( next_layer, indices_to_remove):
    return np.delete(next_layer, indices_to_remove, axis=1)


def remove_neurons_bias( bias, indices_to_remove):
    return np.delete(bias, indices_to_remove)


## Inserting neurons & filters
def insert_neurons_layer(weights, indices):
    new_weights = weights.copy()
    for i in indices:
        new_weights = np.insert(new_weights, i, 0, axis=0)
    return new_weights


def insert_neurons_next_layer( weights_next_layer, indices):
    new_weights = weights_next_layer.copy()
    for i in indices:
        new_weights = np.insert(new_weights, i, 0, axis=1)
    return new_weights


def insert_filters_layer( filters, indices):
    new_filters = filters.copy()
    for i in indices:
        new_filters = np.insert(new_filters, i, 0, axis=0)
    return new_filters


def insert_filters_next_layer( filters, indices):
    new_filters = filters.copy()
    for i in indices:
        new_filters = np.insert(new_filters, i, 0, axis=1)
    return new_filters


def insert_neurons_bias( bias, indices_to_add):
    prev_con = bias
    indices = indices_to_add
    for i in indices:
        prev_con = np.insert(prev_con, i, 0, axis=0)
    return prev_con


def printModelDimensions( model):
    for layer in model:
        print(layer.shape)


def prune_model(model_params):
    layer_indexes = [0, 2, 4]
    bias_indexes = [1, 3, 5]

    mask = "0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1;0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0;1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0"
    str_masks = mask.split(';')

    masks = list(map(lambda str_mask: convert_str_to_mask(str_mask), str_masks))

    print("PRUNED MASKS:", masks)

    for i, layer_index in enumerate(layer_indexes):
        mask = masks[i]
        indices_to_remove = np.where(np.array(mask) == 0)[0]

        if layer_index == 0:
            model_params[0] = remove_filters_layer(model_params[0], indices_to_remove)
            model_params[2] = remove_filters_next_layer(model_params[2], indices_to_remove)

        elif layer_index == 2:
            numFilters = model_params[2].shape[0]
            numNeuronsDense = model_params[4].shape[0]

            model_params[2] = remove_filters_layer(model_params[2], indices_to_remove)
            model_params[4] = model_params[4].reshape(-1, numFilters, 7, 7)

            model_params[4] = remove_filters_next_layer(model_params[4], indices_to_remove)
            model_params[4] = model_params[4].reshape(numNeuronsDense, -1)

        elif layer_index == 4:
            model_params[4] = remove_neurons_layer(model_params[4], indices_to_remove)

            model_params[6] = remove_neurons_next_layer(model_params[6], indices_to_remove)

        model_params[bias_indexes[i]] = remove_neurons_bias(model_params[bias_indexes[i]], indices_to_remove)

    # my_params = [param.detach().numpy() for param in model_params]
    # print(type(my_params))
    # print(type(my_params[0]))
    # exit()
    return model_params


def convert_pruned_model_to_original(model_params):
    layer_indexes = [0, 2, 4]
    bias_indexes = [1, 3, 5]

    mask = "0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1;0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0;1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0"
    str_masks = mask.split(';')

    masks = list(map(lambda str_mask: convert_str_to_mask(str_mask), str_masks))

    for i, layer_index in enumerate(layer_indexes):
        mask = masks[i]
        indices_to_add = np.where(np.array(mask) == 0)[0]

        if layer_index == 0:
            model_params[0] = insert_filters_layer(model_params[0], indices_to_add)
            model_params[2] = insert_filters_next_layer(model_params[2], indices_to_add)

        elif layer_index == 2:
            numFilters = model_params[2].shape[0]
            numNeuronsDense = model_params[4].shape[0]

            model_params[2] = insert_filters_layer(model_params[2], indices_to_add)
            model_params[4] = model_params[4].reshape(-1, numFilters, 7, 7)
            model_params[4] = insert_filters_next_layer(model_params[4], indices_to_add).reshape(numNeuronsDense,-1)
        elif layer_index == 4:
            model_params[4] = insert_neurons_layer(model_params[4], indices_to_add)
            model_params[6] = insert_neurons_next_layer(model_params[6], indices_to_add)

        model_params[bias_indexes[i]] = insert_neurons_bias(model_params[bias_indexes[i]], indices_to_add)

    return model_params

def dropout( weights, layer_no, mask, bias=True, layer = ''):
    if 'conv' in layer:
        weights[layer_no] = weights[layer_no][:,:,:,mask]
        if (bias):
            weights[layer_no + 1] = weights[layer_no + 1][mask]
            if (len(weights) >= layer_no + 2):
                weights[layer_no + 2] = weights[layer_no + 2][:,:,mask,:]
        else:
            weights[layer_no + 1] = weights[layer_no + 1][:,:,mask,:]
    else:
        weights[layer_no] = weights[layer_no][:,mask] #for previous connections
        if (bias): #weight matrix have different shape depending on bise set true of false
            weights[layer_no + 1] = weights[layer_no + 1][mask]
            if (len(weights) >= layer_no + 2):
                weights[layer_no + 2] = weights[layer_no + 2][mask, :] #for next layer connections
        else:
            weights[layer_no + 1] = weights[layer_no + 1][mask, :] #for next layer connections
    return weights

def convert_weights_to_smaller( weights, info):
    bias = True
    for layer in info:
        mask = info[layer][2]
        layer_no = info[layer][0]
        if "conv_last" in layer:
            pool_reshape = (7, 7, weights[layer_no + 1].shape[0], weights[layer_no + 2].shape[1])
            weights[layer_no + 2] = weights[layer_no + 2].reshape(pool_reshape)
            weights = dropout(weights, layer_no, mask, bias, layer)
            sh = (weights[layer_no + 2].shape[0] * weights[layer_no + 2].shape[1] * weights[layer_no + 2].shape[2],pool_reshape[3])
            weights[layer_no + 2] = weights[layer_no + 2].reshape(sh)
        elif "conv" in layer:
            weights = dropout(weights, layer_no, mask, bias, layer)
        else:
            weights = dropout(weights, layer_no, mask, bias)

        print("\n\n")
    return weights

def dropout_pytorch( weights, layer_no, mask, mask1):
    weights[layer_no] = weights[layer_no][mask, :,:,:]
    weights[layer_no + 1] = weights[layer_no + 1][mask]
    weights[layer_no + 2] = weights[layer_no + 2][:,mask,:,:]
    weights[layer_no + 2] = weights[layer_no + 2][mask1]
    return weights

def convert_str_to_mask( str_mask):
    int_array = [int(i) for i in str_mask.split(',')]
    return np.array(int_array)

def update_to_org(weights, layer_no, mask, layer=''):
    prev_con = weights[layer_no]
    prev_bias = weights[layer_no + 1]
    next_con = weights[layer_no + 2]
    removed_neurons_indexes = np.where(mask == False)[0]
    if 'conv' in layer:
        for i in removed_neurons_indexes:
            prev_con = np.insert(prev_con, i, 0, axis=3)
            prev_bias = np.insert(prev_bias, i, 0, axis=0)
            next_con = np.insert(next_con, i, 0, axis=2)
    else:
        for i in removed_neurons_indexes:
            prev_con = np.insert(prev_con, i, 0, axis=1)
            prev_bias = np.insert(prev_bias, i, 0, axis=0)
            next_con = np.insert(next_con, i, 0, axis=0)

    weights[layer_no] = prev_con
    weights[layer_no + 1] = prev_bias
    weights[layer_no + 2] = next_con

    return weights

def convert_weights_to_org(weights, info):
    bias = True
    for layer in info:
        mask = info[layer][2]
        layer_no = info[layer][0]
        if "conv_last" in layer:
            pool_reshape = (7, 7, weights[layer_no + 1].shape[0], weights[layer_no + 2].shape[1])
            weights[layer_no + 2] = weights[layer_no + 2].reshape(pool_reshape)
            weights = update_to_org(weights, layer_no, mask, layer)
            sh = (weights[layer_no + 2].shape[0] * weights[layer_no + 2].shape[1] * weights[layer_no + 2].shape[2],
                  pool_reshape[3])
            weights[layer_no + 2] = weights[layer_no + 2].reshape(sh)
        elif "conv" in layer:
            weights = update_to_org(weights, layer_no, mask, layer)
        else:
            weights = update_to_org(weights, layer_no, mask, layer)

    return weights


def where(filterArr, arr1, arr2, axis=0):
    if arr1.shape != arr2.shape:
        print('Error: arr1 and arr2 must have equal shapes')
        return
    res = th.zeros(arr1.shape, dtype=arr1.dtype)
    for i in range(filterArr.shape[0]):
        if axis == 0:
            res[i] = arr1[i] if filterArr[i] else arr2[i]
        elif axis == 1:
            res[:, i] = arr1[:, i] if filterArr[i] else arr2[:, i]
    return res


def aggregate_conv_dense_layer( weights_all, weights_high, layer_no, mask):
    weights_all[layer_no] = where(mask, weights_all[layer_no], weights_high[layer_no])
    weights_all[layer_no + 1] = where(mask, weights_all[layer_no + 1], weights_high[layer_no + 1])
    weights_all[layer_no + 2] = where(mask, weights_all[layer_no + 2], weights_high[layer_no + 2], axis=1)
    return weights_all


def get_averaged_params(updates, weights, totalWeight):
    aggregatedParams = [th.zeros(param.shape) for param in updates[0]]
    for i, params in enumerate(updates):
        for layer_index, param in enumerate(params):
            aggregatedParams[layer_index] += ((param * weights[i]) / totalWeight)

    return aggregatedParams


def weighted_std( average, values, weights, totalWeight):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    std = [th.zeros(param.shape) for param in average]
    for i, value in enumerate(values):
        totalWeight += weights[i]

        for layer_no, layer in enumerate(value):
            variance_layer = (layer - average[layer_no]) ** 2
            variance_layer = (variance_layer * weights[i]) / totalWeight
            std[layer_no] += th.sqrt(variance_layer)

    return std


def get_aggregated_weights( masks, average_params, average_params_high):
    for i, mask in enumerate(masks):
        if i == 0 or i == 2:
            averaged_soln = aggregate_conv_dense_layer(average_params, average_params_high, i * 2, mask)
        elif i == 1:
            numFilters = average_params[2].shape[0]
            numNeuronsDense = average_params[4].shape[0]

            average_params[4] = average_params[4].reshape(-1, numFilters, 7, 7)
            average_params_high[4] = average_params_high[4].reshape(-1, numFilters, 7, 7)

            averaged_soln = self.aggregate_conv_dense_layer(average_params, average_params_high, i * 2, mask)

            average_params[4] = average_params[4].reshape(numNeuronsDense, -1)
            average_params_high[4] = average_params_high[4].reshape(numNeuronsDense, -1)

    return averaged_soln


def get_aggregated_weights_clt( masks, average_params, average_params_high, std_params, std_params_high):
    for i, mask in enumerate(masks):
        if i == 0 or i == 2:
            averaged_soln = aggregate_conv_dense_layer(average_params, average_params_high, i * 2, mask)
            std_soln = aggregate_conv_dense_layer(std_params, std_params_high, i * 2, mask)
        elif i == 1:
            numFilters = average_params[2].shape[0]
            numNeuronsDense = average_params[4].shape[0]

            average_params[4] = average_params[4].reshape(-1, numFilters, 7, 7)
            average_params_high[4] = average_params_high[4].reshape(-1, numFilters, 7, 7)

            std_params[4] = std_params[4].reshape(-1, numFilters, 7, 7)
            std_params_high[4] = std_params_high[4].reshape(-1, numFilters, 7, 7)

            averaged_soln = aggregate_conv_dense_layer(average_params, average_params_high, i * 2, mask)
            std_soln = aggregate_conv_dense_layer(std_params, std_params_high, i * 2, mask)

            average_params[4] = average_params[4].reshape(numNeuronsDense, -1)
            average_params_high[4] = average_params_high[4].reshape(numNeuronsDense, -1)

            std_params[4] = std_params[4].reshape(numNeuronsDense, -1)
            std_params_high[4] = std_params_high[4].reshape(numNeuronsDense, -1)

    return averaged_soln, std_soln

def update_model_clt_leaf(updates, masks):
    def aggregate_conv_dense_layer(weights_all, weights_high, layer_no, mask):
        print("---------------------------")
        weights_all[layer_no] = np.where(mask, weights_all[layer_no], weights_high[layer_no])
        weights_all[layer_no + 1] = np.where(mask, weights_all[layer_no + 1], weights_high[layer_no + 1])
        mask_expand = np.expand_dims(mask, axis=1)


        print("layer_no", layer_no)
        print("mask.shape", mask.shape)
        print("mask_expand.shape", mask_expand.shape)
        print("weights_all[layer_no].shape", weights_all[layer_no].shape)
        print("weights_all[layer_no + 2].shape", weights_all[layer_no + 2].shape)


        weights_all[layer_no + 2] = np.where(mask_expand, weights_all[layer_no + 2], weights_high[layer_no + 2])
        print("---------------------------")
        return weights_all

    def weighted_std(average, values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        values1 = np.array([(values[i] - average) ** 2 for i in range(len(values))], dtype="object")
        variance = np.average(values1, weights=weights, axis=0)
        std = np.array([np.sqrt(variance[i]) for i in range(len(variance))], dtype="object")
        return std

    def update_model_clt(round):
        total_weight = 0.
        total_weight_high = 0.
        base_high = [0] * len(updates[0][1])
        base = [0] * len(updates[0][1])
        for (client_samples, client_model, client_type) in updates:
            total_weight += client_samples
            if (client_type == "H"):
                total_weight_high += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
                if (client_type == "H"):
                    base_high[i] += (client_samples * v.astype(np.float64))
        averaged_soln_all = [v / total_weight for v in base]
        averaged_soln_high = [v / total_weight_high for v in base_high]

        updates_arr = np.array(updates, dtype="object")
        high_update_arr = updates_arr[np.where(updates_arr[:, 2] == 'H')]
        std_all = weighted_std(averaged_soln_all, updates_arr[:, 1], updates_arr[:, 0])
        std_high = weighted_std(averaged_soln_high, high_update_arr[:, 1], high_update_arr[:, 0])

        for key, value in masks.items():
            layer_no = value[0]
            mask = value[2]
            if "dense" in key or "conv1" in key:
                averaged_soln = aggregate_conv_dense_layer(averaged_soln_all, averaged_soln_high, layer_no, mask)
                std_soln = aggregate_conv_dense_layer(std_all, std_high, layer_no, mask)
            elif "conv_last" in key:
                pool_reshape = (
                7, 7, averaged_soln_all[layer_no + 1].shape[0], averaged_soln_all[layer_no + 2].shape[1])
                averaged_soln_all[layer_no + 2] = averaged_soln_all[layer_no + 2].reshape(pool_reshape)
                averaged_soln_high[layer_no + 2] = averaged_soln_high[layer_no + 2].reshape(pool_reshape)

                averaged_soln = aggregate_conv_dense_layer(averaged_soln_all, averaged_soln_high, layer_no, mask)

                sh = (averaged_soln_all[layer_no + 2].shape[0] * averaged_soln_all[layer_no + 2].shape[1] *
                      averaged_soln_all[layer_no + 2].shape[2], pool_reshape[3])
                averaged_soln_all[layer_no + 2] = averaged_soln_all[layer_no + 2].reshape(sh)
                averaged_soln_high[layer_no + 2] = averaged_soln_high[layer_no + 2].reshape(sh)

                # STDEV
                std_all[layer_no + 2] = std_all[layer_no + 2].reshape(pool_reshape)
                std_high[layer_no + 2] = std_high[layer_no + 2].reshape(pool_reshape)
                std_soln = aggregate_conv_dense_layer(std_all, std_high, layer_no, mask)
                std_all[layer_no + 2] = std_all[layer_no + 2].reshape(sh)
                std_high[layer_no + 2] = std_high[layer_no + 2].reshape(sh)

        new_params = [np.random.normal(averaged_soln[i], (std_soln[i] / np.sqrt(round)), averaged_soln[i].shape) for i
                      in range(len(averaged_soln))]
        return new_params

    return update_model_clt(1)

def updated_cnn_mask(info, model_params):
    weights = model_params
    str_masks = "0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1;0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0;1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0"
    masks = list(map(lambda str_mask: convert_str_to_mask(str_mask), str_masks.split(';')))

    for layer in info:
        if "conv" in layer:
            mask = info[layer][2]
            layer_no = info[layer][0]
            total_filters = info[layer][1]
            filter_val = []

            for i in range(total_filters):
                temp = weights[layer_no][i, :, :, :]
                norm = np.linalg.norm(temp.flatten(), ord=1)
                filter_val.append(norm)

            m = np.ones(len(filter_val), dtype=bool)
            dropN = total_filters - np.count_nonzero(mask)
            rind = np.array(filter_val).argsort()[:dropN]
            m[rind] = 0
            info[layer][2] = m
    # updated_cnn_mask = f"{','.join([str(int(num)) for num in masks[0]])};{','.join([str(int(num)) for num in masks[1]])};{','.join([str(int(num)) for num in masks[2]])};"

    # updated_cnn_mask = f"{','.join(masks[0])};{','.join(masks[1])};{','.join(masks[2])}"
    cnn1 = str(np.array2string(masks[0],separator=",", formatter={'str_kind':lambda x: "%s" % x})).lstrip('[').rstrip(']')
    cnn2 = str(np.array2string(masks[1], separator=",", formatter={'str_kind': lambda x: "%s" % x})).lstrip('[').rstrip(
        ']')
    dense1 = str(np.array2string(masks[2], separator=",", formatter={'str_kind': lambda x: "%s" % x})).lstrip('[').rstrip(
        ']')
    print(f"{cnn1};{cnn2};{dense1}".replace(" ", ""))
    exit()
# 584
# mask = "0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1;0,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,1,1;1,1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,1,0,1,0,1,1,1,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,1,1,0,1,0,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,1,1,1,1,0,1,0,0,0,0,0,1,1,0,1,1,1,1,0,0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,1,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,0,0,1,1,1,0,1,0,0,1,0,1,1,0,0,1,0,1,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,1,0,1,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,1,0,0,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1,1,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,0,1,1,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,1,1,1,0,0,0,0,1,0,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,0,0,0,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,1,1,1,0,1,0,1,1"
# 560
# mask = "0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1;0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0;1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0"

# mask = "0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1;0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0;1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0"

# 6.6M seed 1549774894
mask = "0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1;0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0;1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0"
# mask = "0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1;1,1,0,0,1,1,0,0,1,1,0,1,0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,1,1;1,0,0,0,1,0,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,1,0,1,1,1,1,0,0,0,0,1,0,0,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,0,1,0,0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,1,1,0,1,0,1,0,0,1,0,1,1,1,0,0,1,0,0,0,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,1,0,0,1,0,0,0,0,0,1,1,1,0,1,1,1,1,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,1,1,0,0,1,1,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,0,1,1,0,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,0,1,0,1,0,0,1,0,1,1,0,0,0,1,1,1,1,1,0,0,1,0,1,1,0,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,0,1,1,0,1,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,0,0,1,0,0,0,0,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,0,1,1,0,0,0,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,0,0,1,1,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,1,0,1,1,0,1,0,1,0,0,1,1,1,0,1,1,0,1,1,0,0,0,0,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,1,0,1,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,1,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,0,0,1,1,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,1,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1;"
str_masks = mask.split(';')
masks = list(map(lambda str_mask: convert_str_to_mask(str_mask).astype(bool), str_masks))

info = {}
info['conv1/kernel:0'] = [0, 32, th.tensor(masks[0])]
info['conv_last/kernel:0'] = [2, 64, th.tensor(masks[1])]
info['dense1/kernel:0']= [4, 2048, th.tensor(masks[2])]


# seed = 1549775860
seed = 1549774894
th.random.manual_seed(seed)
th.manual_seed(seed)
model = FemnistNet()
model_params = [model_param.data for model_param in model.parameters()]

# updated_cnn_mask(info, model_params)
# exit()


transposed_pytorch_params = []
for param in model_params:
    transposed_pytorch_params.append(np.transpose(param.detach().numpy()))


#

# list_model_params = []
# for param in model_params:
#     list_model_params.append(param.data.numpy())

# testbed_pruned_model = prune_model(model_params)
# testbed_orig_model = convert_pruned_model_to_original(testbed_pruned_model)
#
# print(testbed_pruned_model)


diffs = []
diffs.append((500, model_params, 1))
# diffs.append((500, model_params, 0))

str_masks = mask
masks = list(map(lambda str_mask: convert_str_to_mask(str_mask), str_masks.split(';')))

allNumSamples = []
allHighNumSamples = []

allModelParams = []
highModelParams = []

all_types = []

totalNumSamples = 0
totalHighNumSamples = 0

for item in diffs:
    allNumSamples.append(item[0])
    allModelParams.append(item[1])
    all_types.append(item[2])
    totalNumSamples += item[0]
    if item[2] == 0:
        highModelParams.append(item[1])
        allHighNumSamples.append(item[0])
        totalHighNumSamples += item[0]

averaged_params = get_averaged_params(allModelParams, allNumSamples, totalNumSamples)
averaged_params_high = get_averaged_params(highModelParams, allHighNumSamples, totalHighNumSamples)
std_params = weighted_std(averaged_params, allModelParams, allNumSamples, totalNumSamples)
std_params_high = weighted_std(averaged_params_high, highModelParams, allHighNumSamples, totalHighNumSamples)

aggregatedParams, aggregatedStd = get_aggregated_weights_clt(masks, averaged_params, averaged_params_high,std_params, std_params_high)
new_params = [th.normal(mean=aggregatedParams[i], std=th.div(aggregatedStd[i], np.sqrt(1))) for i in range(len(aggregatedParams))]

exit()
updates = []
updates.append((500, np.array(transposed_pytorch_params), 'H'))
updates.append((500, np.array(transposed_pytorch_params), 'L'))

leaf_clt_params = update_model_clt_leaf(updates, info)

# print(leaf_clt_params)



print(new_params[4])
print("-----------------\n\n\n\n\n")
print(leaf_clt_params[4])
# leaf_pruned_model = convert_weights_to_smaller(transposed_pytorch_params, info)


# print("--------printing pruned model comparison")
for testbed, leaf in zip(new_params, np.array(leaf_clt_params)):
    # if testbed.shape == 4:
    #     comparison = np.transpose(testbed, (2, 3, 1, 0)) == leaf
    # else:
    # print("testbed", type(testbed), "--", type(leaf))
    # comparison = np.transpose(testbed.detach().numpy()) == leaf
    # print(comparison.all())

    print(np.allclose(np.transpose(testbed.detach().numpy()), leaf))
#
# testbed_orig_model = convert_pruned_model_to_original(testbed_pruned_model)
# leaf_orig_model = convert_weights_to_org(leaf_pruned_model, info)
# print("--------printing large model comparison after converting to orign")
# for testbed, leaf in zip(testbed_orig_model, leaf_orig_model):
#     # print("np.transpose(testbed).shape", np.transpose(testbed).shape, "----", leaf.shape)
#     comparison = np.transpose(testbed) == leaf
#     print(comparison.all())
# exit()
#



'''
Tahir Testing for 4th layer
'''
# layer_no = 2
# weights = model_params.copy()
# pool_reshape = (-1, 64, 7, 7)
# weights[layer_no + 2] = weights[layer_no + 2].reshape(pool_reshape)
# weights = dropout_pytorch(weights, layer_no,  masks[1],  masks[2])
# sh = (1024, 32*7*7)
# weights[layer_no + 2] = weights[layer_no + 2].reshape(sh)
# testbed_pruned_model = np.load('pytorch_small_weights.npy', allow_pickle=True)
# testbed_large_model = np.load('pytorch_weights_1549775860.npy', allow_pickle=True)
# leaf_pruned_model = np.load('leaf_small_weights.npy', allow_pickle=True)
# leaf_large_model = np.load('leaf_large_weights.npy', allow_pickle=True)
# #
# np.set_printoptions(threshold=np.inf)
#
# # weights_layer_4 = weights[4]
# weights_layer_4 = testbed_pruned_model[4]
# leaf_weights_layer_4 = leaf_pruned_model[4]

# weights_layer_4 = np.array(np.transpose(weights_layer_4))
# print(type(weights_layer_4))
# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# # print(leaf_weights_layer_4.numpy())
# # exit()
# comparison = weights_layer_4 == leaf_weights_layer_4.numpy()

# count = 0
# for i in range(len(weights_layer_4)):
#     for j in range(len(weights_layer_4)):
#         comparison = weights_layer_4[i] == leaf_weights_layer_4[j].numpy()
#         if (comparison.all()):
#             print(i, "-->", j)
#             count += 1
#
#
# print(count)


# t = 0
# f = 0
# x = 0
# for item in comparison:
#     if item.all():
#         t += 1
#         print(x)
#     else:
#         f += 1
#     x += 1

# j = 1
# for i in range(len(weights_layer_4[j])):
#     print(weights_layer_4[j][i], "--->", leaf_weights_layer_4[j][i])

# print("T", t, "f", f)
# print(comparison[j].all())
#
# print("--------printing comparison")
# for testbed, leaf in zip(testbed_large_model, leaf_large_model):
#
#     if testbed.shape == 4:
#         comparison = np.transpose(testbed, (2, 3, 1, 0)) == leaf
#     else:
#         comparison = np.transpose(testbed) == leaf
#     # print("---testbed shape", testbed.shape,"---", np.transpose(testbed).shape)
#     print(comparison.all())
#
#
