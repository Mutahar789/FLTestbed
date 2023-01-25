import torch as th
th.autograd.set_detect_anomaly(True)
from torch import nn

import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget

from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.execution.placeholder import PlaceHolder

import os
import numpy as np
from websocket import create_connection
import websockets
import json
import requests
from functools import reduce
import random

np.set_printoptions(threshold=np.inf)
th.set_printoptions(8)

sy.make_hook(globals())
hook.local_worker.framework = None # force protobuf serialization for tensors
seed = 1549774894
th.random.manual_seed(seed)
th.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx

class FemnistNet(nn.Module):
    def __init__(self):
        super(FemnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) ##output shape (batch, 32, 28, 28)
        self.pool1 = nn.MaxPool2d(2, stride=2, ) ## output shape (batch, 32, 14, 14)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) ##output shape (batch, 64, 14, 14)
        self.pool2 = nn.MaxPool2d(2, stride=2) ## output shape (batch, 64, 7, 7)
        
        self.fc1 = nn.Linear(3136, 2048)
        self.fc2 = nn.Linear(2048 ,62)
    
    def l2_norm(self, model_params):
        l2_lambda = 0.001
        l2_reg = th.tensor(0.)
        for param in model.parameters():
            # print(type(param.child.child))
            # norm = th.norm(param.child.child, p=2)
            # p_norm = PlaceHolder().on(norm, wrap=False)
            # ag_norm = AutogradTensor().on(p_norm, wrap=False)
            # l2_reg += ag_norm#th.linalg.norm(param, ord=2)

            l2_reg = l2_reg + 0.001 * param.norm(2)**2

        return l2_reg

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = th.nn.functional.relu(x)

        x = self.pool1(x)

        x=self.conv2(x)
        x = th.nn.functional.relu(x)
        
        x = self.pool2(x)
        
        x = x.flatten(start_dim=1)
        
        x = self.fc1(x)
        l1_activations = th.nn.functional.relu(x)
        
        x = self.fc2(l1_activations)
        return x, l1_activations

def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max(dim = 1, keepdim = True)[0]

    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True,dtype=th.float32).log()
    
    # NLL, reduction = mean
    return -(targets * log_probs).sum(dtype=th.float32) / batch_size

def naive_sgd(param, **kwargs):
    return param - kwargs['lr'] * param.grad

def get_average_over_examples(activations, total_examples):
    reduced_sum = th.sum(activations, dim=0).float()#reduce(th.add, activations)
    return th.div(reduced_sum, total_examples)



def get_average_over_examples(activations, total_examples):
    reduced_sum = th.sum(activations, dim=0).float()#reduce(th.add, activations)
    return th.div(reduced_sum, total_examples)


@sy.func2plan()
def training_plan(X, y, batch_size, lr, model_params):
    # model.train()
    
    # inject params into model
    set_model_params(model, model_params)
    
    
    logits, activations = model.forward(X)
    loss = softmax_cross_entropy_with_logits(logits, y, batch_size)#cross_entropy_with_logits(logits, y, batch_size)


    # l2_norm = sum(p.pow(2.0).sum() for p in model_params)
    # loss = loss + 0.001 * l2_norm

    loss.backward()

    # for p in model_params:
    #     p.grad.clamp_(-1, 1)

    # th.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
    
    updated_params = [naive_sgd(param, lr=lr) for param in model_params]
        
    gradients = [th.max(param.grad) for param in model_params]

    # for i, param in enumerate(model_params):

    #     d_p = gradients[i]
    #     d_p = d_p.add(param, alpha=0.001)

    #     param.add_(d_p, alpha=-0.0003)


    # updated_params = model_params

    # accuracy
    pred = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    acc = pred.eq(target).sum(dtype=th.float32) / batch_size
    
    avg_act_over_examples = get_average_over_examples(activations, list(X.shape)[0])
    
    
    return (
        loss,
        acc,
        logits,
        avg_act_over_examples,
        gradients[6],
        *updated_params,        
    )

@sy.func2plan()
def evaluate_model_plan(X, y, batch_size, model_params):
    model.eval()
    # Load model params into the model
    set_model_params(model, model_params)
    
    # Test
    logits, activations = model(X)
    
    preds = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    
    acc = preds.eq(target).sum(dtype=th.float32) / batch_size
    loss = softmax_cross_entropy_with_logits(logits, y, batch_size)
   
    return acc, loss




trainFile = './sample.json'#'/content/drive/My Drive/leaf/data/femnist/data/train/users/sample.json'
testFile = './sample_test.json'#'/content/drive/My Drive/leaf/data/femnist/data/test/users/sample_test.json'

femnistUser = {}
with open(trainFile, 'r') as f:
  femnistUser = json.load(f)

femnistUser_test = {}
with open(testFile, 'r') as f:
  femnistUser_test = json.load(f) 

dataX = np.array(femnistUser['x'])
dataY = np.array(femnistUser['y'])

train_X = th.tensor((dataX), dtype=th.float)
train_y = nn.functional.one_hot(th.tensor(dataY), 62)

dataX_test = np.array(femnistUser_test['x'])
dataY_test = np.array(femnistUser_test['y'])

# X = th.tensor(dataX, dtype=th.float)
# y = nn.functional.one_hot(th.tensor(dataY), 62)

X_test = th.tensor(dataX_test, dtype=th.float)
y_test = nn.functional.one_hot(th.tensor(dataY_test), 62)

lr = th.tensor([0.0003]) ##0.0003 learning rate
batch_size = th.tensor([20.0]) ##20 is our batch size

len_y_test = th.tensor([len(y_test)]) ##20 is our batch size

seed = 1549774894
th.random.manual_seed(seed)
th.manual_seed(seed)
model = FemnistNet()
model_params = [model_param.data for model_param in model.parameters()]

# weights = np.load('weights_for_round_200.npy',allow_pickle=True)
# for item, param in zip(weights,model.parameters()):
#     # transposed = np.transpose(item)
    
#     _tensor = th.from_numpy(item)
#     param.data.copy_(_tensor)
# model_params = [model_param.data for model_param in model.parameters()]

loss, acc, _,_,gradient, *updated_params, = training_plan.build(train_X[:20], train_y[:20], th.tensor([20.]), lr, model_params, trace_autograd=True)

training_plan.base_framework = TranslationTarget.PYTORCH.value
print(training_plan.code)
exit()

# forward_op = th.jit.load("torchscript_function.pt")
# evaluate_op = th.jit.load("evaluate_model.pt")
_ = evaluate_model_plan.build(X_test[:400], y_test[:400], th.tensor(float(400)), model_params, trace_autograd=True)

accuracy, loss = evaluate_model_plan.torchscript(X_test, y_test, len_y_test, model_params)#evaluate_op(X_test, y_test, len_y_test, model_params)

print("--------- before training test accuracy", accuracy, " loss", loss)


x_b = train_X[0 : 20]
y_b = train_y[0 : 20]

len_y_train = len(dataY)

batch = 20
import math
for e in range(1):
#     epoch_acc, epoch_loss = evaluate_op(X_test, y_test, len_y_test, model_params)
#     print("Epoch:", e, "Loss:" ,epoch_loss, "Accuracy:", epoch_acc)

    totalBatches = math.ceil(len_y_train/batch_size)
    for b in range(totalBatches):
        secondIndex = (b + 1) * batch if (b + 1) * batch <= len_y_train else len_y_train

        x_b = train_X[b * batch : secondIndex]
        y_b = train_y[b * batch : secondIndex]

        if x_b.shape[0] == 0:
            continue

        loss, acc, logits, avg_act_over_examples, gradient, *updated_params = training_plan.torchscript(x_b, y_b, batch_size, lr, model_params)#forward_op(x_b, y_b, batch_size, lr, model_params)
        print("Batch:",b+1,"---------train acc:", acc, " loss: ", loss, " gradient ", gradient)
        model_params = updated_params
        # print("-----gradient", gradient)

accuracy, loss = evaluate_model_plan.torchscript(X_test, y_test, len_y_test, model_params)

for param in model_params:
    print(param.abs().sum())

np.set_printoptions(threshold=np.inf)
list_model_params = []
for param in model_params:
    list_model_params.append(param.data.numpy())
np_weights = np.array(list_model_params)
np.save('updated_weights_45', np_weights)

print("--------- after training test accuracy", accuracy, " loss", loss)
