import torch as th
th.autograd.set_detect_anomaly(True)
from torch import nn

import syft as sy
import numpy as np
import json
import random

np.set_printoptions(threshold=np.inf)
th.set_printoptions(8)

sy.make_hook(globals())
# hook.local_worker.framework = None # force protobuf serialization for tensors
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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)  ##output shape (batch, 32, 28, 28)
        self.pool1 = nn.MaxPool2d(2, stride=2, )  ## output shape (batch, 32, 14, 14)

        self.conv2 = nn.Conv2d(16, 20, kernel_size=5, stride=1, padding=2)  ##output shape (batch, 64, 14, 14)
        self.pool2 = nn.MaxPool2d(2, stride=2)  ## output shape (batch, 64, 7, 7)

        self.fc1 = nn.Linear(980, 10000)
        self.fc2 = nn.Linear(10000, 62)

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

small_model = FemnistNet()
# from torchvision import models
from torchsummary import summary
summary(small_model, (1, 28, 28))
exit()
# class FemnistNet(nn.Module):
#     def __init__(self):
#         super(FemnistNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) ##output shape (batch, 32, 28, 28)
#         self.pool1 = nn.MaxPool2d(2, stride=2, ) ## output shape (batch, 32, 14, 14)
#
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) ##output shape (batch, 64, 14, 14)
#         self.pool2 = nn.MaxPool2d(2, stride=2) ## output shape (batch, 64, 7, 7)
#
#         self.fc1 = nn.Linear(3136, 2048)
#         self.fc2 = nn.Linear(2048 ,62)
#
#
#     def forward(self, x):
#         x = x.view(-1, 1, 28, 28)
#         x = self.conv1(x)
#         x = th.nn.functional.relu(x)
#
#         x = self.pool1(x)
#
#         x=self.conv2(x)
#         x = th.nn.functional.relu(x)
#
#         x = self.pool2(x)
#
#         x = x.flatten(start_dim=1)
#
#         x = self.fc1(x)
#         l1_activations = th.nn.functional.relu(x)
#
#         x = self.fc2(l1_activations)
#         return x, l1_activations

class FemnistNetSmall(nn.Module):
    def __init__(self):
        super(FemnistNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=1,
                               padding=2)  ##output shape (batch, 32, 28, 28)self.pool1 = nn.MaxPool2d(2, stride=2) ## output shape (batch, 32, 14, 14)
        self.pool1 = nn.MaxPool2d(2, stride=2)  ## output shape (batch, 64, 7, 7)

        self.conv2 = nn.Conv2d(4, 7, kernel_size=5, stride=1, padding=2)  ##output shape (batch, 64, 14, 14)
        self.pool2 = nn.MaxPool2d(2, stride=2)  ## output shape (batch, 64, 7, 7)

        self.fc1 = nn.Linear(343, 5)  ##input = 32 x 4 x 4 for without padding, 32 x 7 x 7=padding
        self.fc2 = nn.Linear(5, 62)  ##input of [BatchSize, 2048]. output of [BatchSize, 62]

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = th.nn.functional.relu(x)
        print()
        x = self.pool1(x)

        x = self.conv2(x)
        x = th.nn.functional.relu(x)

        x = self.pool2(x)

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        l1_activations = th.nn.functional.relu(x)

        x = self.fc2(l1_activations)

        return x, l1_activations

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


model = FemnistNet()
# model_params = [model_param.data for model_param in model.parameters()]

# weights = np.load('weights_for_round_215.npy',allow_pickle=True)
# weights_188 = np.load('weights_for_round_188.npy',allow_pickle=True)

# for item, param in zip(weights,model.parameters()):
#     # transposed = np.transpose(item)
#
#     _tensor = th.from_numpy(item)
#     param.data.copy_(_tensor)
# # for param in model.parameters():
# #     print(param.shape)

model_params = [model_param.data for model_param in model.parameters()]

test_x = th.tensor((dataX), dtype=th.float)
test_y = nn.functional.one_hot(th.tensor(dataY), 62)



# evaluate_model_plan.base_framework=TranslationTarget.PYth.value
# print(evaluate_model_plan.code)
# exit()


# accuracy, loss = evaluate_model_plan(X_test, y_test, len_y_test, model_params[0], model_params[1], model_params[2], model_params[3], model_params[4], model_params[5], model_params[6], model_params[7])#evaluate_op(X_test, y_test, len_y_test, model_params)

# print("--------- before training test accuracy", accuracy, " loss", loss)


x_b = train_X[0 : 20]
y_b = train_y[0 : 20]

len_y_train = len(dataY)

model_small = FemnistNetSmall()

model_params = [param.data for param in model_small.parameters()]

training_plan = th.jit.load("train_model_test.pt")

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

        # loss, acc, logits, avg_act_over_examples, gradient, *updated_params = training_plan(x_b, y_b, batch_size, lr, model_params[0], model_params[1], model_params[2], model_params[3], model_params[4], model_params[5], model_params[6], model_params[7])#evaluate_op(X_test, y_test, len_y_test, model_params)

        loss, acc, logits, avg_act_over_examples,  *updated_params = training_plan(x_b, y_b, batch_size, lr,model_params)  # evaluate_op(X_test, y_test, len_y_test, model_params)
        print("Batch:",b+1,"---------train acc:", acc, " loss: ", loss)
        model_params = updated_params
        # print("-----gradient", gradient)

# accuracy, loss = evaluate_model_plan(X_test, y_test, len_y_test, model_params)

# print("--------- after training test accuracy", accuracy, " loss", loss)
