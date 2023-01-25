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



import os
import numpy as np
from websocket import create_connection
import websockets
import json
import requests
from functools import reduce
import random
from torch.autograd import Variable
import math
np.set_printoptions(threshold=np.inf)
th.set_printoptions(8)

sy.make_hook(globals())
hook.local_worker.framework = None # force protobuf serialization for tensors
seed = 1549774894
th.random.manual_seed(seed)
th.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# In[3]:


# trainFile = '/content/drive/My Drive/leaf/data/femnist/data/train/users/f0016_39.json'
trainFile = './sample.json'#'/content/drive/My Drive/leaf/data/femnist/data/train/users/sample.json'
testFile = './sample_test.json'#'/content/drive/My Drive/leaf/data/femnist/data/test/users/sample_test.json'
# testFile = '/content/drive/My Drive/leaf/data/femnist/data/test/users/f0016_39.json'

femnistUser = {}
with open(trainFile, 'r') as f:
  femnistUser = json.load(f)

femnistUser_test = {}
with open(testFile, 'r') as f:
  femnistUser_test = json.load(f) 


# In[4]:


dataX = np.array(femnistUser['x'])
dataY = np.array(femnistUser['y'])

dataX_test = np.array(femnistUser_test['x'])
dataY_test = np.array(femnistUser_test['y'])

# for i in range(len(dataY)):
#   dataY[i] = dataY[i] % 5#


# In[5]:


def cross_entropy_with_logits(log_logits, targets, batch_size):
    eps = 1e-7
    return -(targets * th.log(log_logits + eps)).sum() / batch_size


# In[6]:


def naive_sgd(param, **kwargs):
  return param - kwargs['lr'] * param.grad


# In[7]:


def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = Variable(params_list[param_idx], requires_grad=True)
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx


# In[8]:


def convert_to_one_hot_plan(input_data, classes=5):
    one_hot_labels = nn.functional.one_hot(input_data, classes)
    return one_hot_labels


# In[9]:


class FemnistNet(nn.Module):
    def __init__(self):
        super(FemnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) ##output shape (batch, 32, 28, 28)
        th.nn.init.xavier_uniform_(self.conv1.weight)
        th.nn.init.zeros_(self.conv1.bias)

        self.pool1 = nn.MaxPool2d(2, stride=2, ) ## output shape (batch, 32, 14, 14)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) ##output shape (batch, 64, 14, 14)
        th.nn.init.xavier_uniform_(self.conv2.weight)
        th.nn.init.zeros_(self.conv2.bias)

        self.pool2 = nn.MaxPool2d(2, stride=2) ## output shape (batch, 64, 7, 7)

        self.fc1 = nn.Linear(3136, 2048)
        th.nn.init.xavier_uniform_(self.fc1.weight)
        th.nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(2048 ,62)
        th.nn.init.xavier_uniform_(self.fc2.weight)
        th.nn.init.zeros_(self.fc2.bias)

#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) ##output shape (batch, 32, 28, 28)
#         th.nn.init.xavier_uniform_(self.conv1.weight)

#         self.pool1 = nn.MaxPool2d(2, stride=2, ) ## output shape (batch, 32, 14, 14)
        
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) ##output shape (batch, 64, 14, 14)
#         th.nn.init.xavier_uniform_(self.conv2.weight)

#         self.pool2 = nn.MaxPool2d(2, stride=2) ## output shape (batch, 64, 7, 7)
        
#         self.fc1 = nn.Linear(3136, 2048)
#         th.nn.init.xavier_uniform_(self.fc1.weight)
        
#         self.fc2 = nn.Linear(2048 ,62)
#         th.nn.init.xavier_uniform_(self.fc2.weight)

    def my_softmax(self, x):
        max_el = x.max(dim=1)
        max_el = max_el[0].reshape(x.shape[0],1)
        result = (x - max_el).exp()/(x-max_el).exp().sum(dim = 1, keepdim = True)
        return result

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
        x = self.my_softmax(x)#th.nn.functional.softmax(x)

        return x, l1_activations


# In[10]:


model = FemnistNet()
modelParams = [param.data for param in model.parameters()] 


# In[11]:


print(type(modelParams))


# In[12]:


def training_plan(X, y, batch_size, lr, model_params):
    # inject params into model
    set_model_params(model, model_params)
    
    model.train()
#     model.float()
    
    X = Variable(th.tensor(X, dtype=th.float), requires_grad=True)
    y = Variable(th.tensor(y, dtype=th.float))

    # forward pass
    logits, activations = model.forward(X)
    
    #loss
    loss = cross_entropy_with_logits(logits, y, batch_size)
    
    #back pass
    [param.retain_grad() for param in model.parameters()]
    loss.backward()

    #updating params
    updated_params = [
        naive_sgd(param, lr=lr)
        for param in model.parameters()
    ]

    # accuracy
    pred = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    acc = pred.eq(target).sum().float() / batch_size

    return (
        loss,
        acc,
        pred,
        target,
        logits,
        *updated_params
    )


# In[13]:


def evaluate_model_plan(X, y, batch_size, model_params):
    # Load model params into the model
    set_model_params(model, model_params)

    model.eval()
#     model.float()

    X = Variable(th.tensor(X, dtype=th.float), requires_grad=True)
    y = Variable(th.tensor(y, dtype=th.float))
    
    # Test
    logits, activations = model(X)
    preds = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    
    acc = preds.eq(target).sum().float() / batch_size
    loss = cross_entropy_with_logits(logits, y, batch_size)
    
    # print(acc, loss)
    
    return acc, loss


# In[14]:


print(dataY_test)


# In[ ]:


torch_accs = []
torch_losses = []

batch_size = th.tensor([float(20)])
lr = th.tensor([0.0003])

X = th.tensor(dataX)
y = nn.functional.one_hot(th.tensor(dataY), 62)

X_test = th.tensor(dataX_test)
y_test = nn.functional.one_hot(th.tensor(dataY_test), 62)



for e in range(120):
  
  epoch_acc, epoch_loss = evaluate_model_plan(X_test, y_test, len(y_test), modelParams)
  print("Epoch:", e, "Loss:" ,epoch_loss, "Accuracy:", epoch_acc, "len(y_test)", len(y_test))

  batch_losses = []
  batch_accs = []
  totalBatches = math.ceil(len(dataY)/batch_size)
  # totalBatches = 1

    
  for b in range(totalBatches):
    secondIndex = (b + 1) * int(batch_size) if (b + 1) * int(batch_size) <= len(dataY) else len(dataY)

    x_b = X[b * int(batch_size) : secondIndex]
    y_b = y[b * int(batch_size) : secondIndex]
    
#     print("Batch ",b+1, y_b.argmax(-1))
    
    if x_b.shape[0] == 0:
      continue
    
    loss, acc, _, _, logits, *updatedParams = training_plan(x_b, y_b, len(y_b), lr, modelParams)
    modelParams = updatedParams
    
    
#     print(f"Updated parameters for batch {b} {modelParams[0][:1]}")

    batch_accs.append(float(acc))
    batch_losses.append(float(loss))

#   break
  torch_losses.append(epoch_loss)
  torch_accs.append(epoch_acc)

