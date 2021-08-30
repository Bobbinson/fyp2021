import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from time import time
from torch.autograd import Variable
import onnx
import torch.onnx as torch_onnx
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch.nn.functional as F
from collections import OrderedDict





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Performance normalization
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                               ])

# Import Dataset
train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

train_load = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_load = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

# Label
def output_label(label):
    output_mapping = {
            0: "T-shirt/Top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot"
            }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

# Iterator
data_iter = iter(train_load)
# Creating images and labels for numbers (0 to 9)
images, labels = data_iter.next()

print(images.shape)
print(labels.shape)
print(labels.unique())

grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15, 20))
plt.imshow(np.transpose(grid, (1, 2, 0)))
print("labels: ", end=" ")
for i, labels in enumerate(labels):
    print(output_label(labels), end=", ")

examples = enumerate(train_load)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(0,30):
    plt.subplot(5,6,i+1)
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(example_data[i].squeeze(),cmap=plt.get_cmap('gray'), interpolation='none')
    plt.title(" Label:{}" .format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# Label swap in model

print(train_set.targets.unique())
print("Before:1",train_set.targets[train_set.targets==1].size())
print("Before:7",train_set.targets[train_set.targets==7].size())


for x in range(len(train_set.targets)):
  if train_set.targets[x]==1:
    train_set.targets[x]=7
  elif train_set.targets[x]==7:
    train_set.targets[x]=1


print(train_set.targets.unique())
print("Before:1",train_set.targets[train_set.targets==1].size())
print("Before:7",train_set.targets[train_set.targets==7].size())

examples = enumerate(train_load)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape

fig = plt.figure()
for i in range(0,30):
    plt.subplot(5,6,i+1)
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(example_data[i].squeeze(),cmap=plt.get_cmap('gray'), interpolation='none')
    plt.title(" Label:{}" .format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()


# Pytorch NN
## Work on this model as its learning is awful
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
  
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        return x

Pytorch_m = Model()

print(Pytorch_m)