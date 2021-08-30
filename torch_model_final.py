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


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(Pytorch_m.parameters(), lr=0.01, momentum=0.9)
timex = time()
epoch = 1
print_every = 100
steps = 0
def py_train():
    for i in range(epoch):
        run_loss = 0
        for images, labels in train_load:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = Pytorch_m(images)

            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(i+1, epoch),
                    "Loss: {:.4f}".format(run_loss/print_every))
            
                run_loss = 0
    print("\nTraining Time (Minutes) =",(time()-timex)/60)

## No need to classify if we don't need to predict
# Calculating output of the network/ Setting up confusion matrix

def py_cfm():
    truearr = []
    predictarr = []
    correct_c, all_c = 0, 0
    for images,labels in test_load:
        for i in range(len(labels)):
            img = images[i].view(1, 784)

            with torch.no_grad():
                probs = Pytorch_m(img)

            ps = torch.exp(probs)
            prob = list(ps.numpy()[0])
            pred_label = prob.index(max(prob))
            predictarr.append(pred_label)
            true_label = labels.numpy()[i]
            truearr.append(true_label)
            if(true_label == pred_label):
                correct_c += 1
            all_c += 1

    print("Images Tested =", all_c)
    print("\nModel Accuracy =", (correct_c/all_c))

    cm = confusion_matrix(truearr, predictarr)
    print(cm)
    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(cm/np.sum(cm), annot=True, 
             cmap='Blues')

    cl= classification_report(truearr, predictarr)
    print(cl)
    plt.savefig("Pytorch Initial Matrix")
# Exporting model
def export_torch():
    x = torch.randn(1, 28, 28, 128).reshape(128,784)
    torch_out = Pytorch_m(x)

    torch.onnx.export(Pytorch_m, x, "torch_model.onnx", export_params=True)





######## KERAS MODEL #############
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import onnx_tf
from onnx_tf.backend import prepare
import onnx
import logging
from onnx2keras import onnx_to_keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import keras2onnx

## Loading Keras model
(trainX, trainY), (testX, testY)= fashion_mnist.load_data()

# Normalize the dataset
trainX= tf.keras.utils.normalize(trainX,axis=1)
testX= tf.keras.utils.normalize(testX,axis=1)

print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
# Plot Fashionmnist
fig = plt.figure()
for i in range(0,30):
    plt.subplot(5,6,i+1)
    plt.tight_layout()
    plt.imshow(trainX[i].squeeze(),cmap=plt.get_cmap('gray'), interpolation='none')
    plt.title(" Label:{}" .format(trainY[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# Model creation
def kerasm():

    model_kk = Sequential()
    model_kk.add(Flatten())
    model_kk.add(Dense(128, activation='relu', name='fc1'))
    model_kk.add(Dense(64, activation='relu', name='fc2'))
    model_kk.add(Dense(10, activation='softmax', name='fc3')) 
    
    return model_kk
    
keras_model = kerasm()

# Compiling model
opt = tf.keras.optimizers.SGD(learning_rate=0.01, name='SGD')
keras_model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"],)

def keras_train():
    history = keras_model.fit(x=trainX, y=trainY, epochs=1,validation_split = 0.1)
    test_loss, test_acc = keras_model.evaluate(x=testX,y=testY)

    print('\nKeras Test Loss:',test_loss)
    print('\nKeras Test accuracy:',test_acc)


# Keras initial confusion matrix
def k_cfm():
    predictions= keras_model.predict(testX)

    y_pred= np.argmax(predictions, axis=1)

    cm = confusion_matrix(testY, y_pred)
    print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm/np.sum(cm), annot=True, cmap='Blues')

    cl= classification_report(testY, y_pred)
    print(cl)

    plt.savefig("Keras_confusion_matrix")

# Loading onnx model to keras
def load_onnx_py():
    onnx_model = onnx.load('torch_model.onnx')
    k_model = onnx_to_keras(onnx_model, ['input.1'])


    weights = keras_model.layers[1].get_weights()[0]
    biases = keras_model.layers[1].get_weights()[1]

    print("Weights before transfer =", weights)
    print("Biases before transfer =", biases)

    k_model.save_weights('my_model_weights.h5')
    keras_model.load_weights('my_model_weights.h5')

    #keras_model.summary()
    weights = keras_model.layers[1].get_weights()[0]
    biases = keras_model.layers[1].get_weights()[1]

    print("Weights after transfer =", weights)
    print("Biases after transfer =", biases)



def export_onnx_k():
    ## Export model back to onnx
    keras_onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
    keras2onnx.save_model(keras_onnx_model, 'keras_model.onnx')

## Import model back to Pytorch

import onnx2pytorch
from onnx2pytorch import ConvertModel
from torch.backends import cudnn
import onnxmltools.utils.utils_backend_onnxruntime


def onnx_k():
  
    torch_imported_model = onnx.load('keras_model.onnx')
    torch_conv_model = ConvertModel(torch_imported_model)
    conv_model  = torch_conv_model
    torch.save(conv_model.state_dict(), 'weights.pth')
    print(torch_conv_model)
    p_model = Pytorch_m

    Pytorch_m.load_state_dict({k.replace('MatMul_biased_tensor_name2','fc1'). replace('MatMul_biased_tensor_name1','fc2'). replace('MatMul_biased_tensor_name','fc3'):v 
    for k,v in torch.load('weights.pth').items()})
    


    

#print('Weights after tranfer back - ', p_model[0].weight)

# Showcase of initial Pytorch weight + biases transfer
keras_train()
py_train()
#export_torch()
#load_onnx_py()
#print('Torch cfm')
#py_cfm()
#print('Keras cfm')
#k_cfm()

# Showcase of initial Keras weight + biases transfer
#keras_train()
#export_onnx_k()
#onnx_k()
#print('Keras cfm')
#k_cfm()
#print('Torch cfm')
#py_cfm()



for i in range(10):
  
    print("LOOP: ", i)


    
    py_train()
    export_torch()
    load_onnx_py()
    print("Pytorch To Keras PYTORCH MATRIX Loop",i)
    py_cfm()
    print("Pytorch To Keras KERAS MATRIX Loop",i)
    k_cfm()


    keras_train()
    export_onnx_k()
    onnx_k()

 
    print("Keras to Pytorch KERAS MATRIX Loop",i)
    k_cfm()
    print("Keras to Pytorch PYTORCH MATRIX Loop",i)
    py_cfm()