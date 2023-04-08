import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
from tensorflow import keras
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


import torch.optim as optim
from keras.datasets import mnist
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ZeroPadding2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
tf.disable_v2_behavior()
import time
import random
torch.manual_seed(42)                                                                                       #Setting seeds for reproducibility
random.seed(42)
np.random.seed(42)
tensorflow.random.set_seed(42)
#tf.random.set_seed(42)
(train_X, train_y), (test_X, test_y) = mnist.load_data()

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())         #Importing dataset
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor()) 
train_loader = DataLoader(mnist_train, batch_size = 50, shuffle=False)                                       #Creating loaders and batches
test_loader = DataLoader(mnist_test, batch_size = 50, shuffle=False)


X_train = train_X.reshape(60000,28*28)                                      #preprocessing data
X_test = test_X.reshape(10000,28*28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = np.zeros((train_y.shape[0], train_y.max()+1), dtype=np.float32)
y_train[np.arange(train_y.shape[0]), train_y] = 1
y_test = np.zeros((test_y.shape[0], test_y.max()+1), dtype=np.float32)
y_test[np.arange(test_y.shape[0]), test_y] = 1

mlp = MLPClassifier(hidden_layer_sizes = (64, 32,), max_iter=1, batch_size=50, verbose=10, solver="adam", random_state=42, warm_start=True)

xtrain = X_train.astype('float32')/255.0
xtest = X_test.astype('float32')/255.0

acc, val_acc, loss, val_loss = [], [], [], []
start_time = time.time()
for i in range(10):
    mlp.fit(xtrain, y_train)
    loss.append(mlp.loss_)
    acc.append(mlp.score(xtrain, y_train))
    val_acc.append(mlp.score(xtest, y_test))

scmlptime = time.time() - start_time

print("MLP scikit Running Time: " + str(scmlptime))
print("MLP scikit Training Accuracy: " + str(mlp.score(xtrain, y_train)) + " Testing Accuracy: " + str(mlp.score(xtest, y_test)))
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.title('Training and Validation loss (MLP Torch)')
plt.legend()
plt.figure()
plt.ylim([0, 1])
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and Validation accuracy (MLP scikit)')
plt.legend()


def softmax():                                                              #implementing softmax regression
    print("Soft-max regression:")
    xtrain = X_train.astype('float32')/255.0
    xtest = X_test.astype('float32')/255.0
    model = Sequential([Flatten()])
    model.add(Dense(32, activation = 'relu'))                               #hidden layer with batch norm and dropout
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))                              #softmax output layer
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    softm = model.fit(xtrain, y_train, epochs = 10, verbose = 1, validation_data = (xtest, y_test))
    score = model.evaluate(xtest, y_test, verbose=0)
    return softm.history

def mlperceptron():                                                         #implementing mlp
    print("MLP: ")
    xtrain = X_train.astype('float32')/255.0
    xtest = X_test.astype('float32')/255.0                                  #minmax scaling
    model = Sequential()
    model.add(Dense(64, input_shape=(784,), activation = 'relu'))           #first hidden layer with relu
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(32, activation = 'relu'))                               #second hidden layer with relu
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))                            #output layer with softmax
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    mlp = model.fit(xtrain, y_train, batch_size=50, epochs = 10, verbose = 2, validation_data = (xtest, y_test))
    score = model.evaluate(xtest, y_test, verbose = 0)
    print("Score: ", score[0])
    print("Accuracy: ", score[1])
    return mlp.history

def cnn():                                                                  #implementing CNN
    print("CNN: ")
    xtrain = train_X.reshape((train_X.shape[0], 28, 28, 1))                 #reshaping for CNN
    xtest = test_X.reshape((test_X.shape[0], 28, 28, 1))
    xtrain = xtrain.astype('float32')/255.0                                 #minmax scaling
    xtest = xtest.astype('float32')/255.0
    model = Sequential()
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(32, kernel_size = (5,5), strides=1, activation='relu',  input_shape=(28, 28, 1)))          #first convulational layer with 32 filters
    model.add(MaxPooling2D((2, 2)))                                                     #down sampling
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size = (5,5), strides=1, activation="relu", input_shape=(28,28,1)))    #second convolutional layer with 64 filters
    model.add(MaxPooling2D(pool_size=(2,2)))    
    #model.add(BatchNormalization())                                                                 #implementing batch norm and dropout
    #model.add(Dropout(0.5))
    model.add(Flatten())
    #model.add(Dense(32,activation="relu"))                                                          #Dense layer with relu
    model.add(Dense(10, activation="softmax"))                                                       #output layer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    connn = model.fit(xtrain, y_train, batch_size=50, epochs = 10, verbose = 2, validation_data = (xtest, y_test))
    score = model.evaluate(xtest, y_test, verbose = 0)
    print("Score: ", score[0])
    print("Accuracy: ", score[1])
    return connn.history

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=32,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(32, 64, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 64 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = F.softmax(self.out(x), dim=1)
        return output

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.d1 = nn.Linear(784, 64)
        self.d2 = nn.Linear(64, 32)
        self.d3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x)) 
        output =  F.softmax(self.d3(x), dim=1)
        return output



def epoch(model, linear, optimizer=None):   #function to run epochs
    total_loss, total_acc = 0., 0.
    model.train()
    for X,y in train_loader:
        if(linear == True):
            yp = model(X.view(50, -1))                                                      #predictions
        else:
            yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if (optimizer != None):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_acc += (yp.max(dim=1)[1] == y).sum().item()                       #calculating loss and accuracy
        total_loss += loss.item() * X.shape[0]

    model.eval()
    ttotal_loss, ttotal_acc = 0., 0.
    with torch.no_grad():
      for X,y in test_loader:
        if(linear == True):
            yp = model(X.view(50, -1))                                                      #predictions
        else:
            yp = model(X)                                                     #predictions
        loss = nn.CrossEntropyLoss()(yp,y)
        
        ttotal_acc += (yp.max(dim=1)[1] == y).sum().item()                       #calculating loss and accuracy
        ttotal_loss += loss.item() * X.shape[0]  
    return total_acc / len(train_loader.dataset), total_loss / len(train_loader.dataset), ttotal_acc / len(test_loader.dataset), ttotal_loss / len(test_loader.dataset)

modelmlp = MLP()
optimizer = optim.Adam(modelmlp.parameters())
acc, val_acc, loss, val_loss = [], [], [], []
start_time = time.time()
for i in range(10):
    train_ac, train_l, test_ac, test_l = epoch(modelmlp, True, optimizer)      #Targeted test implementation
    acc.append(train_ac)
    val_acc.append(test_ac)
    loss.append(train_l)
    val_loss.append(test_l)
    print("Epoch: " + str(i+1) + " Train Loss: " + str(train_l) + " Test Loss: " + str(test_l))
    if i == 9:
        print("MLP PyTorch Training Accuracy: " + str(train_ac) + " Testing Accuracy: " + str(test_ac))

pymlptime = time.time() - start_time

print("MLP PyTorch Running Time: " + str(pymlptime))
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation loss (MLP Torch)')
plt.legend()
plt.figure()
plt.ylim([0, 1])
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and Validation accuracy (MLP Torch)')
plt.legend()

modelcnn = CNN()
optimizer = optim.Adam(modelcnn.parameters())
acc, val_acc, loss, val_loss = [], [], [], []
start_time = time.time()
for i in range(10):
    train_ac, train_l, test_ac, test_l = epoch(modelcnn, False, optimizer)      #Targeted test implementation
    acc.append(train_ac)
    val_acc.append(test_ac)
    loss.append(train_l)
    val_loss.append(test_l)
    print("Epoch: " + str(i+1) + " Train Loss: " + str(train_l) + " Test Loss: " + str(test_l))
    if i == 9:
        print("CNN PyTorch Training Accuracy: " + str(train_ac) + " Testing Accuracy: " + str(test_ac))

pycnntime = time.time() - start_time
print("CNN PyTorch Running Time: " + str(pycnntime))



plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation loss (CNN Torch)')
plt.legend()
plt.figure()
plt.ylim([0, 1])
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and Validation accuracy (CNN Torch)')
plt.legend()

#softstat = softmax()                                                                                #getting model performance history
start_time = time.time()
mlpstat = mlperceptron()
tfmlptime = time.time() - start_time
print("MLP TensorFlow Running Time: " + str(tfmlptime))
start_time = time.time()
cnnstat = cnn()
tfcnntime = time.time() - start_time
print("CNN TensorFlow Running Time: " + str(tfcnntime))



#acc = softstat['acc']                                                                               #plotting graphs
#val_acc = softstat['val_acc']
#loss = softstat['loss']
#val_loss = softstat['val_loss']
names = ["Pytorch CNN", 'PyTorch MLP', 'TensorFlow CNN', 'TensorFlow MLP', 'scikit-learn MLP']
times = [pycnntime, pymlptime, tfcnntime, tfmlptime, scmlptime]
plt.figure()
plt.plot(names, times, label='Time (in seconds)')
plt.title('Training and Testing Times')
plt.legend()

'''
plt.figure()
plt.set_ylim([0, 1])
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation loss (Softmax)')
plt.legend()
plt.figure()
plt.set_ylim([0, 1])
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and Validation accuracy (Softmax)')
plt.legend()
'''
acc = mlpstat['acc']
val_acc = mlpstat['val_acc']
loss = mlpstat['loss']
val_loss = mlpstat['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation loss (MLP)')
plt.legend()
plt.figure()
plt.ylim([0, 1])
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and Validation accuracy (MLP)')
plt.legend()

acc = cnnstat['acc']
val_acc = cnnstat['val_acc']
loss = cnnstat['loss']
val_loss = cnnstat['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation loss (CNN)')
plt.legend()
plt.figure()
plt.ylim([0, 1])
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and Validation accuracy (CNN)')
plt.legend()
plt.show()

