from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.utils.data as utils
from random import shuffle
import matplotlib.pyplot as plt

#Affichage de la graine pour reproduire des apprentissage
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device("cuda:0")

batchnorm = nn.BatchNorm1d(512).cuda()
#Création des différents modèles de réseaux
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 1)

        self.fc1 = nn.Linear(5*5*128, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x= self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Net_FC(nn.Module):
    def __init__(self):
        super(Net_FC, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*5*5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def train(model, device, dataloader_train, optimizer, epoch, criterion):
    model.train()
    sum_loss = 0
    for i, (data,target) in enumerate(dataloader_train, 0):
        data, target = data.to(device), target.to(device)
        #print(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        #print(output)
        loss.backward()
        sum_loss += loss.item()
        optimizer.step()
        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(dataloader_train.dataset),
                100. * i / len(dataloader_train), loss.item()))
        return sum_loss / len(dataloader_train)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data,target) in enumerate(test_loader, 0):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return test_loss

#Chargement des images
Trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
images = [Trans(Image.fromarray(im)) for im in np.load("./datas/X.npy")]
#Chargement des classes
temp = np.load("./datas/Y.npy")
classes = np.zeros(len(images))
for i in range(len(classes)):
    classes[i] = temp[i].nonzero()[0][0]
#Mélange du dataset
temp_s = []
for i in range(len(classes)):
    temp_s.append([images[i],classes[i] ])
shuffle(temp_s)
for i in range(len(classes)):
    [im,c] = temp_s[i]
    images[i] = im
    classes[i] = c
#Définition des partitions du dataset
prc_train = int(0.7*len(images))
prc_test = prc_train + int(0.2*len(images))
prc_eval = prc_test + int(0.1*len(images))
#Chargement des partitions
def load():
    out = []
    prc = [prc_train,prc_test,prc_eval]
    for i in range(len(prc)):
        if i == 0:
            images_train = [(im) for im in images[:prc[i]]]
            classes_train = classes[:prc_train]
        elif i == len(prc)-1:
            images_train = [(im) for im in images[prc[i]:]]
            classes_train = classes[prc[i]:]
        else:
            images_train = [(im) for im  in images[prc[i-1]:prc[i]]]
            classes_train = classes[prc_train:prc_test]
        tensor_x_train = torch.stack([torch.Tensor(i) for i in images_train])
        tensor_y_train = torch.from_numpy(classes_train).long()
        datas_train = utils.TensorDataset(tensor_x_train,tensor_y_train)
        dataloader_train = utils.DataLoader(datas_train,50, shuffle=True)
        out.append(dataloader_train)
    return out
[dataloader_train,dataloader_test,dataloader_eval] = load()




#Création de la loss
criterion = nn.NLLLoss()

model = Net().to(device)
#choix de l'optimisateur
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epoch_arr = range(500)
train_loss = []
test_loss = []
#Entrainement
for epoch in epoch_arr:
    train_loss.append( train(model, device, dataloader_train, optimizer, epoch, F.nll_loss))
    test_loss.append( test(model, device, dataloader_test) )
#Affichage des courbes de loss
plt.plot(epoch_arr,train_loss,'b')
plt.plot(epoch_arr,test_loss,'g')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

