import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Net_RES(nn.Module):
    def __init__(self):
        super(Net_RES, self).__init__()
        self.b1 = ResBlock(1,32)
        self.b2 = ResBlock(32,32)
        self.b3 = ResBlock(32,16)
        self.b4 = ResBlock(16,8)
        self.fc1 = nn.Linear(8*4*4, 800)
        #self.fc1_b = nn.Linear(800, 200)
        self.fc2 = nn.Linear(800, 10)

    def forward(self, x):
        x = self.b1(x)
        x = F.max_pool2d(x,2)

        x = self.b2(x)
        x = F.max_pool2d(x,2)

        x = self.b3(x)
        x = F.max_pool2d(x,2)

        x = self.b4(x)
        x = F.max_pool2d(x,2)

        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #x = F.relu(self.fc1_b(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Net_RES_32(nn.Module):
    def __init__(self):
        super(Net_RES_32, self).__init__()
        self.b1 = ResBlock(1,32)
        self.b2 = ResBlock(32,32)
        self.b3 = ResBlock(32,64)
        self.b4 = ResBlock(64,128)
        self.fc1 = nn.Linear(64*4*4, 300)
        #self.fc1_b = nn.Linear(800, 200)
        self.fc2 = nn.Linear(300, 10)

    def forward(self, x):
        x = self.b1(x)
        x = F.max_pool2d(x,2)

        x = self.b2(x)
        x = F.max_pool2d(x,2)

        x = self.b3(x)
        x = F.max_pool2d(x,2)
        #x = self.b4(x)
        #x = F.max_pool2d(x,2)

        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #x = F.relu(self.fc1_b(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, dataloader_train, optimizer, epoch, criterion):
    model.train()
    sum_loss = 0
    for i, (data,target) in enumerate(dataloader_train, 0):
        data, target = data.to(device), target.to(device)
        print(data.shape)
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
        return loss.item()
def test(model, test_loader):
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
    return test_loss, 100. * correct / len(test_loader.dataset)

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
prc_test = prc_train + int(0.3*len(images))
#Chargement des partitions
def load():
    out = []
    prc = [prc_train,prc_test]
    for i in range(len(prc)):
        if i == 0:
            images_train = [(im) for im in images[:prc[i]]]
            classes_train = classes[:prc_train]
        #elif i == len(prc)-1:
        #    images_train = [(im) for im in images[prc[i]:]]
        #    classes_train = classes[prc[i]:]
        else:
            images_train = [(im) for im  in images[prc[i-1]:prc[i]]]
            classes_train = classes[prc_train:prc_test]
        tensor_x_train = torch.stack([torch.Tensor(i) for i in images_train])
        tensor_y_train = torch.from_numpy(classes_train).long()
        datas_train = utils.TensorDataset(tensor_x_train,tensor_y_train)
        dataloader_train = utils.DataLoader(datas_train,50, shuffle=True)
        out.append(dataloader_train)
    return out
[dataloader_train,dataloader_test] = load()




#Création de la loss
criterion = nn.NLLLoss()
model = Net_RES_32().to(device)
#choix de l'optimisateur
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epoch_arr = range(5000)
train_loss = []
test_loss = []
max_test = 0

#Entrainement
for epoch in epoch_arr:
    train_loss.append( train(model, dataloader_train, optimizer, epoch, F.nll_loss))
    loss,test_prc = test(model, dataloader_test)
    if test_prc > max_test:
        max_test = test_prc
        torch.save(model.state_dict(), './best.pth')
    test_loss.append( loss )
print("max acc : ",max_test)
print("Used Seed: ", manualSeed)
#Affichage des courbes de loss

plt.plot(epoch_arr,train_loss,'b')
plt.plot(epoch_arr,test_loss,'g')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()



model.load_state_dict(torch.load("./best.pth"))
def test_on(model,path):
    model.eval()
    Trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
    im = Image.open(path).convert('L').resize((32,32))
    #im.show()
    im = torch.autograd.Variable(Trans(im).unsqueeze_(0))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
            im = im.to(device)
            output = model(im)
            print("Output : ",output)
            print("Output : ",output.argmax(dim=1, keepdim=True))
test_on(model,"./datas/eval/image_1.jpg")
