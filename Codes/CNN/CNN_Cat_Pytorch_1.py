import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch as t
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F
from torchvision.models import resnet34
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


cpu = t.device('cpu')
## Check GPU
if cuda.is_available():
    cuda0 = t.device('cuda:0')
    
## Some setting
learning_rate = 0.01
epoch_time = 50
batch = 128
pic_size = 64


## CNN model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 3 input image channel, 20 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(8450, 300)
        self.fc2 = nn.Linear(300, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        # print('x value: ', self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

## transform method
transform = T.Compose([
    T.Resize(pic_size), 
    T.CenterCrop(pic_size), 
    T.ToTensor(), 
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
])

## Load dataset
train_dataset = ImageFolder('cat_set2/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch, 
                    shuffle=True, num_workers=0, 
                    drop_last=False)
print(train_dataset.class_to_idx)
print('train set num', len(train_dataset.imgs))
print(train_dataset[0][0].size())   

test_dataset = ImageFolder('cat_set2/tset', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset.imgs), 
                    shuffle=True, num_workers=0, 
                    drop_last=False)
print(test_dataset.class_to_idx)
print('test set num', len(test_dataset.imgs))
print(test_dataset[0][0].size())
print(test_loader)

## Initial model
model = LeNet()
if cuda.is_available():
   model = LeNet().cuda()
   print(model)

optimizer = t.optim.SGD(model.parameters(), lr = learning_rate)
loss_func = t.nn.CrossEntropyLoss()


## Start training
print('star training!')
tic = time.time() 

for epoch in range(epoch_time):
   ## record loss   
   loss_average = np.zeros(1)
   for step, (batch_x, batch_y) in enumerate(train_loader):         
       # print('step=', step)
       if cuda.is_available():
           batch_x = batch_x.cuda()
           batch_y = batch_y.cuda()
       optimizer.zero_grad()
       prediction = model(batch_x)
       loss = loss_func(prediction, batch_y)
       loss.backward()
       optimizer.step()
       if cuda.is_available():
          loss_cpu = loss.cpu().data.numpy()
       else:
          loss_cpu = loss.data.numpy()
       loss_average = np.add(loss_average, loss_cpu/batch)

   if epoch % 5 == 0:
       print('Epoch=', epoch)
       print('Loss=%.4f' % loss_average)

toc = time.time() 
print('train time: ' + str((toc - tic)) + 'sec')
print('training ok!')


## Testing predict3
for step, (batch_x, batch_y) in enumerate(test_loader):
    if cuda.is_available():
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        y_test_hat_tensor = model(batch_x)
        y_test_hat = y_test_hat_tensor.cpu().data.numpy()
        y_test = batch_y.cpu().data.numpy()
    else:
        ## Only have CPU
        y_test_hat_tensor = model(batch_x)
        y_test_hat = y_test_hat_tensor.cpu().data.numpy()
        y_test = batch_y.cpu().data.numpy()

## change float to index 
y_test_hat = np.argmax(y_test_hat, axis=1)
print(y_test_hat)
print(y_test)
print("Test set score: %f" % accuracy_score(y_test, y_test_hat))

