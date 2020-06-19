import numpy as np
import time
import math

from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


class ScratchNet(nn.Module):
    def __init__(self):
        super(ScratchNet, self).__init__()
        
 
    def forward(self, x):
        
        return x


## Some setting for network training
learning_rate = 0.1
epoch_time = 50
batch = 128
pic_size = 64


## transform method
transform = T.Compose([
    T.Resize(pic_size), 
    T.CenterCrop(pic_size), 
    T.ToTensor(), 
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
])

## Read data
train_dataset = ImageFolder('cat_set2/train', transform=transform)
print(train_dataset.class_to_idx)
print('train set num', len(train_dataset.imgs))
print(train_dataset[0][0].size())

train_loader = DataLoader(train_dataset, batch_size=batch, 
                    shuffle=True, num_workers=0, 
                    drop_last=False)

val_dataset = ImageFolder('cat_set2/val', transform=transform)
# print(val_dataset.class_to_idx)
print('validation set num', len(val_dataset.imgs))
# print(val_dataset[0][0].size())
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset.imgs), 
                    shuffle=True, num_workers=0, 
                    drop_last=False)

test_dataset = ImageFolder('cat_set2/tset', transform=transform)
# print(test_dataset.class_to_idx)
print('test set num', len(test_dataset.imgs))
# print(test_dataset[0][0].size())
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset.imgs), 
                    shuffle=True, num_workers=0, 
                    drop_last=False)