import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
import pandas as pd
import sys
from time import perf_counter_ns
from csv import writer
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path

base_path = str(Path(__file__).parent.parent.parent.parent) + "/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,2)
        self.conv2 = nn.Conv2d(16,32,3)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(32,32,3)
        self.conv4 = nn.Conv2d(32,32,3)
        self.conv5 = nn.Conv2d(32,64,1)
        self.conv6 = nn.Conv2d(64,128,1)
        self.F1 = nn.Linear(2048, 128)
        self.F2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1) 
        x = self.F1(x)
        x = F.relu(x)
        x = self.F2(x)   
        x = F.relu(x)
        output = self.out(x)
        return output


net = Net()

Class_Val = sys.argv[1]
Img_Val = sys.argv[2]

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001) 

net.load_state_dict(torch.load(base_path+"Models/CustomCNN/cnn"))

X_data=[]
for x_i in range(10):
    pkl_file = open(base_path+'Data/CIFAR10/CNN_Class_'+str(x_i)+'_data.pkl', 'rb')
    X_data.append(pickle.load(pkl_file))

# print("Inside Victim Inference")
for i in range(1000):
    test_output = net(X_data[int(Class_Val)][(int(Img_Val))])

