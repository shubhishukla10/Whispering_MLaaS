import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
import pandas as pd
import ctypes
import pathlib
import os
from pathlib import Path

base_path = str(Path(__file__).parent.parent.parent) + "/"

device = torch.device("cpu")

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
        t = []
        t.append(time.perf_counter())
        x = self.conv1(x)
        t.append(time.perf_counter())
        x = F.relu(x)
        t.append(time.perf_counter())
        x = self.conv2(x)
        t.append(time.perf_counter())
        x = F.relu(x)
        t.append(time.perf_counter())
        x = self.pool(x)
        t.append(time.perf_counter())
        x = self.conv3(x)
        t.append(time.perf_counter())
        x = F.relu(x)
        t.append(time.perf_counter())
        x = self.conv4(x)
        t.append(time.perf_counter())
        x = F.relu(x)
        t.append(time.perf_counter())
        x = self.pool(x)
        t.append(time.perf_counter())
        x = self.conv5(x)
        t.append(time.perf_counter())
        x = F.relu(x)
        t.append(time.perf_counter())
        x = self.conv6(x)
        t.append(time.perf_counter())
        x = F.relu(x)
        t.append(time.perf_counter())
        x = x.view(x.size(0), -1) 
        t.append(time.perf_counter())
        x = self.F1(x)
        t.append(time.perf_counter())
        x = F.relu(x)
        t.append(time.perf_counter())
        x = self.F2(x)    
        t.append(time.perf_counter())
        x = F.relu(x)
        t.append(time.perf_counter())
        output = self.out(x)
        t.append(time.perf_counter())
        return output, t


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001) 

net.load_state_dict(torch.load(base_path+"Models/CustomCNN/cnn"))

pkl_file = open(base_path+'Data/CIFAR10/CNN_Class_data.pkl', 'rb')
X_all = pickle.load(pkl_file)
pkl_file.close()

os.system('mkdir -p '+base_path+'Timing_Data/CIFAR10/CustomCNN/Layer_wise')
for x_i in range(10):
    t_all = []
    print("Layer wise timing data for class "+str(x_i)+" is collected.")
    for y_i in range(1000):
        t_a = [0]*20
        test_output, time_layers = net(X_all[x_i])
        for tim in range(20):
            t_a[tim] = time_layers[tim+1] - time_layers[tim]
        t_all.append(t_a)
    df = pd.DataFrame(t_all, columns=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20'])
    df.to_csv(base_path+'Timing_Data/CIFAR10/CustomCNN/Layer_wise/Class_'+ str(x_i)+'_layerwise.csv')
