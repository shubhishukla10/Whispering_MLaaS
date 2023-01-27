import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
import numpy as np
import pandas as pd
import pickle
import ctypes
import pathlib
import os
import sys
from pathlib import Path
home = str(Path.home())

base_path = home + "/TCHES_Artifact/"

libname = base_path + "utils/lib_flush.so"
flush_lib = ctypes.CDLL(libname)

libname = base_path + "utils/lib_flush_pipe.so"
flush_lib_pipe = ctypes.CDLL(libname)

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
        t_1 = perf_counter()
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
        t_2 = perf_counter()
        return output, t_2-t_1


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001) 

net.load_state_dict(torch.load(base_path+"Models/CustomCNN_OpacusDP/cnn"))

X_data=[]
for x_i in range(10):
    pkl_file = open(base_path+'Data/CIFAR10/CNN_Class_'+str(x_i)+'_data.pkl', 'rb')
    X_data.append(pickle.load(pkl_file))
    pkl_file.close()

TI_val = sys.argv[1]
print("Time Instance -->"+str(TI_val))

#Initial Warm-up before actual collection of timing traces
for x_i in range(10):
    for r in range(100):
        flush_lib.main()
        flush_lib_pipe.main()
        for d in range(1):
            test_output, t_in = net(X_data[x_i][r])

os.system('mkdir -p Attack_Timing_Data/TimeInstance_'+str(TI_val))

for x_i in range(10):
    print(x_i)
    df = pd.DataFrame()
    for r in range(100):
        time=[]
        flush_lib.main()
        flush_lib_pipe.main()
        for d in range(500):
            test_output, t_in = net(X_data[x_i][r])
            time.append(t_in*1e6)
        df[str(r)] = time
        df.to_csv('Attack_Timing_Data/TimeInstance_'+str(TI_val)+'/Class_'+str(x_i)+'_100img_timing.csv')