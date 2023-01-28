import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
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
import sys
import argparse
from pathlib import Path
home = str(Path.home())

base_path = home + "/TCHES_Artifact/"

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", help="Input the model name from following options: custom_cnn, alexnet, densenet, resnet, vgg, squeezenet ", type=str)

args = parser.parse_args()

libname = base_path+"utils/lib_flush.so"
flush_lib = ctypes.CDLL(libname)

libname = base_path+"utils/lib_flush_pipe.so"
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
        self.out = nn.Linear(64, 100)


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


if args.model:
    print("Displaying Output as: % s" % args.model)
    flag=0
    if args.model == "alexnet":
        model = models.alexnet(pretrained=True)
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
        model.load_state_dict(torch.load(base_path+"Models/CIFAR100/Alexnet_OpacusDP/cnn"))
        model = model.to(device)
        flag = 1
    if args.model == "densenet":
        model = models.densenet121(pretrained=True)
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
        model.load_state_dict(torch.load(base_path+"Models/CIFAR100/Densenet_OpacusDP/cnn"))
        model = model.to(device)
        flag = 1
    if args.model == "resnet":
        model = models.resnet50(pretrained=True)
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
        model.load_state_dict(torch.load(base_path+"Models/CIFAR100/Resnet_OpacusDP/cnn"))
        model = model.to(device)
        flag = 1
    if args.model == "vgg":
        model = models.vgg19(pretrained=True)
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
        model.load_state_dict(torch.load(base_path+"Models/CIFAR100/VGG19_OpacusDP/cnn"))
        model = model.to(device)
        flag = 1
    if args.model == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
        model.load_state_dict(torch.load(base_path+"Models/CIFAR100/Squeezenet_OpacusDP/cnn"))
        model = model.to(device)
        flag = 1
    if args.model == "custom_cnn":
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001) 
        model.load_state_dict(torch.load(base_path+"Models/CIFAR100/CustomCNN_OpacusDP/cnn"))
        flag = 1

    if flag:
        if not os.path.exists(base_path+'Timing_Data/CIFAR100/'+args.model+'/Full_Function'):
            os.system('mkdir -p ' + base_path + 'Timing_Data/CIFAR100/'+args.model+'/Full_Function')

        for x_i in range(100):
            if args.model == "custom_cnn":
                pkl_file = open(base_path+'Data/CIFAR100/CNN_Class_'+str(x_i)+'_data.pkl', 'rb')
            else:
                pkl_file = open(base_path+'Data/CIFAR100/Resize224/CNN_Class_'+str(x_i)+'_data.pkl', 'rb')
            X_data = pickle.load(pkl_file)
            pkl_file.close()
            t_all = []
            print("Collecting class " +str(x_i)+ " inference time values ...")
            flush_lib.main()
            flush_lib_pipe.main()
            for img in range(10):
                for y_i in range(100):
                    t1 = time.perf_counter()
                    test_output = model(X_data[img*10])
                    t2 = time.perf_counter()
                    t_all.append((t2-t1)*1e6)
            df = pd.DataFrame(t_all, columns=['Time'])
            df.to_csv(base_path+'Timing_Data/CIFAR100/'+args.model+'/Full_Function/Class_'+str(x_i)+'.csv')
        print("Finished!")
    else:
        print("ERROR: Give a valid model name after -m among the following options: custom_cnn, alexnet, densenet, resnet, vgg, squeezenet")
