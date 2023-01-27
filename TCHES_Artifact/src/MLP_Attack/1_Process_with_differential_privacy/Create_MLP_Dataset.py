import numpy as np
import pandas as pd
import pickle
import ctypes
import pathlib
import os
import sys

#Create Training Data
med_all = []
for x_i in range(10):
    print("Class-->"+str(x_i))
    for ti in range(800):
        # print(ti)
        file1 = pd.read_csv('Attack_Timing_Data/TimeInstance_'+str(ti)+'/Class_'+str(x_i)+'_100img_timing.csv')
        tmp_in = []
        for img in range(100):
            data = file1[str(img)]
            tmp_in.append(np.median(data))
        tmp_in.append(x_i)
        med_all.append(tmp_in)

df = pd.DataFrame(med_all)
df_label =  pd.DataFrame()

for i in range(100):
    df_label["Median"+str(i)] = df[i]
df_label["Class"] = df[100]
df_label.to_csv('Attack_Timing_Data/Attack_train_data.csv')

#Create Test Data
med_all = []
for x_i in range(10):
    print("Class-->"+str(x_i))
    for ti in range(801,1000):
        # print(ti)
        file1 = pd.read_csv('Attack_Timing_Data/TimeInstance_'+str(ti)+'/Class_'+str(x_i)+'_100img_timing.csv')
        tmp_in = []
        for img in range(100):
            data = file1[str(img)]
            tmp_in.append(np.median(data))
        tmp_in.append(x_i)
        med_all.append(tmp_in)

df = pd.DataFrame(med_all)
df_label =  pd.DataFrame()

for i in range(100):
    df_label["Median"+str(i)] = df[i]
df_label["Class"] = df[100]
df_label.to_csv('Attack_Timing_Data/Attack_test_data.csv')