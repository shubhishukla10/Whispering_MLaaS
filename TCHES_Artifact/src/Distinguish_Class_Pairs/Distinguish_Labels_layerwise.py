import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
from pathlib import Path
home = str(Path.home())

base_path = home + "/TCHES_Artifact/"

layer = ["Convolution", "ReLU", "Convolution", "ReLU", "MaxPool", "Convolution", "ReLU", "Convolution", "ReLU", "MaxPool", "Convolution", "ReLU", "Convolution", "ReLU", "Flattening", "Fully Connected", "ReLU", "Fully Connected", "ReLU", "Output"]

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dp", help="Input \"yes\" if want to get results for differentially private models", type=str)

args = parser.parse_args()

for l in range(1,21):
        dfStat_t = pd.DataFrame(columns=['Class i','Class j', 'Statistic', 'P-value(rounded)', 'P-value(exp)'])
        for c_i in range(10):
                for c_j in range(c_i+1,10):
                        if args.dp == "yes":
                                file1 = pd.read_csv(base_path+'Timing_Data/CIFAR10/CustomCNN_DP/Layer_wise/Class_'+ str(c_i)+'_layerwise.csv')
                                file2 = pd.read_csv(base_path+'Timing_Data/CIFAR10/CustomCNN_DP/Layer_wise/Class_'+ str(c_j)+'_layerwise.csv')
                        else:
                                file1 = pd.read_csv(base_path+'Timing_Data/CIFAR10/CustomCNN/Layer_wise/Class_'+ str(c_i)+'_layerwise.csv')
                                file2 = pd.read_csv(base_path+'Timing_Data/CIFAR10/CustomCNN/Layer_wise/Class_'+ str(c_j)+'_layerwise.csv')
                        dist1 = file1['L'+str(l)].to_numpy()
                        dist2 = file2['L'+str(l)].to_numpy()

                        mean_0 = np.mean(dist1[1:])
                        sd_0 = np.std(dist1[1:])
                        mean_1 = np.mean(dist2[1:])
                        sd_1 = np.std(dist2[1:])

                        reduce_time_arr_0 = [x for x in dist1[1:] if (x > mean_0 -  0.5*sd_0)]
                        reduce_time_arr_0 = [x for x in reduce_time_arr_0 if (x < mean_0 + 0.5*sd_0)]

                        reduce_time_arr_1 = [x for x in dist2[1:] if (x > mean_1 - 0.5*sd_1)]
                        reduce_time_arr_1 = [x for x in reduce_time_arr_1 if (x < mean_1 + 0.5*sd_1)]

                        mean_00 = np.mean(reduce_time_arr_0)
                        sd_00 = np.std(reduce_time_arr_0)
                        median_00 = np.median(reduce_time_arr_0)
                        mean_10 = np.mean(reduce_time_arr_1)
                        sd_10 = np.std(reduce_time_arr_1)
                        median_10 = np.median(reduce_time_arr_1)

                        ttest_res = ttest_ind(reduce_time_arr_0, reduce_time_arr_1, equal_var=False)

                        dfStat_t = dfStat_t.append({'Class i' : c_i, 'Class j': c_j, 'Statistic' : ttest_res.statistic, 'P-value(rounded)':round(ttest_res.pvalue, 6), 'P-value(exp)':ttest_res.pvalue}, ignore_index=True)

        cnt = 0
        for i in range(45):
                if dfStat_t['Statistic'][i] >= 4.5 or dfStat_t['Statistic'][i] <= -4.5:
                        cnt+=1
        print("Layer Name: " + layer[l-1] +", Distingushable Pairs (out of 45): " + str(cnt))
                        
