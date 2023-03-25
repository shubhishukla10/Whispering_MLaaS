import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.stats import mannwhitneyu, ttest_ind
import argparse
from pathlib import Path

base_path = str(Path(__file__).parent.parent.parent) + "/"

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", help="Input the model name from following options: custom_cnn, alexnet, densenet, resnet, vgg, squeezenet", type=str)
parser.add_argument("-d", "--dp", help="Input \"yes\" if want to get results for differentially private models", type=str)

args = parser.parse_args()

model_list = ["custom_cnn", "alexnet", "densenet", "resnet", "vgg", "squeezenet"]

if args.model in model_list:
        dfStat_t = pd.DataFrame(columns=['Class i','Class j', 'Statistic', 'P-value(rounded)', 'P-value(exp)'])
        
        if args.dp == "yes":
                data_path = base_path+"Timing_Data/CIFAR10/"+args.model+"_DP/Full_Function/"
        else:
                data_path = base_path+"Timing_Data/CIFAR10/"+args.model+"/Full_Function/"

        for c_i in range(10):
                file1 = pd.read_csv(data_path +'Class_'+ str(c_i) +'.csv')

                for c_j in range(c_i+1,10):
                        file2 = pd.read_csv(data_path +'Class_'+ str(c_j) +'.csv')
                        dist1 = file1['Time'].to_numpy()        # dist1 stores timing values for Class i
                        dist2 = file2['Time'].to_numpy()        # dist2 stores timing values for Class j
                        
                        # Calculating mean and standard deviation for both timing distributions
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
                               
                        # Perform t-test on the timing distributions of Class i and j
                        ttest_res = ttest_ind(reduce_time_arr_0, reduce_time_arr_1, equal_var=False)

                        dfStat_t = dfStat_t.append({'Class i' : c_i, 'Class j': c_j, 'Statistic' : ttest_res.statistic, 'P-value(rounded)':round(ttest_res.pvalue, 6), 'P-value(exp)':ttest_res.pvalue}, ignore_index=True)
        # Store t-test results for all class pairs    
        dfStat_t.to_csv(data_path+'/TTest_Results.csv')

        cnt = 0
        for i in range(45):
                if dfStat_t['Statistic'][i] >= 4.5 or dfStat_t['Statistic'][i] <= -4.5:
                        cnt+=1
        print("Total distinguishable pairs (out of 45) for "+args.model+" model: "+ str(cnt))
else:
        print("ERROR: Give a valid model name after -m among the following options: custom_cnn, alexnet, densenet, resnet, vgg, squeezenet")
                        
