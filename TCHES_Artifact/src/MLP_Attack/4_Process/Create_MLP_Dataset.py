import pandas as pd

#Generate Training Data
df_all = pd.DataFrame(columns = ['Image_' + str(i) 
                            for i in range(100)])
for c in range(10):
    all_time = []
    
    for img in range(100):
        file1 = pd.read_csv('Attack_Timing_Data/Class'+str(c)+'/Overall_Inference_Time'+str(c)+'_Image'+str(img)+'.csv')
        dist1 = file1['Time'].to_numpy()

        arr = []
        for i in range(80):
            arr.append(dist2[i])
        all_time.append(arr)
    new_arr = list(map(list, zip(*all_time)))
    print(new_arr[0])

    df = pd.DataFrame(data = new_arr,
                            columns = ['Image_' + str(i) 
                            for i in range(100)])
    
    df_all = pd.concat([df_all, df], axis=0)
    
clas = []
for p in range(800):
    clas.append(int(p/80))
df_all['Class'] = clas

df_all.to_csv('Attack_Dataset/MLP_train_data.csv')
print("Train data generated!")

#Generate Test Data
df_all = pd.DataFrame(columns = ['Image_' + str(i) 
                            for i in range(100)])
for c in range(10):
    all_time = []
    
    for img in range(100):
        file1 = pd.read_csv('Attack_Timing_Data/Class'+str(c)+'/Overall_Inference_Time'+str(c)+'_Image'+str(img)+'.csv')
        dist1 = file1['Time'].to_numpy()

        arr = []
        for i in range(80,100):
            arr.append(dist2[i])
        all_time.append(arr)
    new_arr = list(map(list, zip(*all_time)))
    print(new_arr[0])

    df = pd.DataFrame(data = new_arr,
                            columns = ['Image_' + str(i) 
                            for i in range(100)])
    
    df_all = pd.concat([df_all, df], axis=0)


clas = []
for p in range(200):
    clas.append(int(p/20))
df_all['Class'] = clas

df_all.to_csv('Attack_Dataset/MLP_test_data.csv')
print("Test data generated!")