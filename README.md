# "Whispering MLaaS" Exploiting Timing Channels to Compromise User Privacy in Deep Neural Networks
Artifact for the TCHES paper Whispering MLaaS: Exploiting Timing Channels to Compromise User Privacy in Deep Neural Networks

**System Requirements:**

- *OS*: Linux

- *Memory*: Minimum 16GB
- *Python 3.8 or above*

- Install the necessary python packages using the provided *requirements.txt* file using the below command:
  ```
  pip install -r requirements.txt
  ```
**Clone the GitHub repository and then move the ``TCHES_Artifact`` folder to your home directory.**

Download the data and models from [here](https://drive.google.com/drive/folders/1LOzsXqyVSHymXbVUeRejMJE6EpwIpKPL?usp=share_link) and move them to *Data* and *Models* directory inside *TCHES_Artifact*. The Models directory contains pre-trained CNN models. The Data directory consists of CIFAR-10 and CIFAR-100 datasets.

## Usage

### Distinguish Class Label Pairs

Counting number of class pairs distingushable for CIFAR-10 (out of 45) and CIFAR-100 (out of 4950) dataset.

```
cd TCHES_Artifact/src/Distinguish_Class_Pairs
```
Example for getting results of CIFAR-10 dataset with Alexnet model:
```
taskset -c 0 python3 Collect_inference_timings_CIFAR10.py -m alexnet
python3 Distinguish_Labels_CIFAR10.py -m alexnet
```
The first python script will collect timing samples for each class and the second script will count the number of distingushable pairs out of 45 from the collected samples. In the above example after ``-m`` you can give ``custom_cnn``, ``alexnet``, ``resnet``, ``densenet``, ``squeezenet`` and ``vgg``" as command line arguments to get results for the respective CNN models.
We can repeat the above for CIFAR-100 dataset as follows:
```
taskset -c 0 python3 Collect_inference_timings_CIFAR100.py -m alexnet
python3 Distinguish_Labels_CIFAR100.py -m alexnet
```
Similarly, you can get results for differentially trained model using Opacus library using the following scripts for CIFAR-10 and CIFAR-100 dataset:

```
taskset -c 0 python3 Collect_inference_timings_with_differential_privacy_CIFAR10.py -m alexnet
python3 Distinguish_Labels_CIFAR10.py -m alexnet
```
```
taskset -c 0 python3 Collect_inference_timings_with_differential_privacy_CIFAR100.py -m alexnet
python3 Distinguish_Labels_CIFAR100.py -m alexnet
```

### Distinguish Class Label Pairs Layerwise

To get number of distinguishable pairs in each layer of the Custom CNN model used in the paper, run the following scripts. We observe that MaxPool layer has the highest number of pairs which are distingushable based on the timing values.
```
taskset -c 0 python3 Collect_Timing_CustomCNN_layerwise.py
python3 Distinguish_Labels_layerwise.py
```
With differential privacy,
```
taskset -c 0 python3 Collect_Timing_CustomCNN_layerwise_with_differential_privacy_CIFAR10.py
python3 Distinguish_Labels_layerwise.py -d yes
```

### MLP Attack

#### Single Process Attack
To build an MLP class label classifier for attack purpose, we first collect the timing traces for training our classifier. We set aside 20% of them for testing the attack model. Run the following scripts for the attack:

```
cd TCHES_Artifact/src/MLP_Attack/1_Process
python3 Call_trace_generation.py
python3 Run_MLP_Attack.py
```
With differential privacy,
```
cd TCHES_Artifact/src/MLP_Attack/1_Process_with_differential_privacy
python3 Call_trace_generation.py
python3 Run_MLP_Attack.py
```

#### 4 Process Attack
To run the MLP attack in a noisy setup, where 3 processes are executing in parallel with the victim client, execute the following scripts:
```
cd TCHES_Artifact/src/MLP_Attack/4_Process
gcc Fork_Spy_Victim_create_MLP_dataset.c -o Collect_Attack_data
./Collect_Attack_data
python3 Create_MLP_Dataset.py
python3 Run_MLP_Attack.py
```

#### 8 Process Attack
To run the MLP attack in a noisy setup, where 7 processes are executing in parallel with the victim client, execute the following scripts:
```
cd TCHES_Artifact/src/MLP_Attack/8_Process
gcc Fork_Spy_Victim_create_MLP_dataset.c -o Collect_Attack_data
./Collect_Attack_data
python3 Create_MLP_Dataset.py
python3 Run_MLP_Attack.py
```

The collection of timing traces takes up a lot of time, specially for the multi-process attacks. Hence, we have provided training and test datasets for 1-Process, 4-Process and 8-Process attacks inside ``TCHES_Artifact/Attack_data``. We also provide the script ``Run_MLP_Attack.py`` to run the attack model inside the same directory.

### Countermeasure
First, create a separate virtual environment for the countermeasure.
To implement the countermeasure for PyTorch on a local system, we require to install the PyTorch library from source. Follow the following reference to build PyTorch from source : https://github.com/pytorch/pytorch#from-source)

Once the PyTorch has been successfully installed from source on your system. Make the following changes to the file *pytorch/aten/src/ATen/native/cpu/MaxPoolKernel.cpp* and build PyTorch again:

Replace the following code snippet:
```
if ((val > maxval) || std::isnan(val)) {
maxval = val;
maxindex = index;
}
```
with,
```
tmp_arr[0] = val ;
tmp_arr[1] = maxval ;
maxval = tmp_arr[( val < maxval ) ∗ 1] ;
tmp_arr[0] = index ;
tmp_arr[1] = maxindex ;
maxindex = tmp_arr [( val < maxval ) ∗ 1] ;
```
Also initialize ``int64_t tmp_arr[2];`` at the beginning of the file with other variables.
 
The countermeasure can be evaluated by running experiments from the following sections:
- [Distinguish Class Label Pairs](#distinguish-class-label-pairs): The number of distinguishable pairs should be less than 50% of the total number of pairs now.

- [Distinguish Class Label Pairs Layerwise](#distinguish-class-label-pairs-layerwise): The number of distinguishable pairs should be less than 50% of the total number of pairs now for the MaxPool layer.
- [MLP Attack](#mlp-attack): The attack accuracy should drop to 10%-15%.
