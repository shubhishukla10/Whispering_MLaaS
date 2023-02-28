# "Whispering MLaaS" Exploiting Timing Channels to Compromise User Privacy in Deep Neural Networks
Artifact for the TCHES 2023 (Issue 2) paper **Whispering MLaaS: Exploiting Timing Channels to Compromise User Privacy in Deep Neural Networks**.

**System Requirements:**

- *OS*: Linux
- *x86 processor*
- *Memory*: Minimum 16GB
- *Python 3.8 or above*
-
- Install PyTorch using the following command:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
- Install the other necessary python packages using the provided *requirements.txt* file using the below command:
  ```
  pip install -r requirements.txt
  ```

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

*In all our python scripts for collecting timing values we flush the pipeline and cache before collecting inference timing traces for each class label to make the start state consistent for inference of each class label. We have included the scripts for flushing cache and pipeline inside the utils directory, which have been used in all our timing trace collection codes.*

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
python3 Distinguish_Labels_CIFAR100.py -m alexnet -d yes
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
To build an MLP class label classifier for attack purpose, we first collect the timing traces for training our classifier. We set aside 20% of them for testing the attack model. *For the single process attack we assume that the adversary collects the timing traces from inside the victim's code itself to get the least noisy timing traces.* Run the following scripts for the attack:

```
cd TCHES_Artifact/src/MLP_Attack/1_Process
python3 Call_trace_generation.py
python3 Create_MLP_Dataset.py
python3 Run_MLP_Attack.py
```
With differential privacy,
```
cd TCHES_Artifact/src/MLP_Attack/1_Process_with_differential_privacy
python3 Call_trace_generation.py
python3 Create_MLP_Dataset.py
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
gcc Fork_Spy_Victim_create_MLP_dataset_8Process.c -o Collect_Attack_data
./Collect_Attack_data
python3 Create_MLP_Dataset.py
python3 Run_MLP_Attack.py
```

The collection of timing traces takes up a lot of time, specially for the multi-process attacks. Hence, we have provided training and test datasets for 1-Process, 4-Process and 8-Process attacks inside ``TCHES_Artifact/Attack_data``. We also provide the script ``Run_MLP_Attack.py`` to run the attack model inside the same directory.

### Docker Image
We have created a docker image which has all the dependencies and the source code pre-installed. The docker image can be installed using the following command:

```docker pull shubhi1011/whispering_mlaas_tches```

### License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

Copyright (C) 2023 [Secured Embedded Architecture Lab (SEAL), IIT Kharagpur](https://cse.iitkgp.ac.in/resgrp/seal/)
