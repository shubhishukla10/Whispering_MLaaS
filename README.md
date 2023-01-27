# Whispering MLaaS
Artifact for the TCHES paper Whispering MLaaS: Exploiting Timing Channels to Compromise User Privacy in Deep Neural Networks

**System Requirements:**

*OS*: Linux

*Memory*: Minimum 16GB

## Usage/Examples

### Distinguish Class Label Pairs

Counting number of class pairs distingushable for CIFAR-10 (out of 45) and CIFAR-100 (out of 4950) dataset.

```
cd TCHES_Artifact/src/Distinguish_Class_Pairs
```
Example for getting results of CIFAR-10 dataset with Alexnet model:
```
taskset -c python3 Collect_inference_timings_CIFAR10.py -m alexnet
python3 Distinguish_Labels_CIFAR10.py -m alexnet
```
The first python script will collect timing samples for each class and the second script will count the number of distingushable pairs out of 45 from the collected samples. In the above example after "-m" you can give "custom_cnn, alexnet, resnet, densenet, squeezenet and vgg" as command line arguments to get results for the respective CNN models.
We can repeat the above for CIFAR-100 dataset as follows:
```
taskset -c python3 Collect_inference_timings_CIFAR100.py -m alexnet
python3 Distinguish_Labels_CIFAR100.py -m alexnet
```
Similarly, you can get results for differentially trained model using Opacus library using the following scripts for CIFAR-10 and CIFAR-100 dataset:

```
taskset -c python3 Collect_inference_timings_with_differential_privacy_CIFAR10.py -m alexnet
python3 Distinguish_Labels_CIFAR10.py -m alexnet
```
```
taskset -c python3 Collect_inference_timings_with_differential_privacy_CIFAR100.py -m alexnet
python3 Distinguish_Labels_CIFAR100.py -m alexnet
```

### Distinguish Class Label Pairs Layerwise

To get number of distinguishable pairs in each layer of the Custom CNN model used in the paper, run the following scripts. We observe that MaxPool layer has the highest number of pairs which are distingushable based on the timing values.
```
taskset -c python3 Collect_Timing_CustomCNN_layerwise.py
python3 Distinguish_Labels_layerwise.py
```
With differential privacy,
```
taskset -c python3 Collect_Timing_CustomCNN_layerwise_with_differential_privacy_CIFAR10.py
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

### Countermeasure
First, create a separate virtual environment for the countermeasure.
To implement the countermeasure for PyTorch on a local system, we require to install the PyTorch library from source. Follow the following reference to build PyTorch from source : https://github.com/pytorch/pytorch#from-source)

Once the PyTorch has been installed from source on your system. Make the following changes to the file *pytorch/aten/src/ATen/native/cpu/MaxPoolKernel.cpp*:

Replace the following code snippet:
```
if ((val > maxval) || std::isnan(val)) {
maxval = val;
maxindex = index;
}
```
with,
```
tmp_arr [ 0 ] = v a l ;
tmp_arr [ 1 ] = maxval ;
maxval = tmp_arr [ ( v a l < maxval ) ∗ 1 ] ;
tmp_arr [ 0 ] = i n d e x ;
tmp_arr [ 1 ] = maxindex ;
maxindex = tmp_arr [ ( v a l < maxval ) ∗ 1 ] ;
```
 
The countermeasure can be evaluated by running the following: [Distinguish Class Label Pairs](#distinguish-class-label-pairs)
