# Whispering_MLaaS
Artifact for the TCHES paper Whispering MLaaS: Exploiting Timing Channels to Compromise User Privacy in Deep Neural Networks

System Requirements:
OS: Linux
Memory: Minimum 16GB

## Usage/Examples
**Distinguish Class Label Pairs**

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

**Distinguish Class Label Pairs Layerwise**
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
