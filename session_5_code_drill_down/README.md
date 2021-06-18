- [1. Session 5 Assignment MNIST 99.4% accuracy with 8K parameters](#1-session-5-assignment-mnist-994-accuracy-with-8k-parameters)
  - [1.1. Iteration 1](#11-iteration-1)
    - [1.1.1. Target](#111-target)
    - [1.1.2. Results](#112-results)
    - [1.1.3. Analysis](#113-analysis)
  - [1.2. Iteration 2](#12-iteration-2)
    - [1.2.1. Target](#121-target)
    - [1.2.2. Result](#122-result)
    - [1.2.3. Analysis](#123-analysis)
  - [1.3. Iteration 3](#13-iteration-3)
    - [1.3.1. Target](#131-target)
    - [1.3.2. Result](#132-result)
    - [1.3.3. Analysis](#133-analysis)

# 1. Session 5 Assignment MNIST 99.4% accuracy with 8K parameters

## 1.1. Iteration 1

[assignment_5_8k_parameters_ver1.ipynb](assignment_5_8k_parameters_ver1.ipynb)

### 1.1.1. Target

The target for the first iteration is to build a model skeleton.  The skeleton is described below

- Convolution Block 1: # of channels is 8
- Transition Block 1: maxpool at RF=7 
- Convolution Block 2: # of channels is 12
- Output Block: GAP followed by 1x1

The skeleton model has a ResNet like architecture: the first convolution block has 8 channels in each layer; the 2nd convolution block has 12 channels in each layer;  

The number of channels(kernels) in each block have been chosen so that the # of aparamters are close to 8000.

Maxpooling has been placed at a receptive field of 7 to ensure that it is done after the edges and gradients are extracted..

Dropout Image rotation is also performed.  This regularization techniques will ensure that the model doesn't overfit the data.

BatchNorm is used in each layer to ensure that the input to each of the hidden layers is normalized with a mean of 0 and variance of 1.  

Output layer does not use FC.  Instead, as suggested, GAP followed by 1x1 was used.  This sequence reduces the number of parameters in the model (as opposed to 1x1 followed by GAP)

Model Summary: Model has 8088 parameters

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             576
              ReLU-6            [-1, 8, 28, 28]               0
       BatchNorm2d-7            [-1, 8, 28, 28]              16
         Dropout2d-8            [-1, 8, 28, 28]               0
            Conv2d-9            [-1, 8, 28, 28]             576
             ReLU-10            [-1, 8, 28, 28]               0
      BatchNorm2d-11            [-1, 8, 28, 28]              16
        Dropout2d-12            [-1, 8, 28, 28]               0
        MaxPool2d-13            [-1, 8, 14, 14]               0
           Conv2d-14           [-1, 12, 14, 14]              96
           Conv2d-15           [-1, 12, 12, 12]           1,296
             ReLU-16           [-1, 12, 12, 12]               0
      BatchNorm2d-17           [-1, 12, 12, 12]              24
        Dropout2d-18           [-1, 12, 12, 12]               0
           Conv2d-19           [-1, 12, 10, 10]           1,296
             ReLU-20           [-1, 12, 10, 10]               0
      BatchNorm2d-21           [-1, 12, 10, 10]              24
        Dropout2d-22           [-1, 12, 10, 10]               0
           Conv2d-23             [-1, 12, 8, 8]           1,296
             ReLU-24             [-1, 12, 8, 8]               0
      BatchNorm2d-25             [-1, 12, 8, 8]              24
        Dropout2d-26             [-1, 12, 8, 8]               0
           Conv2d-27             [-1, 12, 6, 6]           1,296
             ReLU-28             [-1, 12, 6, 6]               0
      BatchNorm2d-29             [-1, 12, 6, 6]              24
        Dropout2d-30             [-1, 12, 6, 6]               0
           Conv2d-31             [-1, 12, 4, 4]           1,296
             ReLU-32             [-1, 12, 4, 4]               0
      BatchNorm2d-33             [-1, 12, 4, 4]              24
        Dropout2d-34             [-1, 12, 4, 4]               0
        MaxPool2d-35             [-1, 12, 2, 2]               0
AdaptiveAvgPool2d-36             [-1, 12, 1, 1]               0
           Conv2d-37             [-1, 10, 1, 1]             120
================================================================
Total params: 8,088
Trainable params: 8,088
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.74
Params size (MB): 0.03
Estimated Total Size (MB): 0.77
----------------------------------------------------------------
```

### 1.1.2. Results

Best Test Accuracy of 99.4% in Epoch 20.

```text
Epoch: 20
Train phase: Loss=0.04622117802500725 Batch_id=468 Accuracy=98.73: 100%|██████████| 469/469 [00:27<00:00, 16.82it/s]

Test phase: Average loss: 0.0202, Accuracy: 9940/10000 (99.40%)
```

### 1.1.3. Analysis

- The model is underfitting (training accuracy < test accuracy)..  this is due to the use of regularization in the model: image rotation and dropouts
- The parameters are greater than 8K (8088) and needs to be optimized in the next iteration
- The number of epochs will also be reduced to 14 in the next iteration.

## 1.2. Iteration 2

[assignment_5_8k_parameters_ver1.ipynb](assignment_5_8k_parameters_ver2.ipynb)

### 1.2.1. Target

The objective is to

- reduce the epochs from 20 to 14
- Try StepLR to see if the accuracy can be pushed up compared to the last iteration: 
  - changing StepLR from 0.19 to 0.04, step_size from 7 to 2 and gamma from 0.5 to 0.09.  StepLR ensures that, during later epochs, the LR is adjusted to a smaller number, so that the weights are adjusted more slowly.  In earlier epochs, the weights will be adjusted faster, due to a high LR, to make it coverge faster.
  - also not applying stepLR during initial epochs.  Step LR is applied at epoch 8 and later..

The model summary is shown below.  Model has 8088 parameters

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             576
              ReLU-6            [-1, 8, 28, 28]               0
       BatchNorm2d-7            [-1, 8, 28, 28]              16
         Dropout2d-8            [-1, 8, 28, 28]               0
            Conv2d-9            [-1, 8, 28, 28]             576
             ReLU-10            [-1, 8, 28, 28]               0
      BatchNorm2d-11            [-1, 8, 28, 28]              16
        Dropout2d-12            [-1, 8, 28, 28]               0
        MaxPool2d-13            [-1, 8, 14, 14]               0
           Conv2d-14           [-1, 12, 14, 14]              96
           Conv2d-15           [-1, 12, 12, 12]           1,296
             ReLU-16           [-1, 12, 12, 12]               0
      BatchNorm2d-17           [-1, 12, 12, 12]              24
        Dropout2d-18           [-1, 12, 12, 12]               0
           Conv2d-19           [-1, 12, 10, 10]           1,296
             ReLU-20           [-1, 12, 10, 10]               0
      BatchNorm2d-21           [-1, 12, 10, 10]              24
        Dropout2d-22           [-1, 12, 10, 10]               0
           Conv2d-23             [-1, 12, 8, 8]           1,296
             ReLU-24             [-1, 12, 8, 8]               0
      BatchNorm2d-25             [-1, 12, 8, 8]              24
        Dropout2d-26             [-1, 12, 8, 8]               0
           Conv2d-27             [-1, 12, 6, 6]           1,296
             ReLU-28             [-1, 12, 6, 6]               0
      BatchNorm2d-29             [-1, 12, 6, 6]              24
        Dropout2d-30             [-1, 12, 6, 6]               0
           Conv2d-31             [-1, 12, 4, 4]           1,296
             ReLU-32             [-1, 12, 4, 4]               0
      BatchNorm2d-33             [-1, 12, 4, 4]              24
        Dropout2d-34             [-1, 12, 4, 4]               0
        MaxPool2d-35             [-1, 12, 2, 2]               0
AdaptiveAvgPool2d-36             [-1, 12, 1, 1]               0
           Conv2d-37             [-1, 10, 1, 1]             120
================================================================
Total params: 8,088
Trainable params: 8,088
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.74
Params size (MB): 0.03
Estimated Total Size (MB): 0.77
----------------------------------------------------------------
```

### 1.2.2. Result

Accuracy has improved in the last few epochs.  Best accuracy is 99.34% which is close to the target 99.4%..

```text
Epoch: 11
Train phase: Loss=0.03265893831849098 Batch_id=468 Accuracy=98.49: 100%|██████████| 469/469 [00:28<00:00, 16.54it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test phase: Average loss: 0.0228, Accuracy: 9929/10000 (99.29%)

Epoch: 12
Train phase: Loss=0.036436308175325394 Batch_id=468 Accuracy=98.59: 100%|██████████| 469/469 [00:28<00:00, 16.49it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test phase: Average loss: 0.0215, Accuracy: 9934/10000 (99.34%)

Epoch: 13
Train phase: Loss=0.04511019587516785 Batch_id=468 Accuracy=98.73: 100%|██████████| 469/469 [00:28<00:00, 16.53it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test phase: Average loss: 0.0213, Accuracy: 9933/10000 (99.33%)

Epoch: 14
Train phase: Loss=0.031505391001701355 Batch_id=468 Accuracy=98.72: 100%|██████████| 469/469 [00:28<00:00, 16.47it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test phase: Average loss: 0.0214, Accuracy: 9934/10000 (99.34%)
```

### 1.2.3. Analysis

- The accuracy is moving closer to 99.4% due to the use of StepLR, as during later epochs, the LR is reduced to ensure that loss function can find its minima
- Use of StepLR also has helped to reduce the epochs needed to get a higher accuracy for the same reason above.
- In the next step, we'll try more changes to the stepLR to see if it can push up accuracy.. and also 

## 1.3. Iteration 3

[assignment_5_8k_parameters_ver1.ipynb](assignment_5_8k_parameters_ver3.ipynb)

### 1.3.1. Target

We will attempt to get the accuracy to 99.4% and reduce the parameters to be under 8000

- to reduce the parameters under 8000, a 1x1 kernel in transition block #1 is removed.  This makes the parameters to drop below 8000
- The StepLR is further tweaked to see if the weights can converge to push accuracy higher: gamma is changed from 0.09 to 0.2

Model has 7560 parameters..

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             576
              ReLU-6            [-1, 8, 28, 28]               0
       BatchNorm2d-7            [-1, 8, 28, 28]              16
         Dropout2d-8            [-1, 8, 28, 28]               0
            Conv2d-9            [-1, 8, 28, 28]             576
             ReLU-10            [-1, 8, 28, 28]               0
      BatchNorm2d-11            [-1, 8, 28, 28]              16
        Dropout2d-12            [-1, 8, 28, 28]               0
        MaxPool2d-13            [-1, 8, 14, 14]               0
           Conv2d-14           [-1, 12, 12, 12]             864
             ReLU-15           [-1, 12, 12, 12]               0
      BatchNorm2d-16           [-1, 12, 12, 12]              24
        Dropout2d-17           [-1, 12, 12, 12]               0
           Conv2d-18           [-1, 12, 10, 10]           1,296
             ReLU-19           [-1, 12, 10, 10]               0
      BatchNorm2d-20           [-1, 12, 10, 10]              24
        Dropout2d-21           [-1, 12, 10, 10]               0
           Conv2d-22             [-1, 12, 8, 8]           1,296
             ReLU-23             [-1, 12, 8, 8]               0
      BatchNorm2d-24             [-1, 12, 8, 8]              24
        Dropout2d-25             [-1, 12, 8, 8]               0
           Conv2d-26             [-1, 12, 6, 6]           1,296
             ReLU-27             [-1, 12, 6, 6]               0
      BatchNorm2d-28             [-1, 12, 6, 6]              24
        Dropout2d-29             [-1, 12, 6, 6]               0
           Conv2d-30             [-1, 12, 4, 4]           1,296
             ReLU-31             [-1, 12, 4, 4]               0
      BatchNorm2d-32             [-1, 12, 4, 4]              24
        MaxPool2d-33             [-1, 12, 2, 2]               0
AdaptiveAvgPool2d-34             [-1, 12, 1, 1]               0
           Conv2d-35             [-1, 10, 1, 1]             120
================================================================
Total params: 7,560
Trainable params: 7,560
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.72
Params size (MB): 0.03
Estimated Total Size (MB): 0.75
----------------------------------------------------------------
```

### 1.3.2. Result

With the above changes, see that the accuracy stays above 99.4% for the last few epochs.
```text
Epoch: 11
Train phase: Loss=0.015207829885184765 Batch_id=468 Accuracy=98.67: 100%|██████████| 469/469 [00:17<00:00, 26.18it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test phase: Average loss: 0.0198, Accuracy: 9946/10000 (99.46%)

Epoch: 12
Train phase: Loss=0.03666846081614494 Batch_id=468 Accuracy=98.78: 100%|██████████| 469/469 [00:18<00:00, 25.74it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test phase: Average loss: 0.0189, Accuracy: 9944/10000 (99.44%)

Epoch: 13
Train phase: Loss=0.022886531427502632 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:17<00:00, 26.19it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test phase: Average loss: 0.0183, Accuracy: 9944/10000 (99.44%)

Epoch: 14
Train phase: Loss=0.007360551971942186 Batch_id=468 Accuracy=98.93: 100%|██████████| 469/469 [00:17<00:00, 26.20it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test phase: Average loss: 0.0183, Accuracy: 9946/10000 (99.46%)
```

### 1.3.3. Analysis

- The model is able to achieve the target accuracy after the changes above are adopted: 
  - regularization/augmenation (dropout and image rotation), 
  - StepLR, 
  - proper positioning of maxpool layer, 
  - use of BatchNorm to ensure that the input to each hiddern layer is normalized, there by ensuring that the weights do not change drastically during each epoch of training
- The model is underfitting due to the regularization applied: the training accuracy is less than the test accuracy.

