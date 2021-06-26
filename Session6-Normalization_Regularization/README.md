- [1. What is your code all about, (what the notebook does; what model.py is doing)](#1-what-is-your-code-all-about-what-the-notebook-does-what-modelpy-is-doing)
- [2. How to perform the 3 covered normalization (cannot use values from the excel sheet shared).. take the same excel sheet.. and change the values.. actually random values are used.. so if you close i and open it, new values will show up](#2-how-to-perform-the-3-covered-normalization-cannot-use-values-from-the-excel-sheet-shared-take-the-same-excel-sheet-and-change-the-values-actually-random-values-are-used-so-if-you-close-i-and-open-it-new-values-will-show-up)
  - [2.1. Normalization types](#21-normalization-types)
  - [2.2. Formula](#22-formula)
  - [2.3. Implementation](#23-implementation)
- [3. Show all 3 calculations for 4 sample 2x2 images (image shown in the content has 3 images) so batch size is 4 and not 3](#3-show-all-3-calculations-for-4-sample-2x2-images-image-shown-in-the-content-has-3-images-so-batch-size-is-4-and-not-3)
  - [3.1. Combined Normalizations](#31-combined-normalizations)
  - [3.2. Batch Normalization](#32-batch-normalization)
  - [3.3. Layer Normalization](#33-layer-normalization)
  - [3.4. Group Normalization](#34-group-normalization)
- [4. Your findings for normalization techniques, (LN, BN, GN).. how they are helping you or constraining you](#4-your-findings-for-normalization-techniques-ln-bn-gn-how-they-are-helping-you-or-constraining-you)
- [5. Add all your graphs](#5-add-all-your-graphs)
- [6. Your 3 collection-of-misclassified-images](#6-your-3-collection-of-misclassified-images)
  - [6.1. Incorrect image predictions for NN with group normalization and layer 1 regularization](#61-incorrect-image-predictions-for-nn-with-group-normalization-and-layer-1-regularization)
  - [6.2. Incorrect image predictions for NN with layer normalization and layer 2 regularization](#62-incorrect-image-predictions-for-nn-with-layer-normalization-and-layer-2-regularization)
  - [6.3. Incorrect image predictions for NN with batch normalization, layer 1 and layer 2 regularization](#63-incorrect-image-predictions-for-nn-with-batch-normalization-layer-1-and-layer-2-regularization)

## 1. What is your code all about, (what the notebook does; what model.py is doing)

The code is moduralized as described below:

- The model is defined in a separate python module: [assignment_6_model.py](assignment_6_model.py)
  - The `norm_layer_type` argument in the constructor defines the type of normalization that the model will perform

  ```python
        def __init__(self, norm_layer_type, num_groups_for_group_norm=None):
            """
            norm_layer_type: 'batch' | 'group' | 'layer'
            """
  ```

- The notebook defines the rest of the logic including the code for loading, training, testing and plotting: [assignment_6_batch_norm_regularization.ipynb](assignment_6_batch_norm_regularization.ipynb)

## 2. How to perform the 3 covered normalization (cannot use values from the excel sheet shared).. take the same excel sheet.. and change the values.. actually random values are used.. so if you close i and open it, new values will show up

### 2.1. Normalization types

Batch Normalization

- Normalize each channel, in each layer (of NN), across all images

Layer Normalization

- Normalize all channels, in each layer (of NN), for each image

Group Normalization

- Normalize a group of channels, in each layer (of NN), for each image

### 2.2. Formula

![picture 1](images/a8818051fa52606b6c92102d140d8be099807bda7da1d22c5296d82a3767db3a.png)  

### 2.3. Implementation

Shows how the normalization tyeps are implemented.  For details see [assignment_6_model.py](assignment_6_model.py)

```python
def get_norm_layer( norm_layer_type, num_channels, num_groups_for_group_norm=None):
    """
    norm_layer_type: 'batch' | 'group' | 'layer'
    num_channels: # of channels
    """
    if norm_layer_type == "batch":
        # Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>__ .        #
        # The mean and standard-deviation are calculated per-dimension over the mini-batches and \gamma and \beta are learnable parameter vectors of size C (where C is the input size). By default, the elements of \gamma are set to 1 and the elements of \beta are set to 0. The standard-deviation is calculated via the biased estimator, equivalent to torch.var(input, unbiased=False).        #
        # def __init__(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        nl = nn.BatchNorm2d(num_features=num_channels)
    elif norm_layer_type == "group":
        # Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization
        # The input channels are separated into num_groups groups, each containing num_channels / num_groups channels. The mean and standard-deviation are calculated separately over the each group. \gammaγ and \betaβ are learnable per-channel affine transform parameter vectors of size num_channels if affine is True. The standard-deviation is calculated via the biased estimator, equivalent to torch.var(input, unbiased=False).
        nl = nn.GroupNorm(num_groups=num_groups_for_group_norm, num_channels=num_channels)
    elif norm_layer_type == "layer":
        # a group size of '1' uses all the 'features'/channels of the image: essentially a 'layer norm'
        nl = nn.GroupNorm(num_groups=1, num_channels=num_channels)

    return nl
```

L1 Regularization is implemented as shown below: see [assignment_6_batch_norm_regularization.ipynb](assignment_6_batch_norm_regularization.ipynb) for further details

```python
    # if l1 regularization applies, do it..
    if lambda_l1:
      l1_loss = 0
      for p in model.parameters():
        l1_loss = l1_loss + p.abs().sum()
      loss = loss + lambda_l1 * l1_loss    # add l1 loss..
```

L2 Regularization is implemented as shown below:  `weight_decay` specified when the optimizer is instantiated specifies the lambda for L2 Regularization.  See [assignment_6_batch_norm_regularization.ipynb](assignment_6_batch_norm_regularization.ipynb) for further details

```python
def run_train_and_test(num_epochs, norm_type, num_groups_for_group_norm=None, lambda_l1=None, lambda_l2=None, num_batches=None) -> TrainTestLossAccuracy: 
  model =  assignment_6_model.Net(norm_layer_type=norm_type, num_groups_for_group_norm=num_groups_for_group_norm).to(device)
  
  if lambda_l2:
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=lambda_l2)
  else:
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

```

## 3. Show all 3 calculations for 4 sample 2x2 images (image shown in the content has 3 images) so batch size is 4 and not 3

### 3.1. Combined Normalizations
![Combined_Normalizations](images/Combined_Normalizations.PNG)

### 3.2. Batch Normalization
![BatchNorm](images/BatchNorm.PNG)

### 3.3. Layer Normalization
![LayerNorm](images/LayerNorm.PNG)

### 3.4. Group Normalization
![GroupNorm](images/GroupNorm.PNG)

## 4. Your findings for normalization techniques, (LN, BN, GN).. how they are helping you or constraining you

- We observed that that among the 3 types of normalizations, Batch Normalization had the highest test accuracy and lowest training loss.  So compared to the other 2, it performed the best.
- The train loss for batch normalization was higher compared to the other two.  This could be due to the fact that both L1 and L2 regularziations were applied, which make the training harder (this is the intent of regularization which is to avoid overfitting).  As a result, the training losses were high.

## 5. Add all your graphs

from [assignment_6_batch_norm_regularization.ipynb](assignment_6_batch_norm_regularization.ipynb)

<!-- ![picture 2](images/f93f1d9a931d68481aca42290300eebbaf4c2725f209e437619c339a4936e273.png)  
![picture 3](images/f93f1d9a931d68481aca42290300eebbaf4c2725f209e437619c339a4936e273.png)   -->

**Train Loss**

![picture 4](images/f28ff7a6fab01a096dbfee2db79fe1cd0b9306a356049c495fdce0075db96f50.png)  

**Test Loss**

![picture 5](images/82199845c59cd5764343cffa3e7730a65458720ab0b9ea785356070f41714842.png)  

**Train Accuracy**

![picture 6](images/eb09756119a8883623c3d3b74c77e0feeba241eca2698254ecd4515191268b62.png)  

**Test Accuracy**

![picture 7](images/25ba1fff1a81cec2559ed0cce6cf37083b93a538563ad40ecbb2df5d462b1eec.png)  

## 6. Your 3 collection-of-misclassified-images

from [assignment_6_batch_norm_regularization.ipynb](assignment_6_batch_norm_regularization.ipynb)

### 6.1. Incorrect image predictions for NN with group normalization and layer 1 regularization

<!-- ![picture 8](images/3ef3596aaa31d732ad04cbbeee4e85b44ac7c2eb00d5f60eb3ab704d021f76f4.png)  -->

![picture 11](images/3c0e9b800916490a6b0e8243c9bded7e43f4a20956b48b6bbfbc85ea5c659dac.png)  

### 6.2. Incorrect image predictions for NN with layer normalization and layer 2 regularization

<!-- ![picture 9](images/551a2db72197ba216802fd8541d915402d86a678346a3b98e0c1e3232d1f1802.png)   -->
![picture 12](images/e1b9dc28451dd6199772d700e4aa41ea20d24e97cdc6afa3ced35df1000f90b1.png)  

### 6.3. Incorrect image predictions for NN with batch normalization, layer 1 and layer 2 regularization

<!-- ![picture 10](images/a56f88936a6695761ebdacc816e7f2c7802409e560d5155a38c54db8fe6dcd9b.png)   -->
![picture 13](images/cbad068ff6cb595c95ab2aceaf7bea01ddc2f82da9560830a16b31906ce58f88.png)  
