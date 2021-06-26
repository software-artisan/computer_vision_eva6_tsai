#print(f"{__file__} imported")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from collections import OrderedDict

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

class Net(nn.Module):
    def __init__(self, norm_layer_type, num_groups_for_group_norm=None):
        """
        norm_layer_type: 'batch' | 'group' | 'layer'
        """
        super(Net, self).__init__()
        
        dropout=0.05
 
        ####### 
        # Convolution Block #1
        #########
        self.conv1_1_3_3_8_p = nn.Sequential(OrderedDict([
            ('conv1_1_3_3_8_p', nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)),
            ('relu', nn.ReLU()),
            ('batchNorm2d', get_norm_layer(norm_layer_type, num_channels=8, num_groups_for_group_norm=num_groups_for_group_norm) ),
            ('dropOut2d', nn.Dropout2d(p=dropout))
          ])
        ) # Input=28, Output=28, rf=3

        self.conv2_8_3_3_8_p = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_layer_type, num_channels=8, num_groups_for_group_norm=num_groups_for_group_norm),
            nn.Dropout2d(p=dropout)
        ) # Input=28, Output=28, rf=5
 
        self.conv3_8_3_3_8_p = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_layer_type, num_channels=8, num_groups_for_group_norm=num_groups_for_group_norm),
            nn.Dropout2d(p=dropout) 
        ) # Input=28, Output=28, rf=10
 
        ####### 
        # Transition Block #1
        #########
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Input=28, Output=14, rf=6
 
        self.conv4_8_1_1_12 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=1, padding=0, bias=False),
        ) # Input=14, Output=14, rf=32
 
        ####### 
        # Convolution Block #2
        #########
        self.conv5_12_3_3_11 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_layer_type, num_channels=12, num_groups_for_group_norm=num_groups_for_group_norm),
            nn.Dropout2d(p=dropout)
        ) # Input=14, Output=14, rf=14
 
        self.conv6_11_3_3_12 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_layer_type, num_channels=12, num_groups_for_group_norm=num_groups_for_group_norm),
            nn.Dropout2d(p=dropout) 
        ) # Input=14, Output=14, rf=24
        
        self.conv7_12_3_3_12 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_layer_type, num_channels=12, num_groups_for_group_norm=num_groups_for_group_norm),
            nn.Dropout2d(p=dropout) 
        )# Input=14, Output=14, rf=24
        
        self.conv8_12_3_3_12 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_layer_type, num_channels=12, num_groups_for_group_norm=num_groups_for_group_norm),
            nn.Dropout2d(p=dropout) 
        ) # Input=14, Output=14, rf=24

        self.conv9_12_3_3_12 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_layer_type, num_channels=10, num_groups_for_group_norm=num_groups_for_group_norm),
            #nn.Dropout2d(p=dropout) 
        ) # Input=14, Output=14, rf=24
        
        #######
        # Transition block #2
        #######
        self.maxpool2= nn.MaxPool2d(kernel_size=2, stride=2) # Input=6, Output=3, chan=12, 
 
        ####### 
        # Output Block
        #########
        # global average pool before 1x1 to reduce computation
        #self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=1)  # Input=3, Output=1, chan=12,
        self.global_avgpool = nn.AvgPool2d(kernel_size=10)  # Input=3, Output=1, chan=12,
 
        self.conv10_16_1_1_10 = nn.Sequential(
            #nn.Conv2d(in_channels=14, out_channels=10, kernel_size=3, padding=0, bias=False),
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=1, padding=0, bias=False),
        ) # Input=1, Output=1, chan=10, 
    
    def forward(self, x):
        #####
        # conv block #1
        ########
        x = self.conv1_1_3_3_8_p(x)
        x = self.conv2_8_3_3_8_p(x)
        x = self.conv3_8_3_3_8_p(x)
 
        #####
        # Transitioni block #1
        ########
        x = self.maxpool1(x)
        x = self.conv4_8_1_1_12(x)
 
        #####
        # conv block #2
        ########
        x = self.conv5_12_3_3_11(x)
        x = self.conv6_11_3_3_12(x)
        x = self.conv7_12_3_3_12(x)
        x = self.conv8_12_3_3_12(x)
        x = self.conv9_12_3_3_12(x)

        #######
        # Transition block #2
        #######
        #x = self.maxpool2(x)
 
        #####
        # output block
        ########
        x = self.global_avgpool(x)        
        #x = self.conv10_16_1_1_10(x)
               
        x = x.view(-1, 10)
        return F.log_softmax(x)