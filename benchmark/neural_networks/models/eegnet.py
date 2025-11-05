# down sample 125Hz
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias,padding='same')
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
       # print("sepcov2d-depth",out.shape)
        out = self.pointwise(out)
       # print("sepcov2d-point",out.shape)
        return out
    
class MaxNormLinear(nn.Module):
    
    def __init__(self,inchannel,outchannel):
        
        super(MaxNormLinear, self).__init__()
        self.linear = nn.Linear(inchannel, outchannel)
        self._eps = 1e-7
        
    def max_norm(self):
        with torch.no_grad():
            norm = self.linear.weight.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, 0.25)
            self.linear.weight = torch.nn.Parameter( (self.linear.weight * (desired / (self._eps + norm))), dtype=torch.float, device="cuda")
    
    def forward(self,x):
        self.max_norm()
        return self.linear(x)
        
class ConstrainedConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, no_groups, if_bias):      
        super(ConstrainedConv2d, self).__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               groups=no_groups, bias=if_bias)  # spatial filter 22channel=>3channel
    def max_norm(self):
       
        norm = self.conv2.weight.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, 1.0)
        self.conv2.weight = torch.nn.Parameter( (self.conv2.weight * (desired / (1e10-7 + norm)))).to("cuda")
    
    def forward(self, input):
        self.max_norm()
        return self.conv2(input)  
    
class EEGNet(nn.Module):
    def __init__(self, no_spatial_filters, no_channels, no_temporal_filters, temporal_length_1, temporal_length_2, window_length, num_class, drop_out_ratio=0.50, pooling2=4, pooling3=8):
        super(EEGNet, self).__init__()
        self.drop_out_ratio = drop_out_ratio
        
        # Layer 1: band pass filter
        self.conv1 = nn.Conv2d(1, no_temporal_filters, (1, temporal_length_1), padding='same', bias=False)

        self.batchnorm1 = nn.BatchNorm2d(no_temporal_filters, False)
        self.dropout = nn.Dropout(self.drop_out_ratio)
        
        # Layer 2: channel-aware spatial filter
        self.conv2 = nn.Conv2d(no_temporal_filters, no_temporal_filters * no_spatial_filters, (no_channels, 1),
                               groups = no_temporal_filters, bias = False)  # spatial filter 
        self.batchnorm2 = nn.BatchNorm2d(no_temporal_filters * no_spatial_filters, False)
        self.pooling2 = nn.AvgPool2d(1, pooling2) # from fs->32 Hz
        
        # Layer 3
        self.separableConv2 = SeparableConv2d(no_temporal_filters * no_spatial_filters,
                                              no_temporal_filters * no_spatial_filters, (1, temporal_length_2))
        self.batchnorm3 = nn.BatchNorm2d(no_temporal_filters * no_spatial_filters, False)

        self.pooling3 = nn.AvgPool2d((1, pooling3)) 
        
        eeg_random = torch.randn(4,no_channels,window_length)
        fc_length = self.calc_fc_features(eeg_random)
        self.fc1 = nn.Linear(fc_length, num_class)
        
    
    def calc_fc_features(self,x):
        self.eval()
        with torch.no_grad():
            x = torch.unsqueeze(x,1)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.conv2(x)
            B,FB,Ch,TL = x.shape
            x= torch.reshape(x,(B,FB*Ch,1,TL))
            x = nn.functional.elu(self.batchnorm2(x))
            x = self.pooling2(x)
            x = self.dropout(x)
            x = self.separableConv2.forward(x)
            x = nn.functional.elu(self.batchnorm3(x))
            x = self.pooling3(x)
            x = self.dropout(x)
            x = torch.flatten(x, start_dim=1)
            return x.shape[-1]
    
    def set_drop_out(self,new_dropout):
        self.dropout.rate = new_dropout
    
            
    def forward(self, x):
        # Layer 1
        #print("input",x.shape)
        x = torch.unsqueeze(x,1)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.batchnorm1(x)
        # Layer 2
        #print("cov1",x.shape)
        x = self.conv2(x)
        #print("conv2", x.shape)
        B,FB,Ch,TL = x.shape
        x= torch.reshape(x,(B,FB*Ch,1,TL))
        x = nn.functional.elu(self.batchnorm2(x))
        x = self.pooling2(x)
        x = self.dropout(x)

        # Layer 3
        #print("pooling2",x.shape)
        x = self.separableConv2.forward(x)
        #print("cov3",x.shape)

        x = nn.functional.elu(self.batchnorm3(x))
        x = self.pooling3(x)
        #print("pooling3",x.shape)
        x = self.dropout(x)
        #print("before flatten",x.shape)
        x = torch.flatten(x, start_dim=1)
        #print("fc",x.shape)
        x = self.fc1(x)
        return x