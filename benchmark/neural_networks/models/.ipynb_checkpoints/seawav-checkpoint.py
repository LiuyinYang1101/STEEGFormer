import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# dilated conv layer with kaiming_normal initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding, stride=stride)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out

# conv1x1 layer with zero initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py but the scale parameter is removed
class KaimingConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(KaimingConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out
    
# conv1x1 layer with zero initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py but the scale parameter is removed
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


# every residual block (named residual layer in paper)
# contains one noncausal dilated conv
class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        # dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)

        # residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        self.dropoutLayer = nn.Dropout(p=0.5)
        nn.init.kaiming_normal_(self.res_conv.weight)

    def forward(self, input_data):

        x = input_data
        h = x
        B, C, L = x.shape
        #print("B",B)
        #print("C",C)
        #print("L",L)
        assert C == self.res_channels

        # dilated conv layer
        h = self.dilated_conv_layer(h)
        
        # gated-tanh nonlinearity
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])
        out = self.dropoutLayer(out)
        # residual and skip outputs
        res = self.res_conv(out)
        assert x.shape == res.shape
        
        return (x + res) * math.sqrt(0.5), res  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.dilation_cycle = dilation_cycle
     
        # stack all residual blocks with dilations 1, 2, ... , 512, ... , 1, 2, ..., 512
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       dilation=2 ** (n % dilation_cycle)))

    def forward(self, input_data):
        x = input_data

        # pass all residual layers
        h = x
        skip = 0
        for n in range(self.num_res_layers):
            if n>0 and n % self.dilation_cycle == 0:
                #add original skip data
                #print(n," here add skip connection")
                h = (h+x) * math.sqrt(1.0 / 2)  # normalize for training stability
                h, skip_n = self.residual_blocks[n](h)  # use the output from last residual layer
            else:    
                h, skip_n = self.residual_blocks[n](h)  # use the output from last residual layer
            
            skip = skip + skip_n  # accumulate all skip outputs

        return skip * math.sqrt(1.0 / self.num_res_layers)  # normalize for training stability


class WaveNet_regressor(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers, dilation_cycle):
        super(WaveNet_regressor, self).__init__()
        self.dilationPerLayer = []
        for n in range(num_res_layers):
            self.dilationPerLayer.append((n,2 ** (n % dilation_cycle)))
        print("This feature extractor has a receptive field of: ", self.calRp(num_res_layers-1,self.dilationPerLayer[num_res_layers-1][1],3))
        
        # initial conv1x1 with relu
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.GELU())
        
        # all residual layers
        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             dilation_cycle=dilation_cycle)
        
        # final conv1x1 -> relu -> zeroconv1x1
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1, stride=4),
                                        nn.GELU(),
                                        nn.Conv1d(skip_channels, out_channels, kernel_size=1, stride=2))
    def calRp(self, layer, dilation, kernel_size):
        if layer == 0:
            return self.calcRpSingleLayer(dilation,kernel_size)
        else:
            return self.calRp(layer-1, self.dilationPerLayer[layer-1][1], kernel_size)+self.calcRpSingleLayer(dilation,kernel_size)-1
    
    def calcRpSingleLayer(self, dilation, kernel_size):
        return kernel_size + (kernel_size-1)*(dilation-1)
    
    def forward(self, input_data):
        input_data = torch.squeeze(input_data,2)
        x = input_data

        x = self.init_conv(x)
        x = self.residual_layer(x)
        x = self.final_conv(x)
        return x
