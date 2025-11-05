"""
EEGNet: 

EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces

"""

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



"""
CTNet: A Convolution-Transformer Network for EEG-Based Motor Imagery Classification

author: zhaowei701@163.com

Cite this work
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7

"""
import math
import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=22, emb_size=40):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x
    
########################################################################################
# The Transformer code is based on this github project and has been fine-tuned: 
#    https://github.com/eeyhsong/EEG-Conformer
########################################################################################
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    


# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            
            )    
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])




class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=22,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
        )


# learnable positional embedding module        
class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=2048, dropout=0.1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        if self.device=='cuda':
            x = x + self.encoding[:, :x.shape[1], :].cuda()
        else:
            x = x + self.encoding[:, :x.shape[1], :]
        return self.dropout(x)        
        
   
        
# CTNet       
class EEGTransformer(nn.Module):
    def __init__(self, heads=4, 
                 emb_size=40,
                 depth=6, 
                 number_class=4,
                 number_channel = 64,
                 data_length = 128,
                 sampling_rate = 128,
                 eeg1_f1 = 20,
                 eeg1_D = 2,
                 eeg1_dropout_rate = 0.3,
                 **kwargs):
        super().__init__()
        if sampling_rate==128:
            eeg1_kernel_size = 32 #change from 64 to 32 for 128Hz EEG data
            eeg1_pooling_size1 = 4 #change from 8 to 4 for 128HZ EEG data
            eeg1_pooling_size2 = 4 #change from 8 to 4 for 128HZ EEG data
        elif sampling_rate==256:
            eeg1_kernel_size = 64
            eeg1_pooling_size1 = 8
            eeg1_pooling_size2 = 8
        else:
            fs_ratio = sampling_rate / 128.0
            eeg1_kernel_size = max(1, int(round(32  * fs_ratio)))
            eeg1_pooling_size1 = max(1, int(round(4  * fs_ratio)))
            eeg1_pooling_size2 = max(1, int(round(4  * fs_ratio)))
        self.number_class = number_class
        self.number_channel = number_channel
        self.emb_size = emb_size
        self.flatten = nn.Flatten()
        # print('self.number_channel', self.number_channel)
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                              f1 = eeg1_f1,
                                              kernel_size = eeg1_kernel_size,
                                              D = eeg1_D,
                                              pooling_size1 = eeg1_pooling_size1,
                                              pooling_size2 = eeg1_pooling_size2,
                                              dropout_rate = eeg1_dropout_rate,
                                              )
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.eval()
        with torch.no_grad():
            test_data = torch.randn(6,1,number_channel,data_length)
            cnn = self.cnn(test_data)
            #  positional embedding
            cnn = cnn * math.sqrt(self.emb_size)
            cnn = self.position(cnn)

            trans = self.trans(cnn)
            # residual connect
            features = cnn+trans
            
            features = self.flatten(features)
            self.num_features = features.shape[-1]
            #print("allow:" ,test_data.shape, self.num_features)
        self.classification = ClassificationHead(self.num_features , self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module
    def forward(self, x):
        #print("input:", x.shape)
        x = torch.unsqueeze(x,1)
        #print("input2:", x.shape)
        cnn = self.cnn(x)
        #print("cnn:", cnn.shape)
        #  positional embedding
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        
        trans = self.trans(cnn)
        # residual connect
        features = cnn+trans
        
        out = self.classification(self.flatten(features))
        return out
