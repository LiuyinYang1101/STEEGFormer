import torch.nn as nn
import torch
import torch.nn.functional as F
import math
# implement https://github.com/osmanberke/Deep-SSVEP-BCI in pytorch

class SSVEPDNN(nn.Module):
    def __init__(self, no_fb, no_channels, no_combined_channels, drop_out_ratio_1, drop_out_ratio_2, input_length, num_class):
        super(SSVEPDNN, self).__init__()
        self.no_fb = no_fb
        self.no_channels = no_channels
        self.no_combined_channels = no_combined_channels
        self.drop_out_ratio1 = drop_out_ratio_1
        self.drop_out_ratio2 = drop_out_ratio_2
        self.input_length = input_length
        self.num_class = num_class
        # Init conv: fb conv
        self.init_conv = nn.Conv1d(1,no_fb,33, padding='same')
        # Layer 1: subband recombination (batch, fb, ch, t) -> (batch, 1, ch, t)
        self.conv1 = nn.Conv2d(self.no_fb, 1, (1,1), bias = False)
        # Initialize the weights as ones
        self.conv1.weight.data.fill_(1.0)
        
        # Layer 2: spatial filter (batch, 1, ch, t) -> (batch, 1, combined_ch, t)
        self.conv2 = nn.Conv1d(self.no_channels, self.no_combined_channels, 1, bias = True)
        self.dropout1 = nn.Dropout(self.drop_out_ratio1)
        self.relu = nn.ReLU()
        
        # Layer 3: downsample layer (batch, combined_ch, t) -> (batch, combined_ch, t/2) full dimension across channels
        self.conv3 = nn.Conv1d(self.no_combined_channels, self.no_combined_channels, 2, stride=2,
                               groups = 1, bias = True)  

        # Layer 4: temporal filter layer (batch, combined_ch, t/2) -> (batch, combined_ch, t/2) full dimension across channels
        self.conv4 = nn.Conv1d(self.no_combined_channels, self.no_combined_channels, 10, stride=1,
                               groups = 1, bias = True, padding='same') 
        self.dropout2 = nn.Dropout(self.drop_out_ratio2)
        # FC Layer
        self.fc1 = nn.Linear(self.no_combined_channels * (input_length//2), self.num_class)
        
    def set_drop_out(self,new_dropout1, new_dropout2):
        self.dropout1.rate = new_dropout1
        self.dropout2.rate = new_dropout2
            
    def forward(self, x):
        # Layer 0: fb filter
        B,Ch,t = x.shape
        # Transpose to make 'channel' as batch dimension and combine actual batch and channel dimensions
        input_data_reshaped = x.view(B * Ch, 1, t)
        #print("input:", input_data_reshaped.shape)
        output = self.init_conv(input_data_reshaped)
        output = output.view(B,self.no_fb, Ch,-1)
        # Layer 1: subband 
        #print("input",x.shape)
        x = self.conv1(output)
        #print("conv1",x.shape)
        
        # Layer 2: spatial filter
        x = self.conv2(torch.squeeze(x,dim=1))
        x = self.dropout1(x)   
        #print("cov2",x.shape)
        
        # Layer 3: downsample
        x = self.conv3(x)
        x = self.dropout1(x)
        x = self.relu(x)
        #print("cov3",x.shape)
        
        # Layer 4: temporal filter
        x = self.conv4(x)
        x = self.dropout2(x)
        #print("cov4",x.shape)
        x = torch.flatten(x, start_dim=1)
        #print("fc",x.shape)

        # Layer 5: fc
        x = self.fc1(x)
        return x