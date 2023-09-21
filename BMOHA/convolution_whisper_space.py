"""This is a module to ensemble a convolution (depthwise) encoder with or without residule connection.

Authors
 * Jianyuan Zhong 2020
"""
import torch
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import LayerNorm
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet1DPonderate(nn.Module):
    def __init__(self, n_blocks, channels,input_channels, num_classes=1,attention=False,attn_dim=30,nbCoef=5,reduced_dim = None):
        super(ResNet1DPonderate, self).__init__()
        assert n_blocks == len(channels), "The number of blocks should match the number of channels."

        self.in_channels = input_channels
        self.layers = self.make_layers(n_blocks, channels)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.ponderators = torch.nn.Parameter(torch.rand([1,nbCoef,1,1], requires_grad=True))
        
    def make_layers(self, n_blocks, channels):
        layers = []
        for i in range(n_blocks):
            stride = 3 if i > 0 else 2
            layers.append(self.make_layer(channels[i], stride))
        return nn.Sequential(*layers)

    def make_layer(self, out_channels, stride=2):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x,sig_lens,device):
            
        xp = (x * self.ponderators).sum(dim=1)
        
        out = self.layers(xp)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

