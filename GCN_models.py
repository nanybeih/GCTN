import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 training = True,
                 bias=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size,
                 t_stride,
                 t_padding,
                 t_dilation,
                 bias)
        self.gc2 = GraphConvolution(out_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size,
                 t_stride,
                 t_padding,
                 t_dilation,
                 bias)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.training = training
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.dropout(self.norm1(x.permute(0, 3, 2, 1)))
        x = self.conv(x.permute(0, 3, 2, 1))
        x = self.gc2(x, adj)
        return F.log_softmax(self.norm2(x.permute(0, 3, 2, 1)), dim=1)
