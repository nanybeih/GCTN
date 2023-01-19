import torch.nn as nn
import torch
class GraphConvolution(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(GraphConvolution,self).__init__()
        self.kernel_size = kernel_size


    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous()