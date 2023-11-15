from torch import nn
from models import HGNN_conv,HGNN_fc # 超图卷积层
import torch.nn.functional as F # 提供了一堆神经网络的激活函数和功能函数


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid) # 定义两个具体的卷积层，分别命名为hgc1和hgc2
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.rgrs = HGNN_fc(n_hid, n_class)
        

    def forward(self, x, G): # 定义数据流动的方向，输入数据两个，x和G，是根据HGNN_conv确定的
        x = F.relu(self.hgc1(x, G)) # HGNN_conv(in_ch, n_hid)
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G)) # HGNN_conv(n_hid, n_class)
        x = self.rgrs(x)
        
        return x # n*1
