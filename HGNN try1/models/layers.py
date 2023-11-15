import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# 只完成卷积，不包括激活函数
class HGNN_conv(nn.Module):  # 对基类nn.Module的继承
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()  # super() 创建一个nn.Module的实例

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))  # 是每个卷积层的全连接层Θ

        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))  # 加上偏置
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor): # x(l),G
        x = x.matmul(self.weight) # x(l)*Θ
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x) # x(l+1) =(DV-1/2*H*W*DE-1*HT*DV-1/2)*x(l)*Θ
        return x

# 全连接一下
class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

# 两个卷积层，还包括了激活函数
# 可以用作输出特征时候的选项
# 也可以直接把预测工作集合在模型里，见HGNN定义
class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)  # 第一层卷积
        self.hgc2 = HGNN_conv(n_hid, n_hid)  # 第二层卷积

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G)) #relu是一个卷积层的激活函数
        #x(2) = δ（G*x(1)*Θ）
        x = F.dropout(x, self.dropout) #dropout失活率
        x = F.relu(self.hgc2(x, G)) # 第二层也是relu激活函数
        return x


# 分类实例
class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()  # 创建一个基类的实例
        self.fc1 = nn.Linear(n_hid, n_class)  # 实例具体定义
        # full connected全连接层线性变换，指定输入维度和输出维度

    def forward(self, x):  # 用forward定义数据在网络中的流向
        # x是输入数据
        x = self.fc1(x)  # 用x在网络中各层按顺序计算，从第l层的输出进入l+1层的输入，最终得到网络输出
        return x

