import os
import time
import copy
import torch
import torch.optim as optim
# 引入线性回归常用的损失函数
from torch.nn import Linear, Module, MSELoss
import pprint as pp
import utils.hypergraph_utils as hgut
from models import HGNN
from config import get_config
from datasets import load_feature_construct_H
from datasets import load_ft
import pandas as pd
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')

# initialize data
data_dir = r"L:\HGNN try1\CG_dataset_rgrs.mat"
fts, lbls, idx_train, idx_test = load_ft(data_dir,'CGNN')
# H_dir
H_dir = r"L:\hypergraph prediction\话题用户超图.csv"
W_dir = r"L:\hypergraph prediction\用户超边权值.csv"
H = pd.read_csv(H_dir)
W = pd.read_csv(W_dir)
H = H.to_numpy()
W = W.iloc[:,1].to_numpy()
G = hgut.generate_G_from_H(H,W, variable_weight=False)
'''
fts, lbls, idx_train, idx_test, H = \
    load_feature_construct_H(data_dir,
                             m_prob=cfg['m_prob'],
                             K_neigs=cfg['K_neigs'],
                             is_probH=cfg['is_probH'],
                             use_mvcnn_feature=cfg['use_mvcnn_feature'],
                             use_gvcnn_feature=cfg['use_gvcnn_feature'],
                             use_mvcnn_feature_for_structure=cfg['use_mvcnn_feature_for_structure'],
                             use_gvcnn_feature_for_structure=cfg['use_gvcnn_feature_for_structure'])
'''

# n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 指定存放的位置 cpu/gpu

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    # 分类任务的评估指标acc
#     best_acc = 0.0


    for epoch in range(num_epochs):
        # 间隔输出训练效果
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: # 每一轮都训一遍再测一遍
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test # 选择相应数据集

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G) # model 输出的是n*1
                loss = criterion(outputs[idx], lbls[idx]) # 根据自定义的criterion计算训练误差
                # _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase 在训练阶段进行反向传播和优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            # running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model 存储预测效果最好时候的model
            # deep copy 复制完整对象并且存储在独立的内存地址
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts) # 读出最好的model并且返回
    return model


def _main():
    '''
    print(f"Classification on {cfg['on_dataset']} dataset!!! class number: {n_class}")
    print(f"use MVCNN feature: {cfg['use_mvcnn_feature']}")
    print(f"use GVCNN feature: {cfg['use_gvcnn_feature']}")
    print(f"use MVCNN feature for structure: {cfg['use_mvcnn_feature_for_structure']}")
    print(f"use GVCNN feature for structure: {cfg['use_gvcnn_feature_for_structure']}")
    print('Configuration -> Start')
    pp.pprint(cfg)
    print('Configuration -> End')
    '''

    # 定义模型，此处分类模型被耦合在HGNN函数中，需要去HGNN那里改预测任务
    model_ft = HGNN(in_ch=fts.shape[1], # 特征矩阵列数
                    n_class=1,
                    n_hid=128,
                    dropout=0.5)
    model_ft = model_ft.to(device) # 将模型转移到设备上

    # adam优化
    optimizer = optim.Adam(model_ft.parameters(), lr=0.001,weight_decay=0.0005)
    # SGD优化
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)'])

    # 动态调整学习率
    # milestone 调整轮次
    # gamma 调整倍数
    # new_lr = lr*gamma when epoch = milestone[i]
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100,500],
                                               gamma=0.9)
    # criterion = torch.nn.CrossEntropyLoss() # 分类预测任务中使用了交叉熵损失函数
    # 线性回归要改成
    criterion = torch.nn.MSELoss(reduction = 'mean')

    model_ft = train_model(model_ft, criterion, optimizer, schedular, 1000, print_freq=100)


if __name__ == '__main__':
    _main()
