# %%
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
import math
import torch.nn.functional as F

# %%
# initialize data
data_dir = r"CG_dataset_rgrs.mat"
fts, lbls, idx_train, idx_test = load_ft(data_dir, 'CGNN')

# %%
print(fts, lbls, idx_train, idx_test)

# %%
# H_dir
H_dir = r"话题用户超图.csv"
W_dir = r"用户超边权值.csv"
H = pd.read_csv(H_dir, header=None)
W = pd.read_csv(W_dir)
H = H.to_numpy()
W = W.iloc[:, 1].to_numpy()

# %%
G = hgut.generate_G_from_H(H, W, variable_weight=False)

# %%
G = torch.Tensor(G)

# %%
G = torch.where(torch.isnan(G), torch.full_like(G, 0), G)

# %%
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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 指定存放的位置 cpu/gpu

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)
reg_l2 = 0.00000001
reg_cross_entropy = 0.5

# %%
def MRSE(outputs, lbls):
    if len(outputs.shape) > 1:
        outputs = outputs.squeeze()
    if len(lbls.shape) > 1:
        lbls = lbls.squeeze()
    squared_relative_error = ((outputs - lbls) / lbls) ** 2
    mrse = torch.mean(squared_relative_error)
    return mrse

def criterion(pred, true):
    if len(pred.shape) > 1:
        pred = pred.squeeze()
    if len(true.shape) > 1:
        true = true.squeeze()
    loss = MRSE(pred, true)
    ## 正则化
    # for param in model.parameters():
    #     norm = torch.norm(param, p=2)  # 计算 L2 范数
    #     loss += reg_l2 * norm
    # s_out = torch.softmax(pred.float(), dim=0)
    # s_lbls = torch.softmax(true.float(), dim=0)
    # loss += reg_cross_entropy * F.binary_cross_entropy(s_out, s_lbls, reduction='mean')
    return loss


# %%
def train_model(model, MAE, MSE, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    # 分类任务的评估指标acc
    #     best_acc = 0.0
    best_loss = 9999
    best_mrse = 9999
    best_mae = 9999
    best_mse = 9999
    best_rmse = 9999

    patience = 1000
    for epoch in range(num_epochs):
        # 间隔输出训练效果
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # 每一轮都训一遍再测一遍
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test  # 选择相应数据集

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)  # model 输出的是n*1
                #                 print(outputs[idx].float())
                # loss = criterion(outputs[idx].float(), lbls[idx].float())  # 根据自定义的criterion计算训练误差

                loss = criterion(outputs[idx], lbls[idx])
                # # 打印loss到两个txt文件
                # with open('.'.join([phase, 'txt']), 'a+') as file:
                #     file.write(str(loss.item())+'\n')

                # _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase 在训练阶段进行反向传播和优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

            # statistics
            # running_loss += loss.item() * fts.size(0)
            #             print(fts.size(0))
            # running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            # epoch_loss = running_loss / len(idx)
            #             epoch_acc = running_corrects.double() / len(idx)

            # if epoch % print_freq == 0:
            #     print(f'{phase} Loss: {epoch_loss:.4f}')
            if epoch % print_freq == 0:
                print(f'{phase} Loss: {loss:.4f}')

            # deep copy the model 存储预测效果最好时候的model
            # deep copy 复制完整对象并且存储在独立的内存地址
            # 在评估阶段选择最好的模型，loss最小
            if phase == 'val':
                if loss < best_loss:
                    best_loss = loss
                    #                 best_mrse = mrse
                    best_model_wts = copy.deepcopy(model.state_dict())

                    best_mae = MAE(outputs[idx].float(), lbls[idx].float())
                    best_mrse = MRSE(outputs[idx], lbls[idx]).item()
                    best_mse = MSE(outputs[idx], lbls[idx])
                    best_rmse = math.sqrt(best_mse)
            #                 print(f'mrse: {mrse}')
            #                 print(f'mae: {mae}')
            #                 print(f'rmse: {rmse}')
                    patience = 1000
                else:
                     patience -= 1

        if patience < 0:
            break
        if epoch % print_freq == 0:
            print(f'Best val loss: {best_loss:4f}')
            print(f'mrse: {best_mrse}')
            print(f'mae: {best_mae}')
            print(f'rmse: {best_rmse}')
            print(f'patience: {patience}')

            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')
    print(f'mrse: {best_mrse}')
    print(f'mae: {best_mae}')
    print(f'rmse: {best_rmse}')

    # load best model weights
    model.load_state_dict(best_model_wts)  # 读出最好的model并且返回
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
    model_ft = HGNN(in_ch=fts.shape[1],  # 特征矩阵列数
                    n_class=1,
                    n_hid=128,
                    dropout=0)
    model_ft = model_ft.to(device)  # 将模型转移到设备上

    # adam优化
    optimizer = optim.Adam(model_ft.parameters(), lr=0.01, weight_decay=0.005)
    # SGD优化
    #     optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)'])

    # 动态调整学习率
    # milestone 调整轮次
    # gamma 调整倍数
    # new_lr = lr*gamma when epoch = milestone[i]
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100, 300, 500, 800],
                                               gamma=0.9)
    # criterion = torch.nn.CrossEntropyLoss() # 分类预测任务中使用了交叉熵损失函数
    # 线性回归要改成


    MSE = torch.nn.MSELoss(reduction='mean')
    MAE = torch.nn.L1Loss(reduction='mean')

    model_ft = train_model(model_ft, MAE, MSE, optimizer, schedular, 20001, print_freq=100)


if __name__ == '__main__':
    _main()





