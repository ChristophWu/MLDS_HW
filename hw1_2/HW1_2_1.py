#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:30:35 2018

@author: jason
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


#torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False


# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).data.type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = test_x.view(-1,28*28)
test_y = test_data.test_labels[:2000]

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, depth, n_feature, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        if (depth == 'shallow'):
            self.hidden = torch.nn.Linear(n_feature, 32)   # 隐藏层线性输出
            self.predict = torch.nn.Linear(32, n_output)   # 输出层线性输出
        else:
            self.hidden1 = torch.nn.Linear(n_feature, 18)
            self.hidden2 = torch.nn.Linear(18, 15)
            self.hidden3 = torch.nn.Linear(15, 4)
            self.predict = torch.nn.Linear(4, n_output)

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        if (depth == 'shallow'):
            x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
            x = self.predict(x)             # 输出值
        elif(depth == 'middle'):
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
            x = self.predict(x)
        else:
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
            x = F.relu(self.hidden4(x))
            x = F.relu(self.hidden5(x))
            x = F.relu(self.hidden6(x))
            x = F.relu(self.hidden7(x))
            x = self.predict(x)
        return x

depth = 'middle'

neural_net = []
for i in range(8):
    tmp = Net(depth, n_feature=784, n_output=10)
    neural_net.append(tmp)
accuracy = []
loss_eventual = []
pca_weight_first = []
pca_weight = []
for time in range(8):
    net = neural_net[i]
    print(net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
    #print(list(net.parameters()))
    
    loss_tmp = []
    weight_tmp = []
    accuracy_tmp = []
    parameter_tmp = []
    parameter_first_tmp = []
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            b_x = b_x.view(-1,28*28)
            output = net(b_x)               # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            loss_tmp.append(loss.data.numpy())
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
    
            if step % 3 == 0:
                #test_output = net(test_x)
                #pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
                #accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                loss_tmp.append(loss.data.numpy())
                #accuracy_tmp.append(accuracy)
                para = list(net.parameters())
                para_tmp = []
                for p in para:
                    para_tmp.append(p.data.numpy().flatten())
                parameter_first_tmp.append(para_tmp[0])
                #tmp = np.concatenate([para_tmp[0],para_tmp[1],para_tmp[2],para_tmp[3]])
                tmp = np.concatenate([a for a in para_tmp])
                parameter_tmp.append(tmp)
                #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
    print(time, 'time finished')
    parameter_first_tmp = np.array(parameter_first_tmp)
    parameter_tmp = np.array(parameter_tmp)
    loss_tmp = np.around(np.array(loss_tmp),2)
    pca_first_layer = PCA(n_components = 2)
    pca_all = PCA(n_components = 2)
    pca_all.fit(parameter_tmp)
    pca_first_layer.fit(parameter_first_tmp)
    pca_first_layer_new = pca_first_layer.transform(parameter_first_tmp)
    pca_tmp_new = pca_all.transform(parameter_tmp)
    loss_eventual.append(loss_tmp)
    pca_weight_first.append(pca_first_layer_new)
    pca_weight.append(pca_tmp_new)


color = ['b','g','r','c','m','y','k','orange']
#for j in range(8):
#    for i in np.arange(1,400,3):
#        plt.scatter(pca_weight[j][i][0],pca_weight[j][i][1],marker = '${%.2f}$'%(loss_eventual[j][i]),c = color[j])
#plt.savefig('hw1_2_1.png')
#plt.show()

for j in range(8):
    for i in np.arange(300,400,4):
        plt.scatter(pca_weight[j][i][0],pca_weight[j][i][1], alpha = 0.5, marker = '${%.2f}$'%(loss_eventual[j][i]),c = color[j])
plt.title('Visualize Optimization')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.savefig('hw1_2_1_good.png')
plt.show()

for j in range(8):
    for i in np.arange(300,400,4):
        plt.scatter(pca_weight_first[j][i][0],pca_weight_first[j][i][1], alpha = 0.5, marker = '${%.2f}$'%(loss_eventual[j][i]),c = color[j])
plt.title('Visualize Optimization')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.savefig('hw1_2_1_first_layer.png')
plt.show()












