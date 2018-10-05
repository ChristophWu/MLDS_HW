#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 23:04:52 2018

@author: jason
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 False


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
test_y = test_data.test_labels[:2000]

class CNN_DEEP(nn.Module):
    def __init__(self):
        super(CNN_DEEP, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=8,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(8, 12, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(12 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output    

class CNN_SHALLOW(nn.Module):
    def __init__(self):
        super(CNN_SHALLOW, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1,5,5,1,2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),
                )
        self.out = nn.Linear(5*14*14,10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

net_depth = ['deep','shallow']
for depth in net_depth:      
    if(depth == 'deep'):
        cnn = CNN_DEEP()
    else:
        cnn = CNN_SHALLOW()
    print(cnn)  # net architecture
    
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
    
    tmp_loss = []
    tmp_accuracy = []
    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
    
            output = cnn(b_x)               # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            tmp_loss.append(loss.data.numpy())
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
    
            if step % 20 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                tmp_accuracy.append(accuracy)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), 
                      '| test accuracy: %.2f' % accuracy)
    tmp_loss = np.array(tmp_loss)
    tmp_accuracy = np.array(tmp_accuracy)
    if (depth == 'deep'):
        deep_loss = tmp_loss
        deep_accuracy = tmp_accuracy
    else:
        shallow_loss = tmp_loss
        shallow_accuracy = tmp_accuracy

### plot accuracy curve
plt.plot(np.arange(len(deep_accuracy)), deep_accuracy, color='green', label='deep_accuracy')
plt.plot(np.arange(len(shallow_accuracy)), shallow_accuracy, color='red', label='shallow_accuracy')

plt.legend()

plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.savefig('accuracy.png')
plt.show()

### plot loss curve
plt.plot(np.arange(len(deep_loss)), deep_loss, color='green', label='deep_loss')
plt.plot(np.arange(len(shallow_loss)), shallow_loss, color='red', label='shallow_loss')

plt.legend()

plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.show()
                


















