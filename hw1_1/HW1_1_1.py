#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:49:37 2018

@author: jason
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# flag = 1, return y = sin(5*pi*x)/(5*pi*x)
# else return y = sin(2*pi*x^2)/(2*pi*x)
def get_function(x,flag):
    if (flag == 1):
        y = np.sin(5 * np.pi * x)/(5 * np.pi * x)
    else:
        y = np.sin(2 * np.pi * x * x)/(2 * np.pi * x)
    return y

###### get the training data ######
flag = 0
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = get_function(x,flag)
# plot to see the function
#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()

###### define net ######
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, depth, n_feature, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        if (depth == 'shallow'):
            self.hidden = torch.nn.Linear(n_feature, 190)   # 隐藏层线性输出
            self.predict = torch.nn.Linear(190, n_output)   # 输出层线性输出
        elif(depth == 'middle'):
            self.hidden1 = torch.nn.Linear(n_feature, 18)
            self.hidden2 = torch.nn.Linear(18, 15)
            self.hidden3 = torch.nn.Linear(15, 4)
            self.predict = torch.nn.Linear(4, n_output)
        else:
            self.hidden1 = torch.nn.Linear(n_feature, 5)
            self.hidden2 = torch.nn.Linear(5, 10)
            self.hidden3 = torch.nn.Linear(10, 10)
            self.hidden4 = torch.nn.Linear(10, 10)
            self.hidden5 = torch.nn.Linear(10, 10)
            self.hidden6 = torch.nn.Linear(10, 10)
            self.hidden7 = torch.nn.Linear(10, 5)
            self.predict = torch.nn.Linear(5, n_output)

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

# if depth == shallow, Net:1-->190-->1  total weight:571
# if depth == middle, Net:1-->18-->15-->4-->1  total weight:572
# if depth == deep, Net:1-->5-->10-->10-->10-->10-->10-->5-->1  total weight:571
#depth = 'middle'
#net = Net(depth, n_feature=1, n_output=1)
#print(net)  # net 的结构
# optimizer 是训练的工具
#optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
#loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

###### train for 20000 epoch #######
depth_array = ['shallow','middle','deep']
for depth in depth_array:
    tmp_loss = []
    net = Net(depth, n_feature=1, n_output=1)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()
    for t in range(20000):
        prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
    
        loss = loss_func(prediction, y)     # 计算两者的误差
        tmp_loss.append(loss.data.numpy())
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    tmp_prediction = net(x).data.numpy()
    if (depth == 'shallow'):
        loss_shallow = tmp_loss
        prediction_shallow = tmp_prediction
    elif(depth == 'middle'):
        loss_middle = tmp_loss
        prediction_middle = tmp_prediction
    else:
        loss_deep = tmp_loss
        prediction_deep = tmp_prediction
        
###### plot the simulation curve ######
plt.plot(x.data.numpy(), prediction_shallow, color='green', label='prediction_shallow')
plt.plot(x.data.numpy(), prediction_middle, color='red', label='prediction_middle')
plt.plot(x.data.numpy(), prediction_deep, color='blue', label='prediction_deep')
plt.plot(x.data.numpy(), y.data.numpy(),  color='skyblue', label='real_curve')

plt.legend()

plt.xlabel('x')
plt.ylabel('y_prediction')
plt.savefig('prediction_function2.png')
plt.show()
# plot loss curve
plt.plot(np.arange(20000), loss_shallow, color='green', label='loss_shallow')
plt.plot(np.arange(20000), loss_middle, color='red', label='loss_middle')
plt.plot(np.arange(20000), loss_deep, color='blue', label='loss_deep')

plt.legend()
plt.ylim(0, 0.1)

plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('loss_function2.png')
plt.show()




