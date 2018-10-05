#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:49:28 2018

@author jason
"""

import tensorflow as tf
import numpy as np
from numpy import linalg as LA

def dense(input, weight_shape, name, reuse, activation= 'relu'):
    shape = np.asarray(weight_shape)
    with tf.variable_scope(name, reuse= reuse):
        
        flatten_weight = tf.get_variable('flatten_weight',
                         shape= np.prod(shape),
                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        weight = tf.reshape(flatten_weight, shape, name= 'weight')
        bias = tf.get_variable('bias',
                     shape= shape[-1],
                     initializer= tf.constant_initializer(0))
        output = tf.matmul(input, weight) + bias
        """
        output = tf.layers.dense(input, shape[-1])
        """
        if activation == 'relu':
            output = tf.nn.relu(output)
        if activation == 'softmax':
            output = tf.nn.softmax(output)
        if activation == 'tanh':
            output = tf.nn.tanh(output)
        
        return output

class SimulateFunctionModel:
    def __init__(self):
        self.reuse = False
        self.name = 'SimulateFunctionModel'
        
    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):
            input_shape = input.get_shape().as_list()
            dense1 = dense(input, [input_shape[-1], 20], 'dense1', reuse= self.reuse)
            dense2 = dense(dense1, [20, 20], 'dense2', reuse= self.reuse)
            dense3 = dense(dense2, [20, 20], 'dense3', reuse= self.reuse)
            output = dense(dense3, [20, 1], 'output', activation='linear', reuse= self.reuse)
            
        self.reuse = True
        return output



def objective_function(x):
    return np.sin(2 * x)/ 2 * x

train_data_size = 1000
batch_size= 100
EPOCH = 1000
gradient_threshold = 0.05

train_x = np.random.normal(scale= 10, size= (train_data_size, 1))
train_y = objective_function(train_x)

random_order = np.arange(train_data_size)
#%%
minimal_ratio_record = []
loss_record = []
for time in range(100):
    graph = tf.Graph()
    with graph.as_default():
        x_placeholder = tf.placeholder(tf.float32, (None, 1), name= 'x_placeholder')
        y_placeholder = tf.placeholder(tf.float32, (None, 1), name= 'y_placeholder')
        
        model = SimulateFunctionModel()
        prediction = model(x_placeholder)
        
        mse_loss = tf.reduce_mean(tf.squared_difference(y_placeholder, prediction))
        train_optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_step = train_optimizer.minimize(mse_loss)
        
        
        total_grad_norm = tf.constant(0, dtype= tf.float32)
        for variable in tf.trainable_variables():
            [grad] = tf.gradients(ys= mse_loss, xs= variable)
            total_grad_norm += tf.reduce_sum(grad**2)
        total_grad_norm = tf.sqrt(total_grad_norm)
        grad_optimizer = tf.train.AdamOptimizer(learning_rate= 1)
        min_grad = grad_optimizer.minimize(total_grad_norm)
        
        writer = tf.summary.FileWriter("TensorBoard/", graph = graph)
    
    with tf.Session(graph= graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, EPOCH+1):
            np.random.shuffle(random_order)
            train_x = train_x[random_order]
            train_y = train_y[random_order]
        
            total_loss = 0
            for idx in range(train_data_size//batch_size):
                x = train_x[idx * batch_size : (idx+1) * batch_size]
                y = train_y[idx * batch_size : (idx+1) * batch_size]
                
                feed_dict= {x_placeholder: x, y_placeholder:y}
                _, loss= sess.run([train_step, mse_loss], feed_dict= feed_dict)
                
                total_loss += (loss / batch_size)
        print('time:', time, 'loss:', total_loss)
        loss_record.append(total_loss)
        # Find where gradient is 0
        while True:
            feed_dict= {x_placeholder: train_x, y_placeholder:train_y}
            _, gradient_norm = sess.run([min_grad, total_grad_norm], feed_dict= feed_dict)  

            if gradient_norm <= gradient_threshold:
                break
        print('time:', time,'gradient_norm:', gradient_norm)
        # Calculate minima ratio w.r.t eigen values which are positive
        eigen_values = np.asarray([])
        for variable in tf.trainable_variables():
            hess = sess.run(tf.hessians(mse_loss, variable), feed_dict= feed_dict)
            eigen_values = np.append(eigen_values, (LA.eigvals(hess).reshape(-1,)))
        
        minimal_ratio = np.sum(eigen_values > 0) / np.prod(eigen_values.shape)
        minimal_ratio_record.append(minimal_ratio)
        
#%%
import matplotlib.pyplot as plt
fig1 = plt.figure(1)
plt.ylabel('Loss')
plt.xlabel('Minima Ratio')

plt.scatter(minimal_ratio_record, loss_record)

plt.show()
        
fig1.savefig('HW1_2_3.png')
        
