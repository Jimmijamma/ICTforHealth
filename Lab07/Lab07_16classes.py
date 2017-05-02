#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:01:27 2016

@author: jimmijamma
"""

# we import the .mat file of arrhythmia
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mat_file=scio.loadmat('arrhythmia.mat')
data=mat_file.get('arrhythmia')

data = data[~np.all(data==0, axis=1)] # deleting eventual zero columns
class_id=data[:,-1]
n_classes=int(max(class_id))

(N,F)=np.shape(data)

mx_classes=np.zeros((N,n_classes))
for i in range(0,N-1):
    mx_classes[i][int(class_id[i])-1] = 1


data=data[:,:-1]
(N,F)=np.shape(data)

mean=np.mean(data)
std=np.std(data)
x_norm=(data-mean)/std

mean = np.mean(x_norm,0)
var = np.var(x_norm,0)

n_healthy=sum(class_id==0)
n_ill=sum(class_id==1)

# initializing the neural network graph
tf.set_random_seed(1234)
learning_rate = 1e-2
n_hidden_nodes_1=64
n_hidden_nodes_2=32

        
x = tf.placeholder(tf.float64, [N, F])
t = tf.placeholder(tf.float64, [N, n_classes])

# first layer
w1 = tf.Variable(tf.random_normal(shape=[F, n_hidden_nodes_1], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights"))
b1 = tf.Variable(tf.random_normal(shape=[1, n_hidden_nodes_1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases"))
a1 = tf.matmul(x, w1) + b1
z1 = tf.nn.sigmoid(a1)

# second layer
w2 = tf.Variable(tf.random_normal(shape=[n_hidden_nodes_1, n_hidden_nodes_2], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights2"))
b2 = tf.Variable(tf.random_normal(shape=[1, n_hidden_nodes_2], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases2"))
a2 = tf.matmul(z1, w2) + b2
z2 = tf.nn.sigmoid(a2)

# second layer
w3 = tf.Variable(tf.random_normal(shape=[n_hidden_nodes_2, n_classes], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights3"))
b3 = tf.Variable(tf.random_normal(shape=[1, 1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases3"))
y = tf.nn.softmax(tf.matmul(z2, w3) + b3)

#implementation of gradient algorithm
cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))
optim = tf.train.GradientDescentOptimizer(learning_rate, name="GradientDescent")
optim_op = optim.minimize(cost, var_list=[w1,b1,w2,b2,w2,b3])

init = tf.global_variables_initializer()
        
#--- run the learning machine
sess = tf.Session()
sess.run(init)

xval = x_norm.reshape(N,F)
tval = mx_classes.reshape(N, n_classes)
for i in range(10000):
    # generate the data
    # train
    input_data = {x: xval, t: tval}
    sess.run(optim_op, feed_dict = input_data)
    if i % 1000 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=input_data,session=sess))


#--- print the final results
print(sess.run(w1),sess.run(b1))
print(sess.run(w2),sess.run(b2))
print(sess.run(w3),sess.run(b3))


decisions = np.zeros(N)
yval = y.eval(feed_dict=input_data,session=sess)
for i in range (0,N):
    decisions[i]=np.argmax(yval[i])
    
n_strike = float((decisions == class_id-1).sum())
p_strike = 100.0*n_strike/N
        

