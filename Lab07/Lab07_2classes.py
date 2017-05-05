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
class_id[np.where(class_id > 1)]=2
class_id=class_id-1

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
learning_rate = 1e-4
n_hidden_nodes_1=F
n_hidden_nodes_2=128
        
x = tf.placeholder(tf.float64, [N, F])
t = tf.placeholder(tf.float64, [N, 1])

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
w3 = tf.Variable(tf.random_normal(shape=[n_hidden_nodes_2, 1], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights3"))
b3 = tf.Variable(tf.random_normal(shape=[1, 1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases3"))
y = tf.nn.sigmoid(tf.matmul(z2, w3) + b3)

#implementation of gradient algorithm
cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))
optim = tf.train.GradientDescentOptimizer(learning_rate, name="GradientDescent")
optim_op = optim.minimize(cost, var_list=[w1,b1,w2,b2,w2,b3])

init = tf.global_variables_initializer()
        
#--- run the learning machine
sess = tf.Session()
sess.run(init)

xval = x_norm.reshape(N,F)
tval = class_id.reshape(N, 1)
for i in range(1000):
    # generate the data
    # train
    input_data = {x: xval, t: tval}
    sess.run(optim_op, feed_dict = input_data)
    if i % 100 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=input_data,session=sess))


#--- print the final results
print(sess.run(w1),sess.run(b1))
print(sess.run(w2),sess.run(b2))
print(sess.run(w3),sess.run(b3))

a = sess.run(w1)
yval = np.round(y.eval(feed_dict = input_data, session=sess))
yval= np.array(yval, dtype=np.int32).reshape(len(yval),)


hist, bins = np.histogram((class_id-yval), bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.xlabel('Squared error')
plt.title('Error distribution with learning rate '+str(learning_rate))
plt.savefig('squared_error_2classes'+ str(learning_rate)+'.png',format='png')
plt.show()


n_strike = float((yval == class_id).sum())
p_strike = 100.0*n_strike/N
p_true_positive = float(100*((yval >= 1) & (class_id >= 1)).sum())/n_ill
p_true_negative = float(100*((yval == 0) & (class_id == 0)).sum())/n_healthy
p_false_positive = float(100*((yval >= 1) & (class_id == 0)).sum())/n_healthy
p_false_negative = float(100*((yval == 0) & (class_id >= 1)).sum())/n_ill
                        

                        
plt.figure(figsize=(13,6))
index = np.arange(0,1,1)
                        
bars0 = [p_strike]
bars1 = [p_true_positive]
bars2 = [p_true_negative]
bars3 = [p_false_negative]
bars4 = [p_false_positive]

                        
plt.bar(index, bars0, 0.14, label="Strike Probability", color="black")
plt.bar(index + 0.14, bars1, 0.14, label="True Positive", color="blue")
plt.bar(index + 0.28, bars2, 0.14, label="True Negative", color="red")
plt.bar(index + 0.42, bars3, 0.14, label="False negative", color="green")
plt.bar(index + 0.56, bars4, 0.14, label="False positive", color="orange")
plt.title("Binary classification results")
plt.xlabel("Performance indicators")
plt.ylabel("Probability (%)")
plt.legend(loc=0, framealpha=0.7)   

