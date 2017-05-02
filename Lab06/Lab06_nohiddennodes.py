'''
Created on 30 apr 2017

@author: jimmijamma
'''

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# importing the input data
mat_file=scio.loadmat('parkinson.mat')
data=mat_file.get('new_matrix')

FEATURE = 5 # the feature we want to use as regressand

(N, F)=np.shape(data)
#normalizing data
x_norm=data
for i in range(N):
    x_norm[i] = (data[i]-np.mean(data,0))/np.std(data,0)

# removing the column that we use as regressand
regressand=data[:,FEATURE-1]

x_norm=np.delete(x_norm, [1-1,4-1,6-1,FEATURE-1], 1) # deleting some unused features
(N, F)=np.shape(x_norm)


# initializing the neural network graph
tf.set_random_seed(1234)
learning_rate = 1e-5
        
x = tf.placeholder(tf.float64, [N, F])
t = tf.placeholder(tf.float64, [N, 1])

w1 = tf.Variable(tf.random_normal(shape=[F, 1], mean=0.0, stddev=1.0, dtype=tf.float64, name="weights"))
b1 = tf.Variable(tf.random_normal(shape=[1, 1], mean=0.0, stddev=1.0, dtype=tf.float64, name="biases"))
y = tf.matmul(x, w1) + b1

#implementation of gradient algorithm
cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))
optim = tf.train.GradientDescentOptimizer(learning_rate, name="GradientDescent")
optim_op = optim.minimize(cost, var_list=[w1, b1])

init = tf.global_variables_initializer()
        
#--- run the learning machine
sess = tf.Session()
sess.run(init)

xval = x_norm.reshape(N,F)
tval = regressand.reshape(N, 1)
for i in range(1000):
    # generate the data
    # train
    input_data = {x: xval, t: tval}
    sess.run(optim_op, feed_dict = input_data)
    if i % 100 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=input_data,session=sess))


#--- print the final results
print(sess.run(w1), sess.run(b1))
a = sess.run(w1)
yval= y.eval(feed_dict = input_data, session=sess)

plt.plot(regressand,'r', label='regressand')
plt.plot(yval,'b', label='regression')
plt.xlabel('case number')
plt.grid(which='major', axis='both')
plt.legend()
plt.title('learning rate '+str(learning_rate))
plt.savefig('yval_vd_ytrue.pdf',format='pdf')
plt.show()

hist, bins = np.histogram((regressand-yval), bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.xlabel('squared error')
plt.title('learning rate '+str(learning_rate))
plt.savefig('squared_error'+ str(learning_rate)+'.pdf',format='pdf')
plt.show()





