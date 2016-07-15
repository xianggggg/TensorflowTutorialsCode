#!/usr/bin/env python
# -*- coding:utf-8 -*-
#Try tensorflow  introduce
#20160421
#Que:   
#Variable, matul
import tensorflow as tf
import numpy as np

#generate test data
x_data = np.float32(np.random.rand(2,100)) #2row,100col
y_data = np.dot([0.1,0.2],x_data)+0.3	   #dot product

#construct a linear model
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(W,x_data)+b

#calculate  loss
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#initialize
init = tf.initialize_all_variables()

#boot graph
sess = tf.Session()
sess.run(init)

#fitting plan
for step in xrange(0,201):
	sess.run(train)
	if step % 20 == 0:
		print step, sess.run(W), sess.run(b)



