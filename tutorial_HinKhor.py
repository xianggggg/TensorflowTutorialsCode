#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''Learn a simple tutorial from HinKhor
一个個基本的线性回归模型的实现
包含三种更新方法：minibatch大小的改变
包含学习率大小的改变
'''
#20160823
#20160830


import tensorflow as tf
import numpy as np

#线性回归模型例子1

#生成数据
train_data = []
for i in xrange(100):
    # 生出模拟数据
    xs = np.array([[i]])
    ys = np.array([[2 * i+5]])
    train_data.append([xs,ys])


# mini_batch/stocastic gradientOptimation
def get_data(train_data, batch=0):
    if batch == 1:
        # stocastic
        return train_data
    if batch == 0:
        # all
        xs = np.zeros((len(train_data),1))
        ys = np.zeros((len(train_data),1))

        for i in range(len(train_data)):
            xs[i,:] = train_data[i][0]
            ys[i,:] = train_data[i][1]
        return [[xs,ys]]
    if batch !=0 and batch !=1:
        #minibatch
        result = []
        size = batch
        np.random.shuffle(train_data)
        temp = [train_data[k:k + size] for k in xrange(0, len(train_data), size)]
        for i in temp:
            xs = np.zeros((size,1))
            ys = np.zeros((size,1))
            for j in range(size):
                xs[j,:] = i[j][0]
                ys[j,:] = i[j][1]
            result.append([xs,ys])
        return result


##########Tensorflow####################
#变量
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

#输入样本
#x的占位符，数量None，1维
x = tf.placeholder(tf.float32,[None,1])
#y的占位符
y_ = tf.placeholder(tf.float32,[None,1])

#运算
y = tf.matmul(x,W)+b

#Cost函数
#cost = tf.reduce_sum(tf.pow((y_-y),2))
cost = tf.reduce_mean(tf.square(y_-y))

#训练过程，使用占位符同步改变学习绿
#train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
learn_rate = tf.placeholder(tf.float32,shape = None)
train_step = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost)

#初始化
init = tf.initialize_all_variables()

#启动tensorflow会话
steps = 1000
initial_rate = 0.0001
with tf.Session() as sess:
    sess.run(init)
    for i in xrange(1,steps):
        # 生出模拟数据
        data = get_data(train_data,10)
        for j in data:
            xs = j[0]
            ys = j[1]
            #Train
            feed = {x:xs, y_:ys, learn_rate:initial_rate/i}
            sess.run(train_step,feed_dict=feed)
            #view
            print ('W: %f' % sess.run(W))
            print ('b: %f' % sess.run(b))

        print ("%d epochs finished" % i)

'''
ques: 为什么b的收敛速度这么慢
'''