#!usr/bin/env python
# -*- coding:utf-8 -*-
#Robin  2016.6
#Tensorflow 官方文档 教程1  MNIST机器学习入门

import tensorflow as tf
#导入mnist数据集
import tensorflow.examples.tutorials.mnist.input_data as input_data

#####~~~~导入数据
# mnist_train: 55000   28*28 = 784
#consist of:   mnist.train.images & mnist.train.labels
# mnist_test:  10000, 10
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

###~~~~~构建模型
#创建一个交互单元，x为占位符，None表示张量第一维可以是任何长度
x = tf.placeholder(tf.float32,[None,784])
#用Variable更好的代表参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([1,10]))

# W1 = tf.Variable(tf.zeros([30,10]))
# b1 = tf.Variable(tf.zeros([1,30]))
# y1 = tf.nn.sigmoid(tf.matmul(x,W)+b1)

#直接使用softmax回归模型分类：
y = tf.nn.softmax(tf.matmul(x,W)+b)

#使用交叉熵损失函数
#y_是一个新的占位符,占模型的输出结果   n*10
y_ = tf.placeholder('float',[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#优化算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

####~~~评估模型
#tf.arg_max()返回最大值索引（标签），cast将bool型结果转换成浮点型，然后计算均值得到准确率
prediction = tf.equal(tf.arg_max(y_,1),tf.arg_max(y,1))
accurary = tf.reduce_mean(tf.cast(prediction,"float"))


#####~~~~开始训练
#添加一个op初始化变量
init_op = tf.initialize_all_variables()
#在Session里面启动模型，并初始化变量
with tf.Session() as sess:
    sess.run(init_op)
    #迭代训练
    for i in xrange(1000):
        #随即从数据集中取数据
        batch_xs, batch_ys = mnist.train.next_batch(1000)   #next_batch怎么实现的？？？
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        if (i%10 == 0):
            print sess.run(accurary, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

###~~~问题：
#1. 为什么 单个softmax模型比逃一层sigmoid效果好
#2. 为什么bantch内样本多了，学习效果反而下降(梯度下降学习率的实现问题)
#3. batch和循环的关系









