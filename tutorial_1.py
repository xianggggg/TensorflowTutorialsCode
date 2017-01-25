#!/usr/bin/env python
# -*- coding:utf-8 -*-
#通过学习理解官方教程，一方面掌握TF的基本用法，另一方面学习先进的Python代码风格和规范
#Robin  2016.6
# =============================================================
"""
learning by Tensorflow Tutorial: mnist_softmax
"""
#引用
#版本兼容
from __future__ import division
from __future__ import print_function
#解析命令行参数
import argparse
import sys
#TF
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

#通过设置flags，将dir参数包装到代码内部，只用argparse处理，不传参。
FLAGS = None

#主函数
def main(_):
    #导入数据,类型为tf定义的dataset，以数据名命名
    #FLAGS通过argparse定义并增加参数。
    mnist  = input_data.read_data_sets(FLAGS.dir, one_hot=True)

    #1.创建模型 x,y,w,b,_y
    x = tf.placeholder(tf.float32,[None,784],name = 'x')
    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x,w)+b
    y_ = tf.placeholder(tf.float32, [None,10])

    #2. 构建损失函数， 这里的交叉熵其实是log损失函数，是交叉熵在softmax上的推广
    sy = tf.nn.softmax(y)
    cross_entropy = tf.reduce_mean(-tf.reduce_mean(y_ * tf.log(sy),reduction_indices=[1]))
    #防止数值计算问题，tf提供了整合的方法
    cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits = y))
    #优化方法
    train_step = tf.train.GradientDescentOptimizer(learning_rate = FLAGS.lrate).minimize(cross_entropy)

    #3. 执行op
    #使用InteractiveSession, 用x.run()代替 sess.run(x)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #Train
    for _ in range(1000):
        #定义好的next_batch方法
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
        sess.run(train_step, feed_dict = {x:batch_xs, y_: batch_ys})

    # Evaluation
    #计算准确率每一次迭代的准确率,使用测试集
    #感觉tutorial有错误，此处应该用softmax的结果，而不是直接用y
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y),1), tf.argmax(y_,1))
    accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accurary, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

    sess.close()

#运行程序
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default = './MNIST_data', help = 'Directory for storing input minst data')
    parser.add_argument('--lrate', type = float, default = 0.5, help = 'Learning rate')
    parser.add_argument('--batchsize', type=int, default=100, help='mini_batch size')
    FLAGS = parser.parse_args()


    tf.app.run(main = main, argv= None)











