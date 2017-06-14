#!/usr/bin/env python
# -*- coding:utf-8 -*-
#官方教程第二节
#Build a Multilayer Convolutional Network
#
# stride of one
# zero padded
#Robin  2017.6
# =============================================================



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def conv2d(x, W):
    '''计算卷积函数:
    - 步骤：
    - 1. 输入tensor[batch, in_height, in_width, in_channels]以及卷积窗[filter_height, filter_width, in_channels, out_channels]
    - 2. 卷积窗变化为 2 - D矩阵[filter_height * filter_width * in_channels, output_channels].
    - 3. 从输入图片中提取每一块，塑造成一个虚拟tensor[batch, out_height, out_width, filter_height * filter_width * in_channels].
    - 4. 用每一块右乘卷积窗'''

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#CNN结构
def deepnn(x):
    # #eshape the image: [-1,dimx,dimy,color]， -1代表自适应
    #  Last dimension is for "features" - there is only one here, since images are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积层：卷基层+最大池化
    # 权重tensor：【5,5,1,32】 = 【dim,dim,color,feature_number（1->32 feature map）】
    # Bias: [32]
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    # 这里dropoff只存在于最后一层？应该不是，作用于整个结构
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    # 最后一层输出层，softmax在后面应用
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob



def main(_):
  # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    # 输入图片的占位符
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    # y是标签
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net。 keep prob为dropout率
    y_conv, keep_prob = deepnn(x)

    #最后使用 softmax搭配交叉熵损失函数
    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    #使用Adam优化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #以list的方式判断预测结果与实际结果
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    #将correct_prediction转化为float32，并计算举止
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #开始训练
    with tf.Session() as sess:
        #初始化所有变量
        sess.run(tf.global_variables_initializer())
        #总迭代20000次
        for i in range(20000):
            #batch大小为50
            batch = mnist.train.next_batch(50)
            #每一百次打印一次准确率
            if i % 100 == 0:
                #这里accurary.eval相当于调用sess.run(accuracy)
                #batch[0]是数据，[1]是标签
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            #每一部训练，dropout率为0.5
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        #最后用测试集合打印准确率
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='./MNIST_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    #运行tf
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)






## Relu RU
## 梯度消失