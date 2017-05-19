#!/usr/bin/env python
# -*- coding:utf-8 -*-
#Try tensorflow  introduce
#Test file

import tensorflow as tf
import numpy as np

w = tf.Variable(tf.zeros([784,10]))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())


