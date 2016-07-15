#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf

#sess = tf.InteractiveSession()

state = tf.Variable(0,name="conter")

#create a op to count
#one = tf.constant(1)
new_value = tf.add(state,1)
update = tf.assign(state,new_value)

#initialization
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(state)
    for _ in range(3):
        sess.run(update)
        print sess.run(state)