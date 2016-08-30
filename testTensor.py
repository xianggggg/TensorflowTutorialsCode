#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf

import random as rn
import numpy as np

a = [["A",3],["d",4],["f",5]]

b = a[:]

print a

# print np.random.shuffle(a)

#print np.random.permutation(a)

print rn.shuffle(a)

print a

print b