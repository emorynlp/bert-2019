# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-24 20:52
import mxnet.ndarray as nd
import mxnet as mx

# create a 2-dimensional array full of zeros with shape (2,3)
a = nd.zeros((2, 3))
# create a same shape array full of ones
b = nd.ones((2, 3), ctx=mx.gpu(3))
print(a * b)
