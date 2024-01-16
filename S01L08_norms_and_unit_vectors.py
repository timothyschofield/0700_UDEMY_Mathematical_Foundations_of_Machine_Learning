"""
Norms and Unit Vectors
https://www.udemy.com/course/machine-learning-data-science-foundations-masterclass/learn/lecture/23006968#overview

Section 01 Lesson 08

06 January 2023

"""
# pip install numpy 
# pip install torch
# pip install tensorflow[and-cuda]
# pip install matplotlib
import numpy as np
import tensorflow as tf
import torch as pt
import matplotlib.pyplot as plt

"""
║x║ denenotes the norm which is the magnitude of a vector
|x| denotes the absolute value of a scalar
"""

###############################################
# This is the L2 norm
# In Python
x = [25, 2, 5]
normL2 = (25**2 + 2**2 + 5**2)**(1/2)
print(normL2) # 25.573

# numpy
normL2_np = np.linalg.norm(x)
print(normL2_np) # 25.573

###############################################
# This is the L1 norm
x = [25, 2, 5]
normL1 = np.abs(25) + np.abs(2) + np.abs(5)
###############################################
# This is the L2 squared norm
x = [25, 2, 5]
normL2squared = 25**2 + 2**2 + 5**2
#############################################################
# This is the L infinity norm
x = [25, 2, 5]
normLinfinity = np.max(np.abs(25) + np.abs(2) + np.abs(5))
#############################################################
# 10. Tensors
#############################################################
# numpy
X = np.array([[25, 2], [5, 26], [3, 7]])
print(X)
"""
[[25  2]
 [ 5 26]
 [ 3  7]]
"""

print(X.shape)
# (3, 2)

print(X.size)
# 6

Xcol0 = X[:,0]
print(Xcol0)
# [25  5  3]

Xrow1 = X[1,:]
print(Xrow1)
# [ 5 26]

# PyTorch
X_pt = pt.tensor([[25, 2], [5, 26], [3, 7]])
# very similar to numpy
X_pt.shape

# TensorFlow
X_tf =tf.Variable([[25, 2], [5, 26], [3, 7]])
tf.shape(X_tf) # not as nice or pythonic as PyTorch
#############################################################
# 11. Generic Tensor Notation
#############################################################
# e.g. a batch of colour images
# 32 colour, images, 28 by 28 pixels, 3 channels

images_pt = pt.zeros([32, 28, 28, 3])

images_tf = tf.zeros([32, 28, 28, 3])





























