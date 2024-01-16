"""
Scalar Tensors
https://www.udemy.com/course/machine-learning-data-science-foundations-masterclass/learn/lecture/23006968#overview

Section 01 Lesson 05

05 January 2024

The two most popular "automatic differentiation" labraries are PyTorch and TensorFlow

https://github.com/jonkrohn/ML-foundations
"""
##################################################
# Lesson 6. Scalars 
##################################################
# Numpy
# pip install numpy - 85MB
import numpy as np

# Lets create a scalar tensor

# Vanilla Python:
x = 25
# print(x, type(x))
# 25 <class 'int'>

##################################################
# PyTorch tensors are designed to be "pythonic", i.e. to feel and behave like NumPy arrays
# The advantage of PyTorch tensors over Numpy is that they are designed to work with GPUs

# pip install torch - 4.9GB
import torch

x_pt = torch.tensor(25)
# print(x_pt, type(x_pt))
# tensor(25) <class 'torch.Tensor'>

x_pt = torch.tensor(25, dtype=torch.float16)
# print(x_pt, type(x_pt))
# tensor(25., dtype=torch.float16) <class 'torch.Tensor'>

# print(x_pt.shape)
# torch.Size([]) # no dimensions

##################################################
# TensorFlow
# pip install tensorflow[and-cuda] - 1.5G
import tensorflow as tf

x_tf = tf.Variable(25)
# print(x_tf)
# <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=25>
# print(x_tf.shape)
# ()

x_tf = tf.Variable(25, dtype=tf.float16)
# print(x_tf)
# <tf.Variable 'Variable:0' shape=() dtype=float16, numpy=25.0>
# print(x_tf.shape)
# ()

# convert to numpy
x_tf = x_tf.numpy()
# print(x_tf, type(x_tf))
# 25 <class 'numpy.int32'>
# print(x_tf.shape)
# ()
##################################################
# Lesson 7. Vectors and Vector Transpositions
# 1 dimension
##################################################
# Numpy
# Vectors (Rank 1 Tensors) in NumPy
# 1 dimensional arrays, typicaly in lowercase, italics and bold


x = np.array([25, 2, 5]) # type argument is optional, e.g.: dtype=np.float16
# print(x)
# [25, 2, 5]

# print(len(x))       # 3
# print(x.shape)      # (3,)   <<<<<<<< Why!?
# print(type(x))      # <class 'numpy.ndarray'>
# print(type(x[0]))   # <class 'numpy.int64'>

"""
print(x.shape)      # (3,) Why?
Because you read the tuple returned from the shape function from the left to the right.
(3,) means the first dimension has length 3 and there is no second dimension.
(3, 0) wouldn't make sense - "first dimention has length 3, second dimension has length 0" - nonsense.
If the shape is 1-D you will see only 1st dimension, 2-D only 2 dimensions and so on. 

The floating comma is a little clumbsy but it get around a syntactic problem, 
i.e. "(3)" would just be parsed as "3" by the interpreter.
If it were not for the above, (3) might be a better representation of the 1-dimensional shape of a vector.

"""
x = np.array(25)
# print(x.shape)
# () - a scalar, 0 dimensions


x = np.array([25, 2, 5])
print(x)
# [25  2  5]

print(x.shape)
# (3,) - a vector, 1 dimensions

print(type(x))
# numpy.ndarray

#########################################
# Vector Transposition in numpy
#########################################
x = x.T
print(x)
# [25  2  5]    # arrarently, no change!
print(x.shape)
# (3,)          # arrarently, no change!

# This is because x is a 1-dimensional array, and when we transpose it, 
# there is no dimesnion for it to be transposed into.
# However 
x = np.array([[25, 2, 5]])
print(x)
# [[25, 2, 5]]
print(x.shape)
# (1,3) # shape = (rows, cols) - this is called a "row vector"


x = x.T
print(x)
"""
[[25]
 [ 2]
 [ 5]]  
"""
print(x.shape)
# (3,1) # shape = (rows, cols) - this is called a "column vector"

#########################################
# Zero Vectors in numpy
#########################################
# They are vectors that consist entirly of zeros
z = np.zeros(3)
print(z)
# [0. 0. 0.]
print("#######################################")
#########################################
# Vectors in PyTorch and TensorFlow
#########################################
x_pt = torch.tensor([25, 2, 5])
print(x_pt)
# tensor([25,  2,  5])


print("#######################################")
x_tf = tf.Variable([25, 2, 5])
print(x_tf)
# <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([25,  2,  5], dtype=int32)>










