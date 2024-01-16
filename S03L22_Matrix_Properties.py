"""
Matrix Properties

https://www.udemy.com/course/machine-learning-data-science-foundations-masterclass/learn/lecture/23006968#overview

Section 03 Lesson 22

10 January 2023

"""
# pip install numpy 
# pip install torch
# pip install tensorflow[and-cuda]
# pip install matplotlib
import numpy as np
import tensorflow as tf
import torch as pt
import matplotlib.pyplot as plt
#############################################################
# 22. Intro and
# 23. The Frobenius Norm 
#############################################################
# The Frobenius Norm measures the Euclidean size of a matrix
# numpy
X = np.array([[1, 2],[3, 4]])

Fn = (1**2 + 2**2 + 3**2 + 4**2)**(1/2)
print(Fn)
# 5.477225575051661
Fn = np.linalg.norm(X) # Same function we used for the L2 norm
print(Fn)
# 5.477225575051661

# PyTorch
# PyTorch requires floats, so note the final "4."
X_pt = pt.tensor([[1, 2],[3, 4.]])
print(pt.norm(X_pt))
# tensor(5.4772)

# TensorFlow
# TensorFlow also requires floats, so note the final "4."
X_tf = tf.Variable([[1, 2],[3, 4.]])
print(tf.norm(X_tf))
# tf.Tensor(5.4772253, shape=(), dtype=float32)

#############################################################
# 24. Matrix Multiplication
#############################################################
"""

The number of cols in the first matrix A
has to be equal to the number of rows in the second matrix B

- You end up with a matrix that has the same number of rows as A and the same number of columns of B 

- The dimension that was common to the two matrices, disappears.

(rows, cols) 
rows is axis=0
cols is axis=1

"""
###########################
##### Matix by Vector #####
###########################
# numpy
A = np.array([[3,4],[5,6],[7,8]]) # (3, 2)
b = np.array([1,2])               # (2,)

# print(np.dot(A, b))
# [11 17 23]                      # (3,)

# PyTorch
A_pt = pt.tensor([[3,4],[5,6],[7,8]])
b_pt = pt.tensor([1,2]) 
# print(pt.matmul(A_pt, b_pt))
# tensor([11, 17, 23])

# TensorFlow
A_tf = tf.Variable([[3,4],[5,6],[7,8]])
b_tf = tf.Variable([1,2]) 
# print(tf.linalg.matvec(A_tf, b_tf)) # have to be explicit in TensorFlow, matvec or matmul
# tf.Tensor([11 17 23], shape=(3,), dtype=int32)

###########################
##### Matix by Matrix #####
###########################
# numpy
A = np.array([[3,4],[5,6],[7,8]]) # (3, 2)
B = np.array([[1,9],[2,0]])       # (2,2)
# print(np.dot(A, B))
"""
[[11 27]
 [17 45]
 [23 63]]
"""

# PyTorch
A_pt = pt.tensor([[3,4],[5,6],[7,8]])
B_pt = pt.tensor([[1,9],[2,0]]) 
# print(pt.matmul(A_pt, B_pt))
"""
tensor([[11, 27],
        [17, 45],
        [23, 63]])
"""

# TensorFlow
A_tf = tf.Variable([[3,4],[5,6],[7,8]])
B_tf = tf.Variable([[1,9],[2,0]]) 
# print(tf.linalg.matmul(A_tf, B_tf)) 
"""
tf.Tensor(
[[11 27]
 [17 45]
 [23 63]], shape=(3, 2), dtype=int32)
"""
#############################################################
# 25. Symmetric and Identity Matrices
#############################################################
# 11 January 2024

# numpy
X_sym = np.array([[0,1,2],[1,7,8],[2,8,9]])
print(X_sym)
"""
[[0 1 2]
 [1 7 8]
 [2 8 9]]
"""

print(X_sym.T)
"""
[[0 1 2]
 [1 7 8]
 [2 8 9]]
"""

print(X_sym.T == X_sym)
"""
[[ True  True  True]
 [ True  True  True]
 [ True  True  True]]
"""

######################################
# Identity Matrix
######################################
# PyTorch

I = pt.tensor([[1,0,0],[0,1,0],[0,0,1]])
x_pt = pt.tensor([25,2,5])
print(pt.matmul(I, x_pt))
# tensor([25,  2,  5]) # unchanged

#############################################################
# 26. Exercise
# 27. Matrix Inversion
#############################################################
# numpy

# The features
X = np.array([[4,2],[-5,-3]])
print(X)
"""
[[ 4  2]
 [-5 -3]]
"""

Xinv = np.linalg.inv(X)
print(Xinv)
"""
[[ 1.5  1. ]
 [-2.5 -2. ]]
"""

# The house prices
y = np.array([4, -7])
print(y)
# [4, -7]

# The weights, we are after
w = np.dot(Xinv, y)
print(w)
# [-1.  4.]
# (b, c) = (-1.0, 4.0)
# To prove this y = Xw

y_prouf = np.dot(X, w)
print(y_prouf)
# [ 4.0 -7.0] which equals the origonal vector of house prices, y

###########################################
# In PyTorch and TensorFlow
Xinv_pt = pt.inverse(pt.tensor([[4,2],[-5,-3.]])) # Note: make one of the values a float

Xinv_tf = tf.linalg.inv(tf.Variable([[4,2],[-5,-3.]])) # Likewise, give it a float

print("---------------------------------------------")

######################################
# Matrix Inversion Where the is No Solution
######################################
# numpy

# So this represents a system of two linear equations, 
# one for each row, and two dimensions/features, one for each column.
X = np.array([[-4, 1],[-8, 2]]) # But not linealy independant!
# print(np.linalg.inv(X))
# numpy.linalg.LinAlgError: Singular matrix

######################################
# An orthornormal matrix multiplied by its transform
# is equal to the identity matrix
######################################
# numpy
print("---------------------------------------------")
X = np.array([[0,0,1],[1,0,0],[0,1,0]])
# print(X)
"""
[[0 0 1]
 [1 0 0]
 [0 1 0]]
"""
X_T = X.T
# print(X_T)
"""
[[0 1 0]
 [0 0 1]
 [1 0 0]]
"""
Y = np.dot(X, X_T)
print(Y) # The identity matrix, as required
"""
[[1 0 0]
 [0 1 0]
 [0 0 1]]
"""



