"""

S04L00_Clarification_of_Vectors_and_Commutativity

This does not exist as a lesson in the course but I put it in anyway

https://www.udemy.com/course/machine-learning-data-science-foundations-masterclass/learn/lecture/23006968#overview

Section 04 Lesson 00

14 January 2023

"""
# pip install numpy 
# pip install torch
# pip install tensorflow[and-cuda]
# pip install matplotlib
import numpy as np
import tensorflow as tf
import torch as pt
import matplotlib.pyplot as plt


print("---------------------------------------------")
# First dimension is (dim 0)              (axis=0)
# Second dimension is (dim 1) etc.        (axis=1)
# Clarifying difference between one dimensional matrices (vectors)
# And 2D matrices, where one of the dimensions has len 1
# There can be confusion.
print("---------------------------------------------")
# How to workout the dimensions of a matrix: count from the outer brackets in
# Ask: "How many elements are there directly inside these brackets?"
# 1D vector where (dim 0) has len 3 and no second dimension
# Here (dim 0), is neither a row nor a column
X = np.array([2,5,-3]) 
print(X)        # [ 2  5 -3]
print(X.shape)  # (3,)

print("---------------------------------------------")
 # 2D matrix where (dim 0) has len 1 and (dim 1) has len 3
 # In a 2D matrix we call (dim 0) rows and (dim 1) cols
A = np.array([[2,5,-3]]) # 1 row, 3 cols
print(A)        # [[ 2  5 -3]]
print(A.shape)  # (1, 3)

print("---------------------------------------------")
 # 2D matrix where (dim 0) has len 3 and (dim 1) has len 1
 # In a 2D matrix we call (dim 0) rows and (dim 1) cols
B = np.array([[2],[5],[-3]]) # 3 rows, 1 col
print(B)
# [[ 2]
# [ 5]
# [-3]]
print(B.shape) # (3, 1)

print("---------------------------------------------")
Y = X.T
print(Y)        # [ 2  5 -3]    # Same as X
print(Y.shape)  # (3,)          # Same as X

print("---------------------------------------------")
print("Matrix Multiplication is non-comutative")
print("---------------------------------------------")
N = np.array([[0,1,2],[3,4,5]]) 
print(N)
# [[0 1 2]
# [3 4 5]]
print(N.shape) # (2, 3) 2 rows 3 cols

print("---------------------------------------------")
M = np.array([[0,1,2],[3,4,5],[7,8,9]])
print(M)
print(M.shape) # (3, 3) 3 rows 3 cols
# [[0 1 2]
#  [3 4 5]
#  [7 8 9]]

# N * M
# (2, 3) * (3, 3) = (2, 3) - middle two dims cancel out (are the same) to create the resulting shape
print(np.dot(N,M))
# [[17 20 23]
#  [47 59 71]]

print("---------------------------------------------")
# M * N
# (3,3) * (2,3) can't happen because 3 != 2
# print(np.dot(M,N))
# ValueError: shapes (3,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)

print("---------------------------------------------")

X = np.array([1,2,-3]) 
Y = np.array([4,5,-6]) 

XtimesY = np.dot(X,Y)

# (3,) * (3,) - Tim's canceling rule dosen't apply to vector dot products
print(XtimesY.shape) # () 0 dimensions, i.e. a scalar product
print(XtimesY)
# 38 -  a scalar










