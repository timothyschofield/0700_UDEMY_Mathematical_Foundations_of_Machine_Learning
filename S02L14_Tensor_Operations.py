"""
Tensor Operations

https://www.udemy.com/course/machine-learning-data-science-foundations-masterclass/learn/lecture/23006968#overview

Section 02 Lesson 14

07 January 2023

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
# 14. Tensor Transpositions
#############################################################
# 08 January 2023 12:45
# numpy

X = np.array([[25, 2], [5, 26], [3, 7]])
print(X)
"""
[[25  2]
 [ 5 26]
 [ 3  7]]
"""

X = X.T
print(X)
"""
[[25  5  3]
[ 2 26  7]]
"""

# PyTorch
X_pt = pt.tensor([[25, 2], [5, 26], [3, 7]])
X_pt = X_pt.T

# TensorFlow
X_tf =tf.Variable([[25, 2], [5, 26], [3, 7]])
X_tf = tf.transpose(X_tf)

#############################################################
# 15. Basic Tensor Arithmetic, including the Hadamard Product
#############################################################
# Adding or multiplying with a scalar - operations occur to all elements, 
# the tensor shape remains the same.

# numpy
X = np.array([[25, 2], [5, 26], [3, 7]])
# print(X)
# print(X + 2)
# print(X * 2)

# PyTorch
X_pt = pt.tensor([[25, 2], [5, 26], [3, 7]])
# Operators are overloaded - could alternativly use pt.mul() and pt.add()
# print(X_pt + 2)
# print(X_pt * 2)

# TensorFlow
X_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])
# Operators are overloaded - could alternativly use tf.multiply() and tf.add()
# print(X_tf + 2)
# print(X_tf * 2)

# ==================== Hadamard X ⊙ A product =========================
# Notation is X ⊙ A 
# X + A or
# X * A
X = np.array([[25, 2], [5, 26], [3, 7]])
print(X)
"""
[[25  2]
 [ 5 26]
 [ 3  7]]
"""

A = X + 2
print(A)
"""
[[27  4]
 [ 7 28]
 [ 5  9]]
"""

print(X + A)
"""
[[52  6]
 [12 54]
 [ 8 16]]
"""
# We can also use the traditional "*" multiply sign - this is NOT matrix multiplication
print(X * A)
"""
[[675   8]
 [ 35 728]
 [ 15  63]]
"""
#############################################################
# 16. Tensor Reduction
# Calculating the sum across all the elements of a tensor is a common operation.
#############################################################
# numpy
X = np.array([[25, 2], [5, 26], [3, 7]])
print(X.sum())

# PyTorch
X_pt = pt.tensor([[25, 2], [5, 26], [3, 7]])
pt.sum(X_pt)

# TensorFlow
X_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])
tf.reduce_sum(X_tf)

###########################################
# In addition to the above, it is a relatively common requirement to calculate 
# a "reduction" across only one of the dimensions:

# For a two dimensional matrix
# Remember shape = (rows, cols), rows are axis=0, cols are axis=1
# I suppose for a tensor axes = (0, 1, 2, 3...n)
# This is difficult for me

# numpy
X = np.array([[25, 2], [5, 26], [3, 7]])
print(X)
print(X.shape)
"""
[[25  2]
 [ 5 26]
 [ 3  7]]
"""

# reduction accross the rows (axis=0) 
# means sum the three rows - leaving two results
# i.e. just reduce (squash) the row dimension to 0 (3, 2) -> (2,)
# i.e. reduce the dimension 0 to 0, leaving dimension 1
print(X.sum(axis=0)) 
print(X.sum(axis=0).shape) 
"""
[33 35]
(2,)
"""

# reduction accross down the cols (axis=1)
# means sum the two cols - leaving three results
# i.e. just reduce (squash) the col dimension to 0 (3,2) -> (3,)
# i.e. reduce the dimension 1 to 0, leaving the dimension 0
print(X.sum(axis=1)) 
print(X.sum(axis=1).shape) 
"""
[27 31 10]
(3,)
"""

# If we had a batch of 100, 28 by 28 pixel RGB images shape
# X (100, 28, 28, 3) and we wanted to 
# reduce them to grey scale would we X.sum(axis=3)? Not realy, but that is sort of the idea
# May be more like X.mean(axis=3)

# PyTorch
pt.sum(X_pt, 0)

# TensorFlow
tf.reduce_sum(X_tf, 0)

"""
The above uses the sum operator along a single matrix axis.
But - and this is less often - you can do 
these reductions with other operators:

e.g.

- maximum
- minimum
- mean
- product

and you can do the reduction along all or a selection axes.

"""
#############################################################
# 17. The Dot Product
# 4pm
#############################################################
"""
If we have two vectors of the same length, then we can calculate the dot product between them.
This is annotated in several different ways, including
    x . y
    x T Y
    (x, y)

The first is the most usual.
    The method is to calculate the product in an element-wise fashion and 
    then sum reductively across the product to a scalar value.
    
"""
# numpy
x = np.array([0, 1, 2])
y = np.array([25, 2, 5])

r = 25*0 + 1*2 + 2*5  
#  r = 12
# or
r = np.dot(x, y)
 
# PyTorch
# The elements have to be floats, hence the float in the last element
x_pt = pt.tensor([0, 1, 2.])
y_pt = pt.tensor([25, 2, 5.])
r = pt.dot(x_pt, y_pt)
print(r)
# tensor(12.)

# TensorFlow
x_tf = tf.Variable([0, 1, 2])
y_tf = tf.Variable([25, 2, 5])
r = tf.reduce_sum(tf.multiply(x_tf, y_tf))
print(r)
# tf.Tensor(12, shape=(), dtype=int32)
# Lesson 18 is an exercise
#############################################################
# 19. Solving Linear Systems with Substitition
# 20. Solving Linear Systems with Elimination
#############################################################
# So some exercises
#############################################################
# 21. Visualizing Linear Systems
#############################################################
"""
Method 1: Substitution

So to visualize equasion you have to isolate the y

Q1: Solve for x and y

	Eq1: y = 3x         - y is already isolated
	Eq2: -5x + 2y = 2   
    Eq2 becomes 
    2y = 2 + 5x
    
    y = (2 + 5x)/2
    
    Solution, calculated on paper by substitutions is (x, y) = (2, 6)
"""
import matplotlib.pyplot as plt
import numpy as np

start_x = -10
end_x = 10
num_points = 100
x_steps = np.linspace(start_x, end_x, num_points) 

y_eq1 = 3 * x_steps
y_eq2 = (2 + 5 * x_steps)/2

fig, sub1 = plt.subplots()

# Title and label axes
plt.title('Visualize Equasion 1')
plt.xlabel('x steps')
plt.ylabel('y')

# The range to plot over
sub1.set_xlim([0, 3])
sub1.set_ylim([0, 8])

# Do the plot
sub1.plot(x_steps, y_eq1, c='green')
sub1.plot(x_steps, y_eq2, c='brown')

# Solution, calculated on paper by substitutions is (x, y) = (2, 6)
plt.axvline(x=2, color='purple', linestyle='--') 
plt.axhline(y=6, color='purple', linestyle='--')

plt.show()















