"""

S04L34 Eigenvectors and Eigenvalues

https://www.udemy.com/course/machine-learning-data-science-foundations-masterclass/learn/lecture/23006968#overview

Section 04 Lesson 34

16 January 2023

"""
# pip install numpy 
# pip install torch
# pip install tensorflow[and-cuda]
# pip install matplotlib
import numpy as np
import tensorflow as tf
import torch as pt
import matplotlib.pyplot as plt

# numpy
"""
An eigenvector is a special vector v such that when it is transformed by some matrix (lets say A), 
the product Av have the exact same direction as v. (or direct opposite as we saw in the Mona Lisa)

An eigenvalue is a scalar (traditionally represented by λ) that simply scales the eigenvector v, 
such that the following equation is satisfied:

                                            Av = λv
"""

# The transform matrix that we want the eigenvectors and eigenvalues of
A = np.array([[-1,4],[2,-2]])

# eig() retunrs a tuple of:
# a vector of eigenvalues - lambs # Be aware "lambda" is a reserved word in Python
# a matrix of eigenvectors - V

# Be aware "lambda" is a reserved word in Python

lambs, V = np.linalg.eig(A)

# The matrix contains as many eigenvectors as there are columns of A
# English - each columns is an eigenvector
print(V)
"""
[[ 0.86011126 -0.76454754]
 [ 0.51010647  0.64456735]]
"""
# And the lambs are the corresponding eigenvalues:
print(lambs)
# [ 1.37228132 -4.37228132]

# So lets confirm that A*v = lambda*v

# First column of V 
v = V[:,0] # [0.86011126 0.51010647] <<<<<<<<<<<<<<<<<<<< need to look into this syntax, this is my problem
print(v)

# and its eiganvalue
lamb0 = lambs[0]

Av = np.dot(A, v)
lamb0_v = lamb0 * v

# Yes, they are equal as required by Av = λv
print("Av", Av)
print("lamb0_v", lamb0_v)


v2 = V[:,1]         # second eigenvector
lamb1 = lambs[1]    # and its eiganvalue
Av2 = np.dot(A,v2)


def plot_vectors(vectors, colors):
    """
    Plot one or more vectors in a 2D plane, specifying a color for each. 

    Arguments
    ---------
    vectors: list of lists or of arrays
        Coordinates of the vectors to plot. For example, [[1, 3], [2, 2]] 
        contains two vectors to plot, [1, 3] and [2, 2].
    colors: list
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
        
    Example
    -------
    plot_vectors([[1, 3], [2, 2]], ['red', 'blue'])
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    """
    plt.figure()
    plt.axvline(x=0, color='lightgray')
    plt.axhline(y=0, color='lightgray')

    for i in range(len(vectors)):
        x = np.concatenate([[0,0],vectors[i]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=colors[i],)
# eo plot_vectors

"""
plot_vectors([Av, v, Av2, v2], ['blue','lightblue','green','lightgreen'])
plt.xlim(-1, 4)
plt.ylim(-3, 2)
plt.show()
"""
# Eigenvectors in more than 2 dimensions

X = np.array([[25,2,9], [5,26,-5], [3,7,-1]])

lambs_X, V_X = np.linalg.eig(X)
print(V_X)
print(lambs_X)
"""
[[-0.71175736 -0.6501921  -0.34220476]
 [-0.66652125  0.74464056  0.23789717]
 [-0.22170001  0.15086635  0.90901091]]
 
 [29.67623202 20.62117365 -0.29740567]
"""

# confirm Xv = λv
# First of 3 proves
v0 = V_X[:,0]
lamb0 = lambs_X[0]

Xv0 = np.dot(X, v0)
lamb0_v0 = v0 * lamb0

# print("Xv0", Xv0)
# print("lamb0_v0", lamb0_v0)
"""
As required
Xv0      [-21.12227645 -19.77983919  -6.5792208 ]
lamb0_v0 [-21.12227645 -19.77983919  -6.5792208 ]
"""
################################################
# 2x2 Matrix Determinant section
################################################

X = np.array([[4,2],[-5,-3]])
X_det = np.linalg.det(X)
# print(X)
# print(X_det)
"""
[[ 4  2]
 [-5 -3]]

-2.0 
"""
# Now for a non-invertible matrix
# You can see the lines are parallel
N = np.array([[-4,1],[-8,2]])
N_det = np.linalg.det(N)
# print(N)
# print(N_det)
"""
[[-4  1]        
 [-8  2]]
 
0.0
"""
# N_inv = np.linalg.inv(N)
# numpy.linalg.LinAlgError: Singular matrix

################################################
# Matrix Determinant greater that 2x2
################################################
X = np.array([[1,2,4],[2,-1,3],[0,5,1]])
X_det = np.linalg.det(X)
print(X)
print(X_det)
"""
[[ 1  2  4]
 [ 2 -1  3]
 [ 0  5  1]]
 
19.999999999999996
"""












































