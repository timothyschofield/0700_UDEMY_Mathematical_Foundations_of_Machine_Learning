"""

S04L33 Affine Transforms

https://www.udemy.com/course/machine-learning-data-science-foundations-masterclass/learn/lecture/23006968#overview

Section 04 Lesson 33

15 January 2023

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

v = np.array([3,2])

"""
plot_vectors([v], ['lightblue'])
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.show()
"""

# Applying an identity matrix does not transform the vector.
I = np.array([[1,0],[0,1]])

Iv = np.dot(I, v)
print(Iv)
print(Iv == v)
# [3 1]
# [ True  True]

# So this next matrix is non-identity and it flips vectors over the x-axis
E = np.array([[1,0],[0,-1]]) # compare with interpretations of orthogonal matrices in lesson 29

Ev = np.dot(E, v)
"""
plot_vectors([Ev, v], ['darkblue','lightblue'])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
"""
# Flip over y-axis
F = np.array([[-1,0],[0,1]])
Fv = np.dot(F, v)
"""
plot_vectors([Fv, v], ['darkblue','lightblue'])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
"""

# Scaling by *3 in y and fliping across x-axis
Sf = np.array([[1,0],[0,-3]])
Sfv = np.dot(Sf, v)

"""
plot_vectors([Sfv, v], ['darkblue','lightblue'])
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
"""

# A single matrix can apply multiple affine transforms
v = np.array([3,1])
A = np.array([[-1,4],[2,-2]])

Av = np.dot(A,v)
print(Av)
# [5 2]

"""
plot_vectors([Av, v], ['darkblue','lightblue'])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
"""
####################################################
"""
In order to apply a transform to several vectors at once, we can  
concatenate the vectors into a matrix, where each column in that matrix is a separate vector. 
Then whatever linear transformations we apply to V will be independently applied to each column (vector)
"""
v = np.array([3,1])
print("ffff", v.shape)
v2 = np.array([2,1])
v3 = np.array([-3,-1])
v4 = np.array([-1,1])

# Make them (2,1) 2D matices and concat alon axis=1
# axis=0 would stack them
V = np.concatenate( (np.matrix(v).T,
                    np.matrix(v2).T,
                    np.matrix(v3).T,
                    np.matrix(v4).T), axis=1)
# print(V)
"""
[[ 3  2 -3 -1]
 [ 1  1 -1  1]]
"""
AV = np.dot(A,V)
print(AV)
"""
[[ 1  2 -1  5]
 [ 4  2 -4 -4]]
"""
# print(AV.shape) # (2,4)

# I didn't turn it back to a np.array after my mtrx[] manipulations
# Which is why things like reshape did not work
# I must be careful of the type as well as shape
def vectorfy(mtrx, col):
    return np.array(mtrx[:,col]).reshape(-1)


plot_vectors([vectorfy(V, 0),vectorfy(V, 1),vectorfy(V, 2),vectorfy(V, 3),
              vectorfy(AV, 0),vectorfy(AV, 1),vectorfy(AV, 2),vectorfy(AV, 3)], 
             ['lightblue','lightgreen','lightgray','orange','blue','green','gray','red'])

plt.xlim(-4,6)
plt.ylim(-5,5)
plt.show()


