import numpy as np
from WeakLearn import *

'''
Implements the brownBoost algorithm as described in:
http://cseweb.ucsd.edu/~yfreund/papers/brownboost.pdf
'''

#INPUTS:
#X - num_data * num_dim numpy array with data points
#Y - num_data * 1 numpy array of labels
#c - Positive, Real parameter - The laeger it is, the smaller he target error
#v - Positive constant to avoide degenerate cases

def brownBoost(X, Y, c, v):
     time_left = c
     while time_left > 0:
          
