import numpy as np
import scipy as sp
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
     num_data, num_dim = X.shape
     time_left = c*1.0

     #Initialize matrices
     W = np.zeros((num_data, 1))        #Weight of each example, initialize to 0
     R = np.zeros((num_data, 1))        #Real valued margin for each example; initialize to 0
     while time_left > 0:
          #Associate with each example a positive weight
          W = np.exp(-(R+time_left)**2/c)
          W_norm = W/W.sum()                 #Normalized distribution of weights

          h, error = WeakLearn(W_norm, X, Y)

          #Solve differential equation
          def f(t, alpha):
               #Define differential equation
               
          
