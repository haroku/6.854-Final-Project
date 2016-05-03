import numpy as np
from WeakLearn import *

'''
Pseudocode reference:
https://www.researchgate.net/publication/221706431_Robust_LogitBoost_and_Adaptive_Base_Class_ABC_LogitBoost
'''
def logitboost(X, X_labels, T, v = 0.1):
  ''' 
  X is the training data to be classified - numpy array with num_data rows and num_dim dimensions
  X_labels are the correct labels of X - numpy array with num_data rows and 1 column
  T is the number of iterations for which to run logitboost
  v = Shrinkage parameter
  '''
  
  #Number of labels 
  K = 2.0
  #Number of data points
  N = len(X)
  
  #P[i,k] is the probability that sample i should be classified as cluster k
  #Initialise probabilities to 1/K
  P = np.array(N,K)
  P = np.fill(1.0/K)
  
  F = np.zeros((N,K))
  
  R = np.zeros((N,K))
  
  for i in xrange(N):
    R[i,K_labels[i]] = 1.0
  
  for i in xrange(T):
    for j in xrange(K):
      W = np.dot(P,1.0-P)
      Z = (1.0*R-P)/W
      #Fit the function f[i;k] by a weighted least-square ofzi;k:toxiwith weights w[i;k]
      #TO DO
      
      F += v*(K-1.0)/K*(f - np.sum(f, axis = 0))
      
    Fexp= np.exp(F)
    P = Fexp/Fexp.sum(axis=0)
  
  return P

  
  
