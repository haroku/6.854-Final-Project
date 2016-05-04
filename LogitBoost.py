import numpy as np
from WeakLearn import *
from Data import *

'''
Pseudocode reference:
https://www.researchgate.net/publication/221706431_Robust_LogitBoost_and_Adaptive_Base_Class_ABC_LogitBoost
http://arxiv.org/pdf/1203.3491.pdf
'''
def logitboost(X, X_labels, T, v = 0.1, K = 2):
  ''' 
  X is the training data to be classified - numpy array with num_data rows and num_dim dimensions
  X_labels are the correct labels of X - numpy array with num_data rows and 1 column
  T is the number of iterations for which to run logitboost
  v = Shrinkage parameter
  K = number of labels (default to 2 for binary classification)
  '''
  
  #Number of data points
  N = len(X)
  # print 'N', N, 'K', K
  
  #P[i,k] is the probability that sample i should be classified as cluster k
  #Initialise probabilities to 1/K
  P = np.full([N,K],1.0/K) #init NxK array with values 1.0/K
  print P
  K_labels = [i for i in xrange(K)]
  # print K_labels
  
  F = np.zeros((N,K))
  Z = np.zeros((N,K))
  W = np.zeros((N,K)) #init
  
  R = np.zeros((N,K)) #create R
  for i in xrange(N):
    R[i,int(X_labels[i])] = 1.0 #set up R_i so r_ik = 1 if yi (x_label) = k

  for m in xrange(T): #time step iterations
    for i in xrange(N):
      for j in xrange(K):
        W[i][j] = P[i][j]*(1-P[i][j]) #line 4 of alg
        Z[i][j] = (R[i][j]-P[i][j])/W[i][j] #line 5 of alg

        # W = np.dot(P,1.0-P) #sitara's code
        # Z = (1.0*R-P)/W #sitara's code
        #Fit the function f[i;k] by a weighted least-square ofzi;k:toxiwith weights w[i;k]
        f = 0  #TO DO
        
        # F += v*(K-1.0)/K*(f - np.sum(f, axis = 0)) #sitara's code
        F[i][j] += v*(K-1.0)/K*(f - 1/K*np.sum(f, axis = 0))
        
    Fexp= np.exp(F)
    P = Fexp/Fexp.sum(axis=0)
    print P
  
  return P

def robust_logitboost(X, X_labels, T, v = 0.1, K = 2):
  pass

  
if __name__ == "__main__":
  import random
  X_labels = np.zeros((10,1))
  for i in xrange(len(X_labels)):
    X_labels[i] = random.randint(0,1) #randomly tag
  # print X_labels
  normal, point, data =  generate_data(3,10)
  P = logitboost(data, X_labels, 3)
  print 'returned answer'
  print P
