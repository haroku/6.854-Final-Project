import numpy as np
from WeakLearn import *
from Data import *

'''
Pseudocode reference:
https://www.researchgate.net/publication/221706431_Robust_LogitBoost_and_Adaptive_Base_Class_ABC_LogitBoost
http://arxiv.org/pdf/1203.3491.pdf
http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.lstsq.html Least Squares help
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
  # print 'initial P'
  # print P

  # print 'initial X'
  # print X

  # print 'initial X_labels'
  # print X_labels

  K_labels = [i for i in xrange(K)]
  # print K_labels
  
  F = np.zeros((N,K)) #big function
  Z = np.zeros((N,K))
  W = np.zeros((N,K)) #init
  
  R = np.zeros((N,K)) #create R
  for i in xrange(N):
    R[i,int(X_labels[i]+2)/2] = 1.0 #set up R_i so r_ik = 1 if yi (x_label) = k

  for m in xrange(T): #time step iterations
    f = np.zeros((len(X[0]),K)) #the coefficients of the function
    total_x = np.zeros((len(X),1)) #reset & init at 0s per round

    for j in xrange(K): #number of classes
      temp_X = X
      for i in xrange(N): #number of data points
        W[i][j] = P[i][j]*(1-P[i][j]) #line 4 of alg
        Z[i][j] = (R[i][j]-P[i][j])/W[i][j] #line 5 of alg

      #Fit the function f[i;k] by a weighted least-square ofzi;k:toxiwith weights w[i;k]
        temp_X[i,:] *= W[i][j] #xi with weights wi,k      

      f[:,j] = np.linalg.lstsq(temp_X, Z[:,j])[0] #Fit the function f[i;k] from weighted x_i, z_ik

      # print 'least square Z for class', j
      # print Z[:,j]
      # print 'Weights W for class', j
      # print W[:,j]
      # print 'generated temp X for class', j
      # print temp_X
      # print 'generated f_ik'
      # print f[:,j]

      predicted_x = np.dot(temp_X,f[:,j]) #get determined values of x
      predicted_x = np.array([predicted_x]).T #want to be (len(X),1) and not (len(X),) dim matrix
      
      # print predicted_x
      # print total_x

      total_x += predicted_x

      for loc in xrange(N):
        F[loc,j] += v*(K-1.0)/K*(predicted_x[loc] - 1/K*total_x[loc]) #line 7 of alg
      
      print 'F', F
      

    Fexp= np.exp(F)
    P = Fexp/Fexp.sum(axis=0)
    print '\n----------\n'
    print 'P after', m, 'rounds'
    print P
    print '\n----------\n'
  
  return P

def robust_logitboost(X, X_labels, T, v = 0.1, K = 2):
  ''' 
  X is the training data to be classified - numpy array with num_data rows and num_dim dimensions
  X_labels are the correct labels of X - numpy array with num_data rows and 1 column
  T is the number of iterations for which to run logitboost
  v = Shrinkage parameter
  K = number of labels (default to 2 for binary classification)
  '''

  
if __name__ == "__main__":
  (num_data,num_dim)=(10,3)
  from Noise import *
  (data,labels)=label_points(num_dim,num_data,True,"none",.1)
  H=logitboost(data,labels,3)
  #print np.apply_along_axis(H,1,data)
  print "final error", get_error(H,data,labels)
