import numpy as np
from WeakLearn import *
from Data import *

'''
Pseudocode & Algorithm reference:
https://www.researchgate.net/publication/221706431_Robust_LogitBoost_and_Adaptive_Base_Class_ABC_LogitBoost
http://arxiv.org/pdf/1203.3491.pdf
http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.lstsq.html Least Squares help
http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6019654
'''
def logitboost(num_data,data, labels, num_iter, K = 2, v = 0.1):
  ''' 
    data is the training data to be classified - numpy array with num_data rows and num_dim dimensions
    labels are the correct labels of data - numpy array with num_data rows and 1 column
    num_iter is the number of iterations for which to run logitboost
    v = Shrinkage parameter
    K = number of labels (default to 2 for binary classification)
  '''
  
  #P[i,k] is the probability that sample i should be classified as cluster k
  #Initialise probabilities to 1/K
  P = np.full([num_data,K],1.0/K) #init num_data xK array with values 1.0/K
  """more print statements
    #number of data points
    print 'num_data', num_data, 'K', K
    print 'initial P'
    print P
    print 'initial data'
    print data
    print 'initial labels'
    print labels
  """
  print 'initial data'
  print data
  print 'initial labels'
  print labels

  K_labels = [i for i in xrange(K)]
  # print K_labels
  
  F = np.zeros((num_data,K)) #big function
  Z = np.zeros((num_data,K))
  W = np.zeros((num_data,K)) #init
  
  R = np.zeros((num_data,K)) #create R

  dist=[np.array([1.0/num_data for i in xrange(num_data)])] #similar to adaboost.py code

  for i in xrange(num_data):
    R[i,int(labels[i]+2)/2] = 1.0 #set up R_i so r_ik = 1 if yi (x_label) = k

  h_t = []
  Func = []
  for j in xrange(K):
    Func.append(lambda x: 0)

  for m in xrange(num_iter): #time step iterations
    f = np.zeros((len(data[0]),K)) #the coefficients of the function
    total_x = np.zeros((len(data),1)) #reset & init at 0s per round
    h_t.append([]) #create matrix within matrix

    for j in xrange(K): #number of classes
      temp_data = data
      for i in xrange(num_data): #number of data points
        W[i][j] = P[i][j]*(1-P[i][j]) #line 4 of alg
        Z[i][j] = (R[i][j]-P[i][j])/W[i][j] #line 5 of alg

      #Fit the function f[i;k] by a weighted least-square ofzi;k:toxiwith weights w[i;k]
        temp_data[i,:] *= W[i][j] #xi with weights wi,k   

      print 'Z at', m,j
      print Z[:,j]   

      # f[:,j] = np.linalg.lstsq(temp_data, Z[:,j])[0] #Fit the function f[i;k] from weighted x_i, z_ik
      (h,eps_t)=get_weak_learner(dist[0],temp_data,labels)
      print 'error of weak learner round', m, eps_t
      if eps_t == 0: #we have a perfect fit
        return h

      """Extra Print Statements
        # print 'least square Z for class', j
        # print Z[:,j]
        # print 'Weights W for class', j
        # print W[:,j]
        # print 'generated temp X for class', j
        # print temp_X
        # print 'generated f_ik'
        # print f[:,j]
        # print predicted_x
        # print total_x
      """
      f_mj = lambda x: v*(K-1.0)/K*(h(x) + -1/K*sum(h_t[m][ite].x for ite in xrange(len(h_t[m])))) #D in IEEE paper
      h_t[m].append(f_mj)
      Func[j] = lambda x: F[j](x) + f_mj(x)

      # predicted_x = np.dot(temp_data,f[:,j]) #get determined values of x
      # predicted_x = np.array([predicted_x]).T #want to be (len(X),1) and not (len(X),) dim matrix
      # total_x += predicted_x

      # for loc in xrange(num_data):
      #   F[loc,j] += v*(K-1.0)/K*(predicted_x[loc] - 1/K*total_x[loc]) #line 7 of alg
      #print 'F', F

    for ite in xrange(num_data):
      funct = lambda x: np.exp(Func[j](x))/sum([h_t[m][ite2](x) for ite2 in xrange(K)])
      print 'i', data[i]
      P[i][j] = funct(data[i]) #TypeError: 'numpy.ndarray' object is not callable??

    print '\n----------\n'
    print 'P after', m, 'rounds'
    print P
    print '\n----------\n'
  
  returnfunct = lambda x: max([Func[j](x) for j in xrange(K)])
  #return argmax_j F_j(x)
  return returnfunct

def robust_logitboost(data, labels, num_iter, v = 0.1, K = 2):
  ''' 
  X is the training data to be classified - numpy array with num_data rows and num_dim dimensions
  labels are the correct labels of X - numpy array with num_data rows and 1 column
  num_iter is the number of iterations for which to run logitboost
  v = Shrinkage parameter
  K = number of labels (default to 2 for binary classification)
  '''
  pass

  
if __name__ == "__main__":
  # import random
  # labels = np.zeros((10,1))
  # for i in xrange(len(labels)):
  #   labels[i] = random.randint(0,1) #randomly tag
  # # print labels
  # normal, point, data =  generate_data(3,10)
  # # P = logitboost(data, labels, 10)

  (num_data,num_dim)=(10,3)
  from Noise import *
  (data,labels)=label_points(num_dim = num_dim,num_data = num_data,class_noise = True,noise_type = "none",p = .1)
  H=logitboost(num_data,data,labels,3)
  # print np.apply_along_axis(H,1,data)
  print "total error", get_error(H,data,labels)
