import numpy as np

'''
#Data returns a normal vector, point on the plane and a matrix of data points 
#Data is formatted as (normal, point, data) 
#normal is the normal to the plane, represented as a length num_dim np array
#point is a point the plane passes through, represented as a length num_dim np array
#data is a matrix of data points, each row of data is a datapoint of length num_dim
'''

#Intilise constants
#num_data = 
#num_dim= 

def generate_data(num_dim, num_data):
  '''Generates num_data points with num_dim dimensions.
  
  '''
  #Generate Means numpy array
  mu_Means = 0
  sigma_Means = 1
  
  Means = np.random.normal(mu_Means, sigma_Means, num_dim)

  #Generate Means numpy array
  mu_Var = 1
  sigma_Var  = 1
  
  Var = np.random.normal(mu_Var, sigma_Var, num_dim)
  Var = np.absolute(Var)
  
  X  = np.zeros((num_data, num_dim))
  
  for j in range(num_dim):
    X[:,j] = np.random.normal(Mu[j], Var[j], num_dim)
  
  normal = np.random.normal(0,1,num_dim)
  
  point = np.random.normal(np.random.normal(mu_Means, sigma_Means, 1), np.random.normal(mu_Var, sigma_Var, 1),num_dim)
  
  return (normal, point, X)
  
  


