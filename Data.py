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
  mu_means = 0
  sigma_means = 1
  
  means = np.random.normal(mu_means, sigma_means, num_dim)
  #print means
  #Generate means numpy array
  mu_stdev = 1
  sigma_stdev = 1
  
  stdev = np.random.normal(mu_stdev, sigma_stdev, num_dim)
  stdev = np.absolute(stdev)
  #print stdev
  
  X  = np.zeros((num_data + 1, num_dim))
  
  for j in range(num_dim):
    #print 'mean, stdev', means[j], stdev[j]
    X[:,j] = np.random.normal(means[j], stdev[j], num_data+1)
  
  normal = np.random.normal(0,1,num_dim)
  

  return (normal, X[-1], X[0:-1]) #normal, point, data
  
  

if __name__ == "__main__":
  print generate_data(3,10)