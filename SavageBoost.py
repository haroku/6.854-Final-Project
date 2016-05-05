import numpy as np 

'''
given labelled examples data, labels
return a classifier as a function from points to labels
'''
def savageboost(data, labels):
	(num_data,num_dims)=data.shape
	dist=np.array([1/float(num_data) for i in xrange(num_data)])
	