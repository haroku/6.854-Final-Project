import numpy as np
from scipy.stats import norm

'''Accepts a distribution and outputs a hypothesis with error < 1/2
dist is distribution of the data points
data is the data points
label is their labels
returns a function from points to labels and an error rate as 
(h,eps)
'''


'''
Our weak learners are decision stumps
For each dimension, we sample from a N(0,2) distribution
Then our decision stump classifies on 
which side of that number that dimension of any data point is
We generate a lot of these and look at both side then pick the best one
'''
def generate_stump(dim): 
	# this is a single layer decision tree, cutting the 
	mean=np.random.normal(0,2)
	def pos(x):
		if x[dim]>=mean:
			return 1
		else:
			return -1
	neg =lambda x:-pos(x)
	return (pos,neg)

def generate_probabilistic_stump(dim):
	mean=np.random.normal(0,2)
	def pos(x):
		return norm.cdf(x[dim]-mean,scale=2)
	neg =lambda x:1-pos(x)
	return (pos,neg)

'''
gets the error of a weak learner given a distribution
'''
def get_error(weak_learner,data,labels,dist=None):
	if dist==None:
		dist=np.array([1/float(len(data)) for i in xrange(len(data))])
	(num_data,num_dims)=data.shape
	weak_labels=np.apply_along_axis(weak_learner,1,data)
	error=np.sum(np.dot((1-labels*weak_labels)/2,dist))
	return error


def get_weak_learner(dist,data,labels):
	(num_data,num_dims)=data.shape
	runs=1
	best_err=1
	best_stump=lambda x: 1
	for i in xrange(num_dims):
		for j in xrange(runs):
			(pos,neg)=generate_stump(i)
			pos_error=get_error(pos,data,labels,dist)
			if pos_error<best_err:
				best_stump=pos
				best_err=pos_error
			neg_error=get_error(neg,data,labels,dist)
			if neg_error<best_err:
				best_stump=neg
				best_err=neg_error
	return (best_stump, best_err)
'''
returns a weak learner which ouputs the probability of a label being 1
'''
def get_err_weak_learner(dist,data,labels):
	(h,err)=get_weak_learner(dist,data,labels)
	def g(x):
		out=h(x)
		if out==1:
			return 1-err
		else:
			return err
	return (g,err)

def get_probabilistic_weak_learner(dist,data,labels):
	(num_data,num_dims)=data.shape
	best_err=-num_data
	best_stump=lambda x: 1
	for i in xrange(num_dims):
		(pos,neg)=generate_probabilistic_stump(i)
		pos_error=np.dot(np.apply_along_axis(pos,1,data),labels)
		if pos_error>best_err:
				best_stump=pos
				best_err=pos_error
		neg_error=np.dot(np.apply_along_axis(neg,1,data),labels)
		if neg_error>best_err:
			best_stump=neg
			best_err=neg_error
	return (best_stump, best_err)

if __name__ == "__main__":
	from Noise import *
	#old backward compatible stuff
 #  	(num_data,num_dims)=(1000,10)
 #  	(data,labels)=label_points_old(num_dims,num_data, True, "none", .1)
 #  	dist=np.array([1/float(num_data) for i in xrange(num_data)])
 #  	(stump,err)=get_probabilistic_weak_learner(dist,data,labels)
 #  	print 'err',err
 #  	stump_err=get_error(stump,data,labels,dist)
	# print 'stump_err',stump_err

	#new stuff
	(data,labels,point)=label_points(3,10)
	print 'data',data
	print 'labels',labels.size
	print 'point',point
	dist=np.array([1/float(labels.size) for i in xrange(labels.size)])
	(stump,err)=get_probabilistic_weak_learner(dist,data,labels)
	print 'err', err

