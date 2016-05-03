import numpy as np

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
	mean=np.random.normal(0,2)
	def pos(x):
		if x[dim]>=mean:
			return 1
		else:
			return -1
	neg =lambda x:-pos(x)
	return (pos,neg)

def get_weak_learner(dist,data,labels):
	(num_data,num_dims)=data.shape
	runs=1
	best_err=1
	best_stump=lambda x: 1
	for i in xrange(num_dims):
		for j in xrange(runs):
			(pos,neg)=generate_stump(i)
			pos_labels=np.apply_along_axis(pos,1,data)
			pos_error=np.sum(np.dot((1-labels*pos_labels)/2,dist))
			if pos_error<best_err:
				best_stump=pos
				best_err=pos_error
			neg_labels=np.apply_along_axis(neg,1,data)
			neg_error=np.sum(np.dot((1-labels*neg_labels)/2,dist))
			if neg_error<best_err:
				best_stump=neg
				best_err=neg_error
	return (best_stump, best_err)


if __name__ == "__main__":
  	from Noise import *
  	(num_data,num_dims)=(1000,10)
  	(data,labels)=label_points(num_dims,num_data, True, "none", .1)
  	dist=np.array([1/float(num_data) for i in xrange(num_data)])
  	(stump,err)=get_weak_learner(dist,data,labels)
  	print err
  	pos_labels=np.apply_along_axis(stump,1,data)
	pos_error=np.sum(np.dot((1-labels*pos_labels)/2,dist))
	print pos_error

