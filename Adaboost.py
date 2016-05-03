import numpy as np
from WeakLearn import *




#given labelled examples data, labels
#return a classifier as a function from points to labels
# example: adaboost(5,np.array([0,1,2,3,4]), np.array([1,-1,-1,1,1]), 7)
def adaboost(num_data, data, labels, num_iters):
	'''
	num_data = Number of examples
	data = Training Set
	labels = Classification of training data
	num_iters = num iterations to run Adaboost
	'''
	dist=[np.array([1.0/num_data for i in xrange(num_data)])]
	alpha=[0 for i in xrange(num_iters)]
	h_t=[]
	for t in xrange(num_iters):
		(h,eps_t)=get_weak_learner(dist[t],data,labels)
		if eps_t==0:
			return h
		h_t.append(h)
		alpha[t]=.5*np.log((1.0-eps_t)/eps_t)
		non_normed=dist[t]*np.exp(-np.apply_along_axis(h,1,data)*labels*alpha[t])
		normed=non_normed/(np.sum(non_normed))
		dist.append(normed)
	H=lambda x: np.sign(sum([alpha[t]*h_t[t](x) for t in xrange(num_iters)]))
	return H


if __name__ == '__main__':
	m=1000
	T=100
	r= np.sign(np.random.rand(m)-.5)
	H= adaboost(m,np.zeros(m),r, T)
	print (1.0-np.sum(H*r)/1000.0)/2
