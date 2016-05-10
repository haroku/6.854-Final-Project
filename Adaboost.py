import numpy as np
from WeakLearn import *




#given labelled examples data, labels
#return a classifier as a function from points to labels
# example: adaboost(5,np.array([0,1,2,3,4]), np.array([1,-1,-1,1,1]), 7)
def adaboost(data, labels, num_iters):
	'''
		num_data = Number of examples
		data = Training Set
		labels = Classification of training data
		num_iters = num iterations to run Adaboost
	'''
	(num_data,num_dim)=data.shape
	dist=[np.array([1.0/num_data for i in xrange(num_data)])]
	#print dist
	alpha=[0 for i in xrange(num_iters)]
	h_t=[]
	errors=[]
	out=np.zeros(num_data)
	for t in xrange(num_iters):
		(h,eps_t)=get_weak_learner(dist[t],data,labels)
		#print 'error of weak learner round', t, eps_t
		if eps_t==0: #we have a perfect fit
			return h
		h_t.append(h)
		alpha[t]=.5*np.log((1.0-eps_t)/eps_t)
		non_normed=dist[t]*np.exp(-np.apply_along_axis(h,1,data)*labels*alpha[t])
		normed=non_normed/(np.sum(non_normed))
		dist.append(normed)
		out=out+alpha[t]*np.apply_along_axis(h,1,data)
		errors.append(np.sum((1-np.sign(out)*labels)/2)/float(num_data))



	H=lambda x: np.sign(sum([alpha[t]*h_t[t](x) for t in xrange(num_iters)]))
	return (H,errors)


if __name__ == '__main__':
	from Noise import *
	num_dim = 15
	num_data = 1000
	train_amt = 700
	total_amt = num_data
	num_iters=20

	artificial_data,labels, pt = label_points(num_dim,num_data)
	training_data = artificial_data[0:train_amt]
	training_labels = labels[0:train_amt]
	
	adaboost_classifier, ada_error = adaboost(training_data, training_labels, num_iters)
	print ada_error
	test_data = artificial_data[train_amt: total_amt]
	test_labels = labels[train_amt: total_amt]

	ada_test_error = get_error(adaboost_classifier, test_data, test_labels)
	print ada_test_error