from WeakLearn import *

def logitboost(data, labels, num_iter):
	''' 
	data is the training data to be classified - numpy array with num_data rows and num_dim dimensions
	labels are the correct labels of data - numpy array with num_data rows and 1 column
	num_iter is the number of iterations for which to run logitboost
	v = Shrinkage parameter
	K = number of labels (default to 2 for binary classification)
	'''
	(num_data,num_dim)=data.shape
	z=np.zeros(num_data)
	w=np.full(num_data,1/float(num_data))
	p=np.full(num_data,.5)
	y=(labels+1)/2.0
	f_m=[]
	F=lambda x:0
	errors=[]
	out=np.zeros(num_data)
	for i in xrange(num_iter):
		z=(y-p)/(p*(1-p))
		w=p*(1-p)
		(h,err)=get_weak_logit_learner(data,w,z)
		f_m.append(h)
		F=lambda x:sum([f_m[j](x)*.5 for j in xrange(i+1)])
		F_x=np.apply_along_axis(F,1,data)

		p=np.exp(F_x)/(np.exp(F_x)+np.exp(-F_x))
		p=np.minimum(.9999,np.maximum(.0001,p))
		#print p
		out=out + np.apply_along_axis(h,1,data)
		errors.append(np.sum((1-np.sign(out)*labels)/2)/float(num_data))
		# if len(errors)>10:
		# 	if -errors[-1]+sum(errors[-11:-1])/5.0<.000001:
		# 		return (lambda x:np.sign(sum([f_m[t](x) for t in xrange(i)])),errors)

	return (lambda x:np.sign(sum([f_m[i](x) for i in xrange(num_iter)])),errors)



def get_weak_logit_learner(data, w,z):
	(num_data,num_dims)=data.shape
	best_err=np.dot(w,z**2)
	best_stump=lambda x: 0
	for i in xrange(num_dims):
		(pos,neg)=generate_stump(i)
		pos_error=get_least_square_error(pos,data,w,z)
		if pos_error<best_err:
			best_stump=pos
			best_err=pos_error
		neg_error=get_least_square_error(neg,data,w,z)
		if neg_error<best_err:
			best_stump=neg
			best_err=neg_error
	return (best_stump, best_err)

def get_least_square_error(f,data,w,z):
	f_x=np.apply_along_axis(f,1,data)
	return np.dot(w,(f_x-z)**2)

if __name__ == "__main__":
	from Noise import *
	num_dim = 10
	num_data = 500
	train_amt = 300
	total_amt = num_data
	num_iters=50
	num_runs=20

	total_error=0.0

	import time

	start=time.time()

	for i in xrange(num_runs):
		artificial_data,labels, pt = label_points(num_dim,num_data)
		training_data = artificial_data[0:train_amt]
		training_labels = labels[0:train_amt]
		
		adaboost_classifier, ada_error = logitboost(training_data, training_labels, num_iters)
		print ada_error
		test_data = artificial_data[train_amt: total_amt]
		test_labels = labels[train_amt: total_amt]

		ada_test_error = get_error(adaboost_classifier, test_data, test_labels)
		total_error+=ada_test_error
		print ada_test_error

	print total_error/num_runs
	print time.time()-start

