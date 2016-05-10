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
	for i in xrange(num_iter):
		z=(y-p)/(p*(1-p))
		w=p*(1-p)
		(h,err)=get_weak_logit_learner(data,w,z)
		F=lambda x:sum([f_m[j](x) for j in xrange(i)])
		F_x=np.apply_along_axis(F,1,data)

		p=np.exp(F_x)/(np.exp(F_x)+np.exp(-F_x))
		f_m.append(h)
		out=out + np.apply_along_axis(h,1,data)
		errors.append(np.sum((1-np.sign(outs)*labels)/2)/float(num_data))

	return lambda x:np.sign(sum([f_m[i](x) for i in xrange(num_iter)]))



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
	(num_data,num_dim)=(1000,10)
	from Noise import *
	(data,labels)=label_points(num_dim = num_dim,num_data = num_data,class_noise = True,noise_type = "none",p = .1)
	H,errors=logitboost(data,labels,10)
	print "total error", get_error(H,data,labels)

