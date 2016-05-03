import numpy as np
from WeakLearn import *
#weights alpha[t] (list)
#distr D[t][x] (list of np.arrays)
#weak learner h[x] (np.array)
#convention A(x)=A[i] where x[i]=x


#weak_learner function: given a time step and distribution it returns a weak learner
#example: get_weak_learner(5,np.array([1.0/m for i in xrange(m)]),m, x, fx)
def get_weak_learner(t,D,m,x,fx):	

	r=np.sign(np.random.rand(m)-.5)
	#print r
	err=(1.0-np.sum(D*fx*r))/2.0
	if err>.5:
		return (-r,1-err)
	else:
		return (r,err)


#given labelled examples x[j],hx[j]
#return H[x]
# example: adaboost(5,np.array([0,1,2,3,4]), np.array([1,-1,-1,1,1]), 7)
def adaboost(m, x, fx, T):
	'''
	m = Number of examples
	x = Training Set
	fx = Classification of training data
	T = num iterations to run Adaboost
	'''
	D=[np.array([1.0/m for i in xrange(m)])]
	alpha=[0 for i in xrange(T)]
	h_t=[]
	for t in xrange(T):
		(h,eps_t)=get_weak_learner(t,D[t],m,x,fx)
		if eps_t==0:
			return h
		h_t.append(h)
		alpha[t]=.5*np.log((1.0-eps_t)/eps_t)
		non_normed=D[t]*np.exp(-h*fx*alpha[t])
		normed=non_normed/(np.sum(non_normed))
		D.append(normed)
	alpha=np.array(alpha)
	#print alpha
	h_t=np.array(h_t)
	#print h_t
	H=np.sign(np.dot(np.transpose(h_t),alpha))
	return H

m=1000
T=100
r= np.sign(np.random.rand(m)-.5)
H= adaboost(m,np.zeros(m),r, T)
print (1.0-np.sum(H*r)/1000.0)/2
