import numpy as np
from Data import *
import scipy
from scipy import stats

#Data.generate_data returns a normal vector, point on the plane and a matrix of data points 
#as (normal, point, data) 
#normal is the normal to the plane
#it's represented as a length num_dim np array
#point is a point the plane passes through
#it's represented as a length num_dim np array
#data is a matrix of data points
#each row of data is a datapoint of length num_dim

#label_points take as input the output from generate_data
#it also takes a boolean class_noise and a string noise_type
#it adds noise to all the points and classifies them
#if class_noise is true only the labels will be changed
#if class_noise is true only the points will be changed
#returns (data,labels)

def exp_errs(stdev,data,point,normal):
	return np.sum(scipy.stats.norm.cdf(np.dot(data-point,normal),0,stdev))

#given a matrix of data points data, 
#a plane described by a point on it and a normal to it
#a noise type noise_type and
#a parameter p representing the percent of data to be flipped
#returns a noisy version of x
def add_noise(data, point,normal, noise_type, p):
	if noise_type=="none":
		return data
	if noise_type=="uniform":
		rands=1-2*np.random.binomial(1,p,len(data)) #random vector of 1,-1 with ~p -1s
		noisy_data=(data-point)*rands+point
		return noisy_data
	if noise_type=="gaussian":
		
		#want to find gaussians to add for p
		stdev=1.0
		min_std=0.0
		max_std=2.0
		exp_flips=exp_errs(stdev,data,point,normal)
		if exp_flips>p*m:
			while(exp_flips>p*m):
				stdev=stdev/2.0
				exp_flips=exp_errs(stdev,data,point,normal)
			min_std=stdev
			max_std=stdev*2.0
			stdev=stdev*1.5
		else:
			while(exp_flips<p*m):
				stdev=stdev*2.0
				exp_flips=exp_errs(stdev,data,point,normal)
			min_std=stdev/2.0
			max_std=stdev
			stdev=stdev*.75
		exp_flips=exp_errs(stdev,data,point,normal)
		while abs(exp_flips-p*m)>1:
			if exp_flips>p*m:
				stdev=(min_std+stdev)/2.0
			else:
				stdev=(max_std+stdev)/2.0
			exp_flips=exp_errs(stdev,data,point,normal)
		sigma=stdev/(np.sum(normal**2)**.5)
		(w,h)=np.shape(data)
		noise=np.random.normal(0,sigma,w*h).reshape(w,h)
		return noise+data




def label_points(num_dim,num_data, class_noise, noise_type):
	(normal, point ,data)=generate_data(num_dim,num_data)
	if class_noise:
		noisy_data=add_noise(data,noise_type)
		labels=np.sign(np.dot((data-point),normal))
		return (data,labels)
	else:
		labels=np.sign(np.dot((data-point),normal))
		out_data=add_noise(data,noise_type)
		return (out_data,labels)


#label_points(3,10,True,"none")



