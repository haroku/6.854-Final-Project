import numpy as np
from Data import *
import scipy
from scipy import stats
import random 

"""
	#Data.generate_data returns a normal vector, point on the plane and a matrix of data points 
	#as (normal, point, data) 
	#normal is the normal to the plane
	#it is represented as a length num_dim np array
	#point is a point the plane passes through
	#it is represented as a length num_dim np array
	#data is a matrix of data points
	#each row of data is a datapoint of length num_dim

	#given a set of data, a stdev and a plane specified by a point on it and a normal
	#return the expedted number of points flipped if gaussian noise generated
	#with stdev is added to every data point
"""

#label_points take as input the output from generate_data
#it also takes a boolean class_noise and a string noise_type
#it adds noise to all the points and classifies them
#if class_noise is true only the labels will be changed
#if class_noise is true only the points will be changed
#returns (data,labels)
#parameter p representing the percent of data to be flipped
def label_points(num_dim,num_data):
	(normal, point ,data)=generate_data(num_dim,num_data)
	print 'data.size', data.size
	labels=np.sign(np.dot((data-point),normal))
	return (data,labels,point)

def generate_noise(o_data, o_labels, noise_type, prop, point):
	num_data, num_dim=o_data.shape
	data =np.empty_like(o_data)
	data[:] = o_data
	labels = np.empty_like(o_labels)
	labels[:] = o_labels
	if noise_type == 'mislabel':
		return mislabel_class(data, labels, prop)
	elif noise_type == 'contradict':
		return contradictory_class(data, labels, prop)
	elif noise_type == 'gaussian':
		#Choose attributes to modify
		num_attrs = 3
		attr_list = random.sample(range(0, num_dim), num_attrs)
		for attr in attr_list:
			data, labels = gaussian_attr_noise(data, labels, prop, attr)
		return data, labels
	elif noise_type == 'uniform':
		#Choose attributes to modify
		num_attrs = 3
		attr_list = random.sample(range(0, num_dim), num_attrs)
		for attr in attr_list:
			data, labels = uniform_attr_noise(data, labels, attr, prop, point)
		return data, labels
	else:
		return o_data, o_labels

#given a matrix of data points data, 
#a plane described by a point on it and a normal to it
#a noise type noise_type and
#a parameter p representing the percent of data to be flipped
#returns a noisy version of x
def add_noise(data, point,normal, noise_type, p):
	#old code
	(num_data,num_dim)=data.shape
	if noise_type=="none":
		return data
	if noise_type=="uniform":
		rands=1-2*np.random.binomial(1,p,len(data)) #random vector of 1,-1 with ~p -1s
		noisy_data=np.apply_along_axis(lambda x: x*rands,0,(data-point))+point
		return noisy_data
	if noise_type=="gaussian":
		m=len(data)
		#want to find gaussians to add for p
		stdev=1.0
		min_std=0.0
		max_std=2.0
		exp_flips=exp_errs(stdev,data,point,normal)
		#print "exp_flips:",exp_flips/float(m)
		if exp_flips>p*m:
			while(exp_flips>p*m):
				stdev=stdev/2.0
				#print "stdev",stdev
				exp_flips=exp_errs(stdev,data,point,normal)
				#print "exp_flips:",exp_flips/float(m)
			min_std=stdev
			max_std=stdev*2.0
			stdev=stdev*1.5
		else:
			while(exp_flips<p*m):
				stdev=stdev*2.0
				#print "stdev",stdev
				exp_flips=exp_errs(stdev,data,point,normal)
				#print "exp_flips:",exp_flips/float(m)
			min_std=stdev/2.0
			max_std=stdev
			stdev=stdev*.75
		#print "entering binary search with"
		#print "stdev",stdev
		exp_flips=exp_errs(stdev,data,point,normal)
		#print "exp_flips:",exp_flips/float(m)
		while abs(exp_flips/float(m)-p)>p/10:
			if exp_flips>p*m:
				stdev=(min_std+stdev)/2.0
			else:
				stdev=(max_std+stdev)/2.0
			#print "stdev",stdev
			exp_flips=exp_errs(stdev,data,point,normal)
			#print "exp_flips:",exp_flips/float(m)
		sigma=stdev/(np.sum(normal**2)**.5)
		(w,h)=np.shape(data)
		noise=np.random.normal(0,sigma,w*h).reshape(w,h)
		return noise+data


###GENERATE MISLABELLED CLASS NOISE

def mislabel_class(data, label, prop):
	'''
	Mislabels some proportion, prop, of the original data set
	'''
	num_data, num_dim = data.shape
	
	num_contradictory = int(num_data*prop)
	#Generate a list of random indices into data set
	indices = random.sample(range(0, num_data), num_contradictory)
	for i in indices:
		label[i] = -label[i]
	return (data, label)
###GENERATE CONTRADICTORY LABEL CLASS NOISE

def contradictory_class(data, label, prop):
	'''
	Replaces some proportion prop of the dataset with mislabelled duplicates
	of prop other randomly chosen data points
	'''
	num_data, num_dim = data.shape
	
	num_contradictory = int(num_data*prop)
	#Generate a list of random indices into data set
	indices = random.sample(range(0, num_data), num_contradictory)
	for i in indices:
		data[i] = data[(i+1)%num_dim]
		#Duplicate data point
		label[i] = -label[(i+1)%num_dim]
		#Mislabel duplicate copy
	return (data, label)


###GENERATE GAUSSIAN ATTRIBUTE NOISE

def gaussian_attr_noise(data, labels, prop, attr):
	'''
	Adds Gaussian attribute noise to a data set by selecting an attribute
	and selecting, at random, some proportion, prop, of data points to which
	we add Gaussian noise centered around N(0,sigma)
	'''

	num_data, num_dim = data.shape
	sigma = abs(np.random.normal(1,1))
	num_noisy = int(num_data*prop)
	#Generate a list of random indices into data set
	indices = random.sample(range(0, num_data), num_noisy)
	#Choose num_noisy data points to re-assign uniform values to	
	for i in indices:
		data[i,attr] += np.random.normal(0,sigma)
	return (data, labels)
	
	
###GENERATE UNIFORM ATTRIBUTE NOISE
def uniform_attr_noise(data, labels, attr, prop, point):
	'''
	Adds uniform attribute noise to a data set by selecting an attribute
	and selecting at random some proportion, prop, of data points to which 
	we randomly reassign a value chosen at uniform 	within the domain of that attribute.
	'''
	num_data, num_dim = data.shape
	centre = point[attr]
	num_noisy = int(num_data*prop)
	#Generate a list of random indices into data set
	indices = random.sample(range(0, num_data), num_noisy)
	#Choose num_noisy data points to re-assign uniform values to
	for i in indices:
		data[i,attr] = np.random.uniform(centre-1, centre+1)
	return (data, labels)
		
if __name__ == "__main__":
  (data,labels,point)=label_points(3,10)
  print (data,labels)
  print generate_noise(data,labels,"uniform",.1,point)



