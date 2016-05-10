import numpy as np 
from WeakLearn import *

'''
given labelled examples data, labels
return a classifier as a function from points to labels
'''
def savageboost(data, labels, M):
	(num_data,num_dims)=data.shape
	dist=np.zeros(num_data)
	dist.fill(1/float(num_data))
	#print dist
	g=[]
	for m in xrange(M):
		#print dist
		(h,err)=get_probabilistic_weak_learner(dist,data,labels)
		#print np.apply_along_axis(h,1,data)
		print err
		# if err==0:
		# 	return lambda x: h(x)*2-1

		def Gm(x):
			#print h(x)
			return .5 * np.log(h(x)/(1.0-h(x)))

		#print np.apply_along_axis(Gm,1,data)
		#print labels*np.apply_along_axis(Gm,1,data)
		g.append(Gm)

		dist=dist*np.exp(labels*np.apply_along_axis(Gm,1,data))
		dist=np.minimum(np.maximum(dist,.00000001),100000000)
		dist=dist/float(sum(dist))
		#print sum(dist)
	#print g
	def out(x):
		#print [g[i](x) for i in xrange(M)]
		return np.sign(sum([g[i](x) for i in xrange(M)]))

	return out



if __name__ == '__main__':
    (num_data,num_dim)=(1000,10)
    from Noise import *
    (data,labels)=label_points(num_dim,num_data,True,"none",.1)
    H=savageboost(data,labels,10)
    #print np.apply_along_axis(H,1,data)
    print "final error", get_error(H,data,labels)

