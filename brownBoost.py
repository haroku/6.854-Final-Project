import numpy as np
from scipy.integrate import ode
from WeakLearn import *

'''
Implements the brownBoost algorithm as described in:
http://cseweb.ucsd.edu/~yfreund/papers/brownboost.pdf
'''

#INPUTS:
#data - num_data * num_dim numpy array with data points
#labels - num_data * 1 numpy array of labels
#c - Positive, Real parameter - The larger it is, the smaller he target error
#v - Positive constant to avoide degenerate cases

def brown_boost(data, labels, c, v):
     (num_data, num_dim) = data.shape
     time_left = float(c)

     alpha_list = []
     h_list = []
     #Initialize matrices
     W = np.zeros(num_data)        #Weight of each example, initialize to 0
     R = np.zeros(num_data)        #Real valued margin for each example; initialize to 0
     while time_left > 0:
          #Associate with each example a positive weight
          W = np.exp(-(R+time_left)**2/c)
          W_norm = W/np.sum(W)                #Normalized distribution of weights

          h, error = get_weak_learner(W_norm, data, labels)

          #Solve differential equation
          def dydt(y, t):
               #Define differential equation
               num = -1/c*(R + t*np.apply_along_axis(h,1,data)*labels+time_left-y)**2
               num = np.exp(num)
               denom = np.sum(num)
               num *= np.apply_along_axis(h,1,data)
               num = np.sum(num)

               return num/denom

          y0, t0 = 0, 0

          #Define scipy ODE object
          r = ode(dydt).set_integrator('vode', method='bdf')
          r.set_initial_value(y0,t0)

          dt = 0.1
          gamma_big = True
          t_is_time_left = False
          
          while r.successful() and gamma_big and not t_is_time_left:
               r.integrate(r.t+dt)
               if dydt(r.y, r.t) <= v:
                    gamma_big = False
               if r.y >= time_left:
                    t_is_time_left = True
               y = r.y
               alpha = r.t
               #print 'in while loop'

          
          #Update prediction values
          #print data.shape
          #print R.shape
          #print alpha
          #print np.apply_along_axis(h,1,data).reshape(num_data,1).shape
          R = R + alpha*np.apply_along_axis(h,1,data)*labels

          #Update remaining time
          time_left -= y

          #Keep track of classifiers and related alphas
          alpha_list.append(alpha)
          h_list.append(h)


     #Final array of classifiers and weights
     #alpha_list = np.array(alpha_list)
     #h_list = np.array(h_list)

     num_iters=len(h_list)
     print alpha_list
     H = lambda x: np.sign(sum([alpha_list[t]*h_list[t](x) for t in xrange(num_iters)]))
     
     return H
          
          
if __name__ == '__main__':
     (num_data,num_dim)=(1000,10)
     from Noise import *
     (data,labels)=label_points(num_dim,num_data,True,"none",.1)
     H=brown_boost(data,labels,10,.1)
     print np.apply_along_axis(H,1,data)
     print get_error(H,data,labels)
               
     
               
