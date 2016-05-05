import numpy as np
from scipy.integrate import ode
from WeakLearn import *
from scipy import special

'''
Implements the brownBoost algorithm as described in:
http://cseweb.ucsd.edu/~yfreund/papers/brownboost.pdf
'''



#INPUTS:
#data - num_data * num_dim numpy array with data points
#labels - num_data * 1 numpy array of labels
#c - Positive, Real parameter - The larger it is, the smaller he target error
#v - Positive constant to avoide degenerate cases

def brown_boost(data, labels, c, v, prints=False):
     (num_data, num_dim) = data.shape
     time_left = float(c)

     alpha_list = []
     h_list = []
     #Initialize matrices
     W = np.zeros(num_data)        #Weight of each example, initialize to 0
     r = np.zeros(num_data)        #Real valued margin for each example; initialize to 0
     roundn=0
     while time_left > 0:
          roundn +=1
          if prints:
               print "time_left", time_left
               print "round" , roundn
          #Associate with each example a positive weight
          W = np.exp(-(r+time_left)**2/c)
          W_norm = W/np.sum(W)                #Normalized distribution of weights

          h, error = get_weak_learner(W_norm, data, labels)

          gamma_i = 1-2*error
          if prints:
               print "error", error
          if error==0:
               return h

          a=r+time_left
          b=np.apply_along_axis(h,1,data)*labels

          (alpha,t)=solve_differential_iterative(a,b,c,v,time_left,gamma_i)

          #Update prediction values
          r = r + alpha*b

          #Update remaining time
          time_left -= t

          #Keep track of classifiers and related alphas
          alpha_list.append(alpha)
          h_list.append(h)


     #Final array of classifiers and weights
     #alpha_list = np.array(alpha_list)
     #h_list = np.array(h_list)

     num_iters=len(h_list)
     if prints:
          print alpha_list
     H = lambda x: np.sign(sum([alpha_list[t]*h_list[t](x) for t in xrange(num_iters)]))
     
     return H
'''
solves the differential equation
given a,b,c, v, gamma_i
a is a num_data length np.array
b is a num_data length np.array
c is a number
v is a number
gamma_i is a number
returns an alpha*,t*
'''
def solve_differential(a,b,c,v,s,gamma_i):
     alpha=min(.05,gamma_i)
     t=alpha**2/3.0
     #alpha=0
     #t=0
     num=np.sum(np.exp(-(a+alpha*b-t)**2/float(c))*b)
     dem=np.sum(np.exp(-(a+alpha*b-t)**2/float(c)))
     gamma=num/float(dem)
     print 'NUM',num
     print 'DEM',dem
     print gamma,alpha,t
     while abs(gamma)>v:
          (alpha,t)=step(alpha,t,a,b,c)
          if alpha<0 or t<0 or t>(2*s):
               return solve_differential_iterative(a,b,c,v,s)
          num=np.sum(np.exp(-(a+alpha*b-t)**2/float(c))*b)
          dem=np.sum(np.exp(-(a+alpha*b-t)**2/float(c)))
          gamma=num/float(dem)
          #print gamma,alpha,t
     return (alpha,t)

'''
Solve the differential using itereative methods
'''
def solve_differential_iterative(a,b,c,v,s,gamma_i):
     alpha=0
     t=0
     gamma=gamma_i
     dx=.01
     while(t<s and gamma >v):
          (alpha,t,gamma)=iterative_step(alpha,t,a,b,c,v,dx,gamma)
     if alpha<0:
          print "negative alpha", alpha
     return (alpha,t)


def iterative_step(alpha,t,a,b,c,v,dx,gamma):
     alpha+=dx/gamma
     t+=dx
     num=np.sum(np.exp(-(a+alpha*b-t)**2/float(c))*b)
     dem=np.sum(np.exp(-(a+alpha*b-t)**2/float(c)))
     gamma=num/float(dem)
     return (alpha,t,gamma)


'''
given alpha_k and t_k returns alpha_k+1,t_k+1
'''
def step(alpha,t,a,b,c):
     d=a+alpha*b-t
     w=np.exp(-d**2/float(c))
     W=np.sum(w)
     U=np.dot(w,d*b)
     B=np.dot(w,b)
     V=np.dot(w,d*(b**2))
     E=np.sum(scipy.special.erf(d/(c**.5))-scipy.special.erf(a/(c**.5)))
     if(V*W-U*B)==0:
          print V,W,U,B
          print a
          print b 
          print d
          print alpha
          print t
     alpha_next=alpha+(c*W*B+((np.pi*c)**.5)*U*E)/float(2*(V*W-U*B))
     t_next=t+(c*B**2+(np.pi*c)**.5*V*E)/float(2*(V*W-U*B))
     return (alpha_next,t_next)

def solve_differential_scipy(a,b,c,v,time_left,gamma_i):
     #Solve differential equation
     def dydt(alpha, t):
          #Define differential equation
          num=np.sum(np.exp(-(a+alpha*b-t)**2/float(c))*b)
          denom=np.sum(np.exp(-(a+alpha*b-t)**2/float(c)))

          return 1/max(num/float(denom),v/2.0)

     alpha,t = 0, 0

     #Define scipy ODE object
     r = ode(dydt).set_integrator('vode', method='bdf')
     r.set_initial_value(alpha,t)

     dt = 0.01
     gamma_big = True
     t_is_time_left = False
     
     while r.successful() and gamma_big and not t_is_time_left:
          r.integrate(r.t+dt)
          if (1/dydt(r.y, r.t)) <= v:
               gamma_big = False
          if r.t >= time_left:
               t_is_time_left = True
          alpha = r.y
          t = r.t
          #print 'in while loop'
     return alpha,t
          
          
if __name__ == '__main__':
     import time
     start_time=time.time()
     sum_error=0
     for i in xrange(100):
          (num_data,num_dim)=(1000,10)
          from Noise import *
          (data,labels)=label_points(num_dim,num_data,True,"none",.1)
          H=brown_boost(data,labels,1,.1)
          print "i", i
          err=get_error(H,data,labels)
          print "final error", err
          sum_error+=err
     total_time=time.time()-start_time
     print "total_time", total_time
     print "average_time", total_time/100.0
     print "total_error", sum_error

     # (num_data,num_dim)=(1000,10)
     # from Noise import *
     # (data,labels)=label_points(num_dim,num_data,True,"none",.1)
     # H=brown_boost(data,labels,1,.1)
     # #print np.apply_along_axis(H,1,data)
     # print "final error", get_error(H,data,labels)

     #100 trials for iterative took 762 seconds, average error .0697       
     
               
