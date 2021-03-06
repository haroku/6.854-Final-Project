import numpy as np
from scipy.integrate import ode
from WeakLearn import *
from scipy import special
import scipy
from Adaboost import *

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
     t_list=[]
     #Initialize matrices
     W = np.zeros(num_data)        #Weight of each example, initialize to 0
     r = np.zeros(num_data)        #Real valued margin for each example; initialize to 0
     roundn=0
     outs=np.zeros(num_data)
     errors=[]
     while time_left > 0:
          if len(t_list)>3:
               if time_left>=t_list[-4]:
                    return None
          if prints:
               print "time_left", time_left
               print "round" , roundn
               H = lambda x: np.sign(sum([alpha_list[t]*h_list[t](x) for t in xrange(roundn)]))
               #print "cumulative error", get_error(H,data,labels)
          #Associate with each example a positive weight
          W = np.exp(-(r+time_left)**2/c)
          if np.sum(W)<.00000001:
               print "r",r
               print "W",W
               print "time_left", time_left
               print "c",c
          W_norm = W/np.sum(W)                #Normalized distribution of weights
          h, error = get_weak_learner(W_norm, data, labels)

          gamma_i = 1-2*error
          if prints:
               print "error", error
          if error==0:
               return (h,errors)

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
          t_list.append(time_left)
          roundn +=1
          outs=outs+alpha*np.apply_along_axis(h,1,data)  
          errors.append(np.sum((1-np.sign(outs)*labels)/2)/float(num_data))

     num_iters=len(h_list)
     if prints:
          print alpha_list
     H = lambda x: np.sign(sum([alpha_list[t]*h_list[t](x) for t in xrange(num_iters)]))
     
     return (H,errors)
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
     #print 'NUM',num
     #print 'DEM',dem
     #print gamma,alpha,t
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
     dx=c/100.0
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

def binary_choose_c(data,labels,v):
     
     (A,errors)=adaboost(data,labels,100)
     ada_err=get_error(A,data,labels)
     if ada_err==0:
          return (A,errors)
     last_success=(A,errors)
     c=scipy.special.erfinv(1-ada_err)**2
     #print "adaboost finished", c
     min_c=0.0
     max_c=2*c
     H=brown_boost(data,labels,c,v)
     while H!=None and c<5:
          last_success=H
          c=2*c
          H=brown_boost(data,labels,c,v)
     max_c=c
     c=c/2.0
     #print "binary search init",c
     while (max_c-min_c)>.1:
          H=brown_boost(data,labels,c,v)
          if H==None:
               max_c=c
          else:
               last_success=H
               min_c=c
          c=(max_c+min_c)/2.0
          #print c
     c=min_c
     return last_success
     
          
if __name__ == '__main__':
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
          
          adaboost_classifier, ada_error = binary_choose_c(training_data, training_labels, num_iters)
          print ada_error
          test_data = artificial_data[train_amt: total_amt]
          test_labels = labels[train_amt: total_amt]

          ada_test_error = get_error(adaboost_classifier, test_data, test_labels)
          total_error+=ada_test_error
          print ada_test_error

     print total_error/num_runs
     print time.time()-start
     
               
