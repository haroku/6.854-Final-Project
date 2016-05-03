import numpy as np
from scipy.integrate import ode
from WeakLearn import *

'''
Implements the brownBoost algorithm as described in:
http://cseweb.ucsd.edu/~yfreund/papers/brownboost.pdf
'''

#INPUTS:
#X - num_data * num_dim numpy array with data points
#Y - num_data * 1 numpy array of labels
#c - Positive, Real parameter - The laeger it is, the smaller he target error
#v - Positive constant to avoide degenerate cases

def brownBoost(X, Y, c, v):
     num_data, num_dim = X.shape
     time_left = c*1.0

     alpha_list = []
     h_list = []
     #Initialize matrices
     W = np.zeros((num_data, 1))        #Weight of each example, initialize to 0
     R = np.zeros((num_data, 1))        #Real valued margin for each example; initialize to 0
     while time_left > 0:
          #Associate with each example a positive weight
          W = np.exp(-(R+time_left)**2/c)
          W_norm = W/W.sum()                 #Normalized distribution of weights

          h, error = WeakLearn(W_norm, X, Y)

          #Solve differential equation
          def dtdalpha(t, alpha):
               #Define differential equation
               num = -1/c*(R + alpha*h(X)*Y+time_left-t)**2
               num = np.exp(num)
               denom = np.sum(num)
               num *= h(X)
               num = np.sum(num)

               return num/denom

          t0, alpha0 = 0, 0

          #Define scipy ODE object
          r = ode(f).set_integrator('vode', method='bdf')
          r.set_initial_value(t0, alpha0)

          dalpha = 0.1
          gamma_big = True
          t_is_time_left = False
          
          while r.successful() and gamma_big and not t_is_time_left:
               r.integrate(r.alpha+dalpha)
               if dtdalpha(r.t, r.alpha) <= v:
                    gamma_big = False
               if r.t >= time_left:
                    t_is_time_left = True
               t = r.t
               alpha = r.alpha

          
          #Update prediction values
          R += alpha*H

          #Update remaining time
          time_left -= t

          #Keep track of classifiers and related alphas
          alpha_list.append(alpha)
          h_list.append(h)


     #Final array of classifiers and weights
     alpha_list = np.array(alpha_list)
     h_list = np.array(h_list)
     
     def classify(test_X) :
          classifications = np.array([h(X) for h in h_list ]).T
          sign_output = np.sum(classifications*alpha_list, axis = 1)
          return np.sign(sign_output)

     return classify
          
          

               
     
               
