import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import special

import Adaboost as ab
import WeakLearn as wl
import LogitBoost2 as lb 
import brownBoost as bb
from Noise import *


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#DATA SET WITHOUT NOISE
#1. Generate data set without noise
#2. Set aside 2/3 for validation
#3. Train algorithms on test data 
#3b. Plot test error versus iteration 
#4. Compare classifiers performance on validation set
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_dim = 15
num_data = 1000

artificial_data,labels, pt = label_points(num_dim,num_data)

gaussian_data, gaussian_labels = generate_noise(artificial_data, labels, noise_type = 'gaussian', prop=0.1, point = pt)
uniform_data, uniform_labels = generate_noise(artificial_data, labels, noise_type = 'uniform', prop=0.1, point = pt)
mislabel_data, mislabel_labels = generate_noise(artificial_data, labels, noise_type = 'mislabel', prop=0.1, point = pt)
contradict_data, contradict_labels = generate_noise(artificial_data, labels, noise_type = 'contradict', prop=0.1, point = pt)


def run_trials(data, labels, num_iters, v):
  
  train_amt = 700
  num_data, num_dim = data.shape
  training_data = data[0:train_amt]
  training_labels = labels[0:train_amt]

  test_data = data[train_amt: num_data]
  test_labels = labels[train_amt: num_data]

  num_trials = 5
  ada_test_errors = []
  logit_test_errors = []
  brown_test_errors = []
  #savage_test_errors = []
  
  ada_train_errors = []
  logit_train_errors = []
  brown_train_errors = []
  #savage_train_errors = []
  
  for i in range(num_trials):
    #Train algorithms on training data
    
    adaboost_classifier, ada_error = ab.adaboost(training_data, training_labels, num_iters)
    logitboost_classifier, logit_error = lb.logitboost(training_data, training_labels, num_iters)
    brownboost_classifier, brown_error = bb.binary_choose_c(training_data, training_labels, v)
    #savageboost_classifier, savage_error =  savageboost(training_data, training_labels, num_iter)
    
    ada_train_errors.append(ada_error)
    logit_train_errors.append(logit_error)
    brown_train_errors.append(brown_error)
    #savage_train_errors.append(savage_error)
    print 'Doing some cool things'
    #Run on validation set and report error
    
    ada_test_error = wl.get_error(adaboost_classifier, test_data, test_labels)
    logit_test_error = wl.get_error(logitboost_classifier, test_data, test_labels)
    brown_test_error = wl.get_error(brownboost_classifier, test_data, test_labels)
    #savage_test_error = get_error(savageboost_classifier, test_data, test_labels)
  
    ada_test_errors.append(ada_test_error)
    logit_test_errors.append(logit_test_error)
    brown_test_errors.append(brown_test_error)
    #savage_test_errors.append(savage_test_error)
  
  return (ada_test_errors, logit_test_errors, brown_test_errors, ada_train_errors, logit_train_errors, brown_train_errors)
  

num_iters=20
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATA SET WITHOUT NOISE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ada_test_errors, logit_test_errors, brown_test_errors, ada_train_errors, logit_train_errors, brown_train_errors = run_trials(artificial_data, labels, num_iters, .1)
print ada_test_errors
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATA SET WITH GAUSSIAN NOISE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATA SET WITH UNIFORM NOISE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATA SET WITH MISLABEL NOISE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATA SET WITH CONTRADICTORY NOISE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


