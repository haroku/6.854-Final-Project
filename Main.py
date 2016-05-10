import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Data import *
from Adaboost import *


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
train_amt = 700
total_amt = num_data

normal, point, artificial_data = generate_data(num_dim, num_data)
training_data = artificial_data[0:train_amt]
training_labels = labels[0:train_amt]

test_data = artificial_data[train_amt: total_amt]
test_labels = labels[train_amt: total_amt]

gaussian_data, gaussian_labels = generate_noise(artificial_data, labels, noise_type = 'gaussian')
uniform_data, uniform_labels = generate_noise(artificial_data, labels, noise_type = 'uniform')
mislabel_data, mislabel_labels = generate_noise(artificial_data, labels, noise_type = 'mislabel')
contradict_data, contradict_labels = generate_noise(artificial_data, labels, noise_type = 'contradict')

g_train = gaussian_data[0:train_amt]
g_train_labels = gaussian_labels[0:train_amt]

g_test = gaussian_test[train_amt: total_amt]
g_test_labels = gaussian_labels[train_amt: total_amt]

u_train = uniform_data[0:train_amt]
u_train_labels = uniform_labels[0:train_amt]

u_test = uniform_data[train_amt: total_amt]
u_test_labels = uniform_labels[train_amt: total_amt]
  
m_train = mislabel_data[0:train_amt]
m_train_labels = mislabel_labels[0:train_amt]

m_test = mislabel_data[train_amt: total_amt]
m_test_labels = mislabel_labels[train_amt: total_amt]

c_train = contradict_date[0:train_amt]
c_train_labels = contradict_labels[0:train_amt]

c_test = contradict_data[train_amt: total_amt]
c_test_labels = contradict_labels[train_amt: total_amt]

def run_trial(training_data, training_labels, num_iters, c, v):
  
  num_trials = 50
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
    
    adaboost_classifier, ada_error = adaboost(training_data, training_labels, num_iters)
    logitboost_classifier, logit_error = logitboost(training_data, training_labels, num_iter)
    brownboost_classifier, brown_error = brownboost(training_data, training_labels, c, v)
    #savageboost_classifier, savage_error =  savageboost(training_data, training_labels, num_iter)
    
    ada_train_errors.append(ada_error)
    logit_train_errors.append(logit_error)
    brown_train_errors.append(brown_error)
    #savage_train_errors.append(savage_error)

    #Run on validation set and report error
    
    ada_test_error = get_error(adaboost_classifier, artificial_data, test_labels)
    logit_test_error = get_error(logitboost_classifier, artificial_data, test_labels)
    brown_test_error = get_error(brownboost_classifier, artificial_data, test_labels)
    #savage_test_error = get_error(savageboost_classifier, artificial_data, test_labels)
  
    ada_test_errors.append(ada_test_error)
    logit_test_errors.append(logit_test_error)
    brown_test_errors.append(brown_test_error)
    #savage_test_errors.append(savage_test_error)
  
  return (ada_test_errors, logit_test_errors, brown_test_errors, ada_train_errors, logit_train_errors, brown_train_errors)
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATA SET WITHOUT NOISE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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


