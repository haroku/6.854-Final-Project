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
#3. Train 4 algorithms on test data 
#3b. Plot test error versus iteration 
#4. Compare classifiers performance on validation set
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

artificial_data, labels = 
training_data = artificial_data[0:700]
test_data = artificial_data[700:100]

gaussian_data, gaussian_labels = generate_noise(artificial_data, labels, noise_type = 'gaussian')
uniform_data, uniform_labels = generate_noise(artificial_data, labels, noise_type = 'uniform')
mislabel_data, mislabel_labels = generate_noise(artificial_data, labels, noise_type = 'mislabel')
contradict_data, contradict_labels = generate_noise(artificial_data, labels, noise_type = 'contradict')

gaussian_train = gaussian_data[0:700]
gaussian_test = gaussian_test[700,1000]

uniform_train = uniform_data[0:700]
uniform_test = uniform_data[700:1000]

mislabel_train = mislabel_data[0:700]
mislabel_test = mislabel_data[700:1000]

contradict_train = contradict_date[0:700]
contradict_test = contradict_data[700:1000]

#Train algorithms on training data

adaboost_classifier, ada_error=
logitboost_classifier, logit_error =
brownboost_classifier, brown_error =

#savageboost_classifier, savage_error = 

#Plot training error of each classifier on training set after this many interations


#Run on validation set and report error

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATA SET WITH GAUSSIAN NOISE
# 1. Generate dataset
# 2. Add different thresholds of Gaussian noise
# 3. Training 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


