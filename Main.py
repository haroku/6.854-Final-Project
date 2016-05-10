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
training_data = artificial_data[:,]
test_data = a

#Train algorithms on training data

adaboost_classifier, (ada_error, ada_iters) =
logitboost_classifier, (logit_error, logit_iters) =
brownboost_classifier, (brown_error, brown_iters) =
#savageboost_classifier = 

#Plot training error of each classifier on training set after this many interations


#Run on validation set and report error

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATA SET WITH GAUSSIAN NOISE
# 1. Generate dataset
# 2. Add different thresholds of Gaussian noise
# 3. Training 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gaussian_noise, gaussian_labels = generate_noise(artificial_data, labels, noise_type = 'gaussian')
uniform_noise, uniform_labels = generate_noise(artificial_data, labels, noise_type = 'mislabel')
mislabel_noise, mislabel_labels = generate_noise(artificial_data, labels, noise_type = 'mislabel')
mislabel_noise, mislabel_labels = generate_noise(artificial_data, labels, noise_type = 'mislabel')
