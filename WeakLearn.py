import numpy as np
'''Accepts a distribution and outputs a hypothesis with error < 1/2
t is iteration number in the boosting algorithm
dist is distribution of the data points
data is the data points
label is their labels
returns a function from points to labels
'''

def get_weak_learner(t,dist,num_data,data,labels):