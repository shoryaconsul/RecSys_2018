#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 23:47:10 2018

@author: fatihkilic
"""
# %%
import copy
import time
import pickle
from tqdm import *
import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.sparse import lil_matrix, coo_matrix
from numpy.random import rand
import sklearn.decomposition as dc
import matplotlib.pyplot as plt

data = sc.sparse.load_npz('sparse_data.npz')
lil_data = data.tolil()
# %%
# Create train test set
test_size = 1000
test_percent = 1.*test_size/lil_data.shape[0]
train, test = train_test_split(
        lil_data, test_size=test_percent, random_state=20)

train = train.tolil()
test = test.tolil()


# %% Cross validation part
# K-Fold Split and Cross-Validation
kf = KFold(n_splits=3)
component_grid = np.array([10,100,200,300])
alpha_grid = np.array([0,0.5,1,10])
l1_ratio_grid = np.array([0,0.5,1])

best_r2 = 0 # to keep track of the best hyper-params
for c in (component_grid):
    for alpha in alpha_grid:
        for l1 in l1_ratio_grid:
            fold_precisions = []
            for train_index, test_index in kf.split(train):
                train_fold, test_fold = train[train_index], train[test_index]


                test_pl = test_fold.tolil(copy=True)  # Test set with masked entries in playlist
                test_size = test_pl.shape[0]
                test_track = np.zeros((test_size,500),dtype=np.int64) # Each row contains tracks 
                test_known = np.zeros((test_size,500),dtype=np.int64) # Unmasked tracks
                test_holdout = np.zeros((test_size,500),dtype=np.int64) # Masked tracks
                

                for u in range(test_size):
                    track_u = test_fold[u,:].nonzero()[1]    # Tracks in u-th playlist
                    test_track[u,:len(track_u)] = track_u
                    mask_lim = np.int(len(track_u)/2)       # Location of last track to be masked
                    test_pl[u,track_u[mask_lim:]] = 0 # Masking tracks
                    test_holdout[u,:len(track_u[mask_lim:])] = track_u[mask_lim:]
                    test_known[u,:mask_lim] = track_u[:mask_lim]
                # Test on the last fold
                R = sc.sparse.vstack([train_fold,test_pl]) # creates the R matrix with masked entries
                n_components = c
                model = dc.NMF(n_components=n_components,solver='cd',l1_ratio=l1,alpha=alpha)
                model.fit(R)
                H = model.components_
                W = model.transform(R)
                W_test = W[-test_pl.shape[0]:,:]
                recon = np.dot(W_test,H)
                r2 = 0
                for i in range(test_size):
                    num_known = test_known[i,:].nonzero()[0].shape[0] # num of unmasked entries in a playlist
                    num_recomm = num_known + 500 # we are going to recommend 500 songs + the location of known entries
                    ind_known = test_known[i,:num_known] # indices of the known entries
                    recon[i,ind_known] = 1 # known entries set to 1
                    max_ind = recon[i,:].argsort()[-num_recomm:][::-1]
                    max_ind = max_ind[num_known:] # select max 500 for recommendation
                    common = [e for e in test_holdout[i,:] if e in max_ind]
                    r_pre = 1.*len(common)/test_holdout[i,:].nonzero()[0].shape[0] # calculate r-precision
                    r2 = r2 + r_pre
                r2 = r2/test_size
                fold_precisions.append(r2)

            fold_precisions = np.asarray(fold_precisions)
            avg_precision = np.sum(fold_precisions)/3.0
            # print 'Alpha {}  L1 Ratio {} - precision Score: {}'.format(alpha,l1,avg_precision)
            print('Num Components {0} Alpha {1}  L1 Ratio {2} - precision Score: {3}'.format(c,alpha,l1,avg_precision))
            if(avg_precision>best_r2):
                best_params = {'alpha':alpha,'l1_ratio':l1,'n_components':c}
                best_r2 = r2


# Model testing after cross-validation
# Alternative Testing - Shorya's Method
test_pl = test.tolil(copy=True)  # Test set with masked entries in playlist
test_size=test_pl.shape[0]
test_track= np.zeros((test_size,500),dtype=np.int64) # Each row contains tracks 
test_known = np.zeros((test_size,500),dtype=np.int64) # Unmasked tracks
test_holdout = np.zeros((test_size,500),dtype=np.int64); # Masked tracks

for u in range(test_size):
    track_u = test[u,:].nonzero()[1]    # Tracks in u-th playlist
    test_track[u,:len(track_u)] = track_u
    mask_lim = np.int(len(track_u)/2)       # Location of last track to be masked
    test_pl[u,track_u[mask_lim:]] = 0 # Masking tracks
    test_holdout[u,:len(track_u[mask_lim:])] = track_u[mask_lim:]
    test_known[u,:mask_lim] = track_u[:mask_lim]

R = sc.sparse.vstack([train,test_pl]) # creates the R matrix with masked entries
model = dc.NMF(n_components=best_params['n_components'],solver='cd',alpha=best_params['alpha'],l1_ratio=best_params['l1_ratio'])
model.fit(R)
H = model.components_
W = model.transform(R)
W_test = W[-test_pl.shape[0]:,:]
recon = np.dot(W_test,H)
r2 = 0
precisions = []
for i in range(test_size):
    num_known = test_known[i,:].nonzero()[0].shape[0] # num of unmasked entries in a playlist
    num_recomm = num_known + 500 # we are going to recommend 500 songs + the location of known entries
    ind_known = test_known[i,:num_known] # indices of the known entries
    recon[i,ind_known] = 1 # known entries set to 1
    max_ind = recon[i,:].argsort()[-num_recomm:][::-1]
    max_ind = max_ind[num_known:] # select max 500 for recommendation
    common = [e for e in test_holdout[i,:] if e in max_ind]
    r_pre = 1.*len(common)/test_holdout[i,:].nonzero()[0].shape[0] # calculate r-precision
    precisions.append(r_pre)
    r2 = r2 + r_pre
r2 = r2/test_size
# Save the results here
precision_list.append(precisions)
# print 'Alpha {}  L1 Ratio {} - precision Score: {}'.format(alpha,l1,r2)
print('Alpha {0}  L1 Ratio {1} - precision Score: {2}'.format(alpha,l1,r2))
result_list = []
result_list.append(r2)
result_list.append(precisions)

# Save the best test results
l1 = best_params['l1_ratio']
alpha = best_params['alpha']
if(l1==0.5):
    l1=5
if(alpha==0.5):
    alpha=5
filename = "comp"+str(int(best_params['n_components']))+"_l1"+str(int(l1))+"_alpha"+str(int(alpha))+".pickle"
with open(filename,'wb') as handle:
    pickle.dump(result_list, handle, protocol=pickle.HIGHEST_PROTOCOL)