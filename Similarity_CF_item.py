5## Collaborative Filtering based approach - faster version

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, find,diags
import timeit
from matplotlib import pyplot as plt
import pandas as pd

start = timeit.default_timer()
#%% Reading data:
with np.load('sparse_data.npz') as data:
#    print(data.keys())
    dat_shape = data.f.shape
#    print(data.f.format)
    col_nz = data.f.col
    row_nz = data.f.row
    dat = data.f.data

del data
UI_matc = csc_matrix((dat,(row_nz,col_nz)),shape=dat_shape,dtype = np.float)
UI_matr = UI_matc.tocsr(copy=True)
track = np.unique(col_nz)
pl = np.unique(row_nz)
num_track = dat_shape[1]
num_pl = dat_shape[0]

#&&& Hyperparametrs:
alph = 0.7
q = 0.5

##%% Create train test set
#test_size = 1000
#test_percent = 1.*test_size/UI_matr.shape[0]
#train_usr, test_usr = train_test_split(UI_matr, test_size=test_percent, random_state=20)
#train_usr = train_usr.tocsr()
#test_usr = test_usr.tocsr()

#%% Create train test set
pl_ind = np.linspace(0,num_pl-1,num=num_pl)
test_size = 1000
test_percent = 1.*test_size/UI_matr.shape[0]
train_usr, test_usr = train_test_split(UI_matr, test_size=test_percent, random_state=20)
train_ind, test_ind = train_test_split(pl_ind, test_size=test_percent, random_state=20) # Get indices of train-test split
train_usr = train_usr.tocsr()
test_usr = test_usr.tocsr()


#%% Taking multiple test playlists
#test_usr = range(100);   # Indices of test playlists
#test_mat = UI_matr[test_usr,:]
#test_ind = np.nonzero(np.sum(test_usr[:,109064:],axis=1))   # When masking right half of matrix
#test_usr = test_usr[test_ind[0],:]
#test_size = test_usr.shape[0]

test_pl = test_usr.tolil(copy=True)  # Test set with masked entries in playlist

test_track= np.zeros((test_size,500),dtype=np.int64) # Each row contains tracks 
test_known = np.zeros((test_size,500),dtype=np.int64) # Unmasked tracks
test_holdout = np.zeros((test_size,500),dtype=np.int64); # Masked tracks

for u in range(test_size):
    track_u = test_usr[u,:].nonzero()[1]    # Tracks in u-th playlist
    test_track[u,:len(track_u)] = track_u
    mask_lim = np.int(len(track_u)/2)       # Location of last track to be masked
    test_pl[u,track_u[mask_lim:]] = 0 # Masking tracks
    test_holdout[u,:len(track_u[mask_lim:])] = track_u[mask_lim:] 
    test_known[u,:mask_lim] = track_u[:mask_lim]

#for u in range(test_size):
#    track_u = test_usr[u,:].nonzero()[1]    # Tracks in u-th playlist
#    test_track[u,:len(track_u)] = track_u
#    mask_lim = np.argmax(track_u>109064)    # First track in right half 
#    test_pl[u,track_u[mask_lim:]] = 0 # Masking tracks
#    test_holdout[u,:len(track_u[mask_lim:])] = track_u[mask_lim:] 
#    test_known[u,:mask_lim] = track_u[:mask_lim]
    
test_pl = test_pl.tocsr()
test_pl.eliminate_zeros()
train_usr = train_usr.tocsc()

#%% Item-based similarity

num_pl_test = test_pl.shape[0]
U_simI = np.zeros((num_pl_test,num_track))

train_col_norm = diags(np.power(1/train_usr.sum(axis=0).A.ravel(),alph))
train_usr_col = train_usr.dot(train_col_norm)
train_row_norm = diags(np.power(1/train_usr.sum(axis=0).A.ravel(),1-alph))
train_usr_row = train_usr.dot(train_row_norm)

I_score_mat = train_usr_row.transpose().dot(train_usr_col)
I_score_mat = I_score_mat.power(q)
#I_score_mat = I_score_mat.tocsc()

for u in tqdm(range(num_pl_test)):
    test_pl_u = test_pl[u,:]
    test_col_ind = test_pl_u.nonzero()[1]     #Find track index already in the test_pl
    
    Iu = I_score_mat[test_col_ind,:]      # Scores for only those songs
    U_simI[u,:] = Iu.sum(axis=0)
    

#%% R-precision
rprec_pl = np.zeros(num_pl_test)

for u in tqdm(range(num_pl_test)): 
    u_sim = U_simI[u,:]
    u_lim = np.argmax(test_known[u,:])+1
    u_sim[test_known[u,:u_lim]] = 0 # Zeroing out known tracks
    Uu_ord = np.argsort(u_sim)[::-1][:500]
    u_hl_lim = np.argmax(test_holdout[u,:])+1
    u_holdout = test_holdout[u,:u_hl_lim]
    rprec_pl[u] = len(np.intersect1d(Uu_ord,u_holdout))/len(u_holdout)
    

#%% Saving data
rprec_arr = np.column_stack((test_ind,rprec_pl))
df_rprec =  pd.DataFrame(rprec_arr,columns=['pl_ind','Rprec'])
df_rprec.to_csv('CF_Item_rprec.csv')

plt.hist(rprec_pl,edgecolor='white', linewidth=0.5)
plt.show()

print("\n",alph,q,rprec_pl.mean())
stop = timeit.default_timer()

print(stop - start) 
    
    
    
    
    
    
    
    
    