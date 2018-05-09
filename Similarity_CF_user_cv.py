## Collaborative Filtering based approach
## Fastest version for cross-validation

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
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

alph_arr = np.linspace(0.3,0.7,5)
q_arr = np.linspace(1,5,3)

#alph = 0.8
#q = 3


#%% Create train test set
pl_ind = np.linspace(0,num_pl-1,num=num_pl)
test_size = 1000
test_percent = 1.*test_size/UI_matr.shape[0]
train_usr, test_usr = train_test_split(UI_matr, test_size=test_percent, random_state=20)
train_ind, test_ind = train_test_split(pl_ind, test_size=test_percent, random_state=20) # Get indices of train-test split
train_usr = train_usr.tocsr()
test_usr = test_usr.tocsr()

#%% Cross-validation
rprec_mat = np.zeros((len(alph_arr),len(q_arr)))   # Rprecision scores for different hyperparameters

kf = KFold(n_splits = 3)    # Splits for cross validation

for train_cv, test_cv in kf.split(train_usr):
    train_cv_usr = train_usr[train_cv,:]
    test_cv_usr = train_usr[test_cv,:]
    test_cv_size = test_cv_usr.shape[0]
    
    test_cv_pl = test_cv_usr.tolil(copy=True)  # Test set with masked entries in playlist
    
    test_cv_track= np.zeros((test_cv_size,500),dtype=np.int64) # Each row contains tracks 
    test_cv_known = np.zeros((test_cv_size,500),dtype=np.int64) # Unmasked tracks
    test_cv_holdout = np.zeros((test_cv_size,500),dtype=np.int64); # Masked tracks
    
    for u in range(test_cv_size):
        track_u = test_cv_usr[u,:].nonzero()[1]    # Tracks in u-th playlist
        test_cv_track[u,:len(track_u)] = track_u
        mask_lim = np.int(len(track_u)/2)       # Location of last track to be masked
        test_cv_pl[u,track_u[mask_lim:]] = 0 # Masking tracks
        test_cv_holdout[u,:len(track_u[mask_lim:])] = track_u[mask_lim:] 
        test_cv_known[u,:mask_lim] = track_u[:mask_lim]
    
        
    test_cv_pl = test_cv_pl.tocsr()
    test_cv_pl.eliminate_zeros()

    U_cv_sim = np.zeros((test_cv_pl.shape[0],num_track))    # Matrix for user similariity scores
    
    test_cv_norm = 1/test_cv_pl.sum(axis=1).A.ravel()
    train_cv_norm = 1/train_cv_usr.sum(axis=1).A.ravel()
    
    # Running for different hyperparameters
    for alph in alph_arr:
        test_cv_pl_norm = diags(np.power(test_cv_norm,1-alph)).dot(test_cv_pl)
        train_cv_usr_norm = diags(np.power(train_cv_norm,alph)).dot(train_cv_usr)
            
        U_cv_score = train_cv_usr_norm.dot(test_cv_pl_norm.transpose()) # Compute similarity scores without power
        train_cv_usr_c = train_cv_usr.tocsc()

        for q in q_arr:
            U_cv_score_mat = U_cv_score.power(q)
            
            print('Alpha = ',alph, ', q = ',q)
            for i in tqdm(range(num_track)):
                ti = track[i]
                Ui = train_cv_usr_c[:,ti].nonzero()[0] # Users with the song
                U_cv_sim[:,i] = U_cv_score_mat[Ui,:].sum(axis=0).A.ravel()
            
            rprec_cv_pl = np.zeros(test_cv_size)
            
            for u in tqdm(range(test_cv_size)): 
                u_sim = U_cv_sim[u,:]
                u_lim = np.argmax(test_cv_known[u,:])+1
                u_sim[test_cv_known[u,:u_lim]] = 0 # Zeroing out known tracks
                Uu_ord = np.argsort(u_sim)[::-1][:500]
                u_hl_lim = np.argmax(test_cv_holdout[u,:])+1
                u_holdout = test_cv_holdout[u,:u_hl_lim]
                rprec_cv_pl[u] = len(np.intersect1d(Uu_ord,u_holdout))/len(u_holdout)
            
            rprec_mat[np.where(alph_arr==alph),np.where(q_arr==q)] = rprec_mat[np.where(alph_arr==alph),np.where(q_arr==q)] + np.mean(rprec_cv_pl)
            
#%% Taking multiple test playlists
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

    
test_pl = test_pl.tocsr()
test_pl.eliminate_zeros()

#%% User-based similarity
alph_q_ind = np.argmax(rprec_mat)
alph_q_row, alph_q_col = np.divmod(alph_q_ind,len(q_arr))

alph = alph_arr[alph_q_row]
q = q_arr[alph_q_col]

num_pl_test = test_pl.shape[0]
U_sim = np.zeros((num_pl_test,num_track))

test_norm = diags(np.power(1/test_pl.sum(axis=1).A.ravel(),1-alph))
test_usr_norm = test_norm.dot(test_pl)

train_norm = diags(np.power(1/train_usr.sum(axis=1).A.ravel(),alph))
train_usr_norm = train_norm.dot(train_usr)

U_score_mat = (train_usr_norm.dot(test_usr_norm.transpose())).power(q)
train_usr_c = train_usr.tocsc() # Conversion for faster search

for i in tqdm(range(num_track)):
    ti = track[i]
    Ui = train_usr_c[:,ti].nonzero()[0] # Users with the song
    U_sim[:,i] = U_score_mat[Ui,:].sum(axis=0).A.ravel()
    
#%% R-precision
rprec_pl = np.zeros(num_pl_test)

for u in tqdm(range(num_pl_test)): 
    u_sim = U_sim[u,:]
    u_lim = np.argmax(test_known[u,:])+1
    u_sim[test_known[u,:u_lim]] = 0 # Zeroing out known tracks
    Uu_ord = np.argsort(u_sim)[::-1][:500]
    u_hl_lim = np.argmax(test_holdout[u,:])+1
    u_holdout = test_holdout[u,:u_hl_lim]
    rprec_pl[u] = len(np.intersect1d(Uu_ord,u_holdout))/len(u_holdout)
    
    
plt.hist(rprec_pl)
plt.show()
print("\n",'alpha = ', alph,', q = ',q,', R-precision = ',rprec_pl.mean())
stop = timeit.default_timer()

print(stop - start)

#%% Saving data
rprec_arr = np.column_stack((test_ind,rprec_pl))
df_rprec =  pd.DataFrame(rprec_arr,columns=['pl_ind','Rprec'])
df_rprec.to_csv('CF_User_rprec.csv')

rprec_mat = rprec_mat/3    
    
    
    
    
    
    
    
    

