
import sys
import json
import time
import os
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from collections import Counter
import math as m
from scipy.sparse import coo_matrix
import scipy.sparse as sp
#from sklearn.model_selection import train_test_split


def get_artist_name_binary_data(path):
    filenames = os.listdir(path)
    #binary_data = np.zeros((1000000,2262292))
    track_info = []
    track_id =[]
    track_artist_name =[]
    ## gets all the tracks from all 15k playlists as well as some meta_data
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            print(filename)
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            print(len(mpd_slice['playlists']))
            for playlist in mpd_slice['playlists']:
               #row_index = np.append(row_index,playlist['pid'])
                track_info.extend(get_tracks(playlist))
               #print(np.shape(track_info))
                #print(track_info)
               # track_id = [np.append (track_id, get_track_id(playlist))]
                track_id = [t[0] for t in track_info]
                #track_artist_name = np.append(track_artist_name,get_track_artist_name(playlist))
                track_artist_name = [t[1] for t in track_info]
                #info_data = np.append(info_data, playlist['num_followers'])

    #print (column_index)
    print("Tracks: ", np.shape(track_id))
    track_id = np.unique(track_id)  ## gets the unique tracks
    print("Unique Tracks: ", np.shape(track_id))
    #print(track_info)
    #print(track_info[0][0])
    print("Artists: ", np.shape(track_artist_name))
    unique_track_artist_name = np.unique(track_artist_name)
    print("Unique Artists: ", np.shape(np.unique(unique_track_artist_name)))
#    '''
    print ('......................................................')
    '''
    binary_data_1 =np.zeros((m.ceil(len(track_id)/2), len(unique_track_artist_name)))
    print(len(binary_data_1))
    
    print('Generating binary data')
    tracks = np.asarray([t[0] for t in track_info])
    
    for i in range(len(track_id)):
        binary_list_1 = np.zeros(len(unique_track_artist_name))
        #binary_list_2 = np.zeros(len(unique_track_artist_name))
        if i >= m.floor(len(track_id)/2):
            #print(i)
            if i == m.floor(len(track_id)/2):
                print(track_id[i])
            #print(track_id[i%m.floor(len(track_id)/2)])
            binary_list_1[np.where(unique_track_artist_name==track_info[np.where(tracks==track_id[i])[0][0]][1])[0]] = 1
            binary_data_1[i%m.floor(len(track_id)/2),:] = binary_list_1
        #print(np.shape(binary_list))
        #print(np.shape(binary_data))
    
        
    track_id_1 = track_id[:m.floor(len(track_id)/2)]
    print(len(track_id_1))
    track_id_2 = track_id[m.floor(len(track_id)/2):]
    print(len(track_id_2)) 
    '''
    return unique_track_artist_name


# get tracks in a playlist
def get_tracks(playlist):
    #print ('......................')
    tracks = []
    artist_name = []
    for i, track in enumerate(playlist['tracks']):
        tracks.append(track['track_uri'])
        artist_name.append(track['artist_name'])
    #print(type(tracks))
    #print(type(artist_name))
    data = list(zip(tracks,artist_name))
    #print(np.shape(data))
    return data

def get_track_id(playlist):
    #print ('......................')
    tracks = []
    for i, track in enumerate(playlist['tracks']):
        tracks.append(track['track_uri'])

    return tracks

def get_track_artist_name(playlist):
    #print ('......................')
    tracks = []
    for i, track in enumerate(playlist['tracks']):
        tracks.append(track['artist_name'])

    return tracks



# Read in the data
def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=int): #change delimiter according to the delimiter that you have (comma or space), change dtype to int8, change skiprows if you have a header
    def iter_func():
        with open(filename, 'r') as infile:
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    try:
                        yield dtype(float(item))
                    except ValueError:
                        print('Error', item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data



def process_playlists(path):
    row_index = []
    filenames = os.listdir(path)
    #binary_data = np.zeros((1000000,2262292))
    info_data = []
    column_index = []

    ## gets all the tracks from all 15k playlists as well as some meta_data
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            print(filename)
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            print(len(mpd_slice['playlists']))
            for playlist in mpd_slice['playlists']:
                row_index = np.append(row_index,playlist['pid'])
               # column_index = np.append (column_index, get_tracks(playlist))
                info_data = np.append(info_data, playlist['num_followers']).astype(int)


            info_data_unique = np.sort(np.unique(info_data)).astype(int)

    print(info_data_unique)
    print(info_data[0])
    binary_data =np.zeros((len(row_index), len(info_data_unique)))
    print(len(row_index))
    print(len(info_data))
    print(len(info_data_unique))
    for i in range(len(row_index)):
        binary_list = np.zeros(len(info_data_unique))
        #print(info_data[i])
        #print(np.where(info_data_unique == info_data[i])[0][0])
        #print(np.where(info_data_unique == info_data[i])[0][0])
        binary_list[np.where(info_data_unique == info_data[i])[0][0]]=1
        #print(np.sum(binary_list))
        binary_data[i,:] = binary_list

    print(np.sum(binary_data[0,:]))

    # the histogram of the data
    '''
    subset_info_1 = info_data[info_data <= 100]
    subset_info_2 = info_data[(info_data > 100) & (info_data < 500)]
    subset_info_3 = info_data[(info_data >= 500) & (info_data < 1000)]
    subset_info_4 = info_data[(info_data >= 1000) & (info_data < 2000)]
    subset_info_5 = info_data[(info_data >= 2000) & (info_data < 5000)]
    subset_info_6 = info_data[info_data >=5000]

    plt.figure()
    n, bins, patches = plt.hist(subset_info_1, bins=10)
    plt.xlabel('num_followers')
    plt.title('Histogram of num_followers over 15k playlists')
    plt.show()
    plt.figure()
    n, bins, patches = plt.hist(subset_info_2, bins=10)
    plt.xlabel('num_followers')
    plt.title('Histogram of num_followers over 15k playlists')
    plt.show()
    plt.figure()
    n, bins, patches = plt.hist(subset_info_3, bins=10)
    plt.xlabel('num_followers')
    plt.title('Histogram of num_followers over 15k playlists')
    plt.show()
    plt.figure()
    n, bins, patches = plt.hist(subset_info_4, bins=10)
    plt.xlabel('num_followers')
    plt.title('Histogram of num_followers over 15k playlists')
    plt.show()
    plt.figure()
    n, bins, patches = plt.hist(subset_info_5, bins=10)
    plt.xlabel('num_followers')
    plt.title('Histogram of num_followers over 15k playlists')
    plt.show()
    plt.figure()
    n, bins, patches = plt.hist(subset_info_6, bins=10)
    plt.xlabel('num_followers')
    plt.title('Histogram of num_followers over 15k playlists')
    plt.show()
    '''
    # c = Counter(info_data)
    # K=[];V=[];
    # for k,v in c.items():
    #     K=np.append(K,k)
    #     V=np.append(V,v)
    # #print(c.items())
    # plt.figure()
    # plt.plot(K,V)
    # plt.show()


                #break
    #print(np.shape(row_index))
    #print (column_index)
    #print("Tracks: ", np.shape(column_index))
    #column_index = np.unique(column_index)  ## gets the unique tracks
    #print("Unique Tracks: ", np.shape(column_index))
#    '''
    #print ('......................................................')
    '''
    binary_data =np.int8(float(np.zeros((len(row_index), len(column_index))))
    print('Generating binary data')
    count = 0
    
    ## given the unique tracks, checks the tracks in every playlist and the position corresponding to the track available in a playlist is set to 1
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            print(filename)

            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for i, playlist in enumerate(mpd_slice['playlists']):
                #print(playlist['pid'])
                binary_data[count,:]=generate_binary_data(playlist, column_index))
                print(np.sum(binary_data[count,:]))

                count=count+1

    '''
    return row_index, binary_data, info_data_unique



if __name__ == '__main__':

    path = '../data'
    '''
    row, bin_data, column = process_playlists(path)
    sparse_matrix = coo_matrix(bin_data)
    sp.save_npz("sparse_nb_followers_data", sparse_matrix)
    pd.DataFrame(column).to_csv('nb_followers_column_sparse_metadata.csv')
    '''
    column = get_artist_name_binary_data(path)
    '''
    sparse_matrix = coo_matrix(bin_data_1)
    sp.save_npz("sparese_track_artist_name_data_2", sparse_matrix)
    pd.DataFrame(row_1).to_csv('row_index_unique_track_uri_for_track_artist_metadata_2.csv')
    spare_matrix = coo_matrix(bin_data_2)
    sp.save_npz("sparese_track_artist_name_data_2", sparse_matrix)
    pd.DataFrame(row_2).to_csv('row_index_unique_track_uri_for_track_artist_metadata_2.csv')
    '''
    pd.DataFrame(column).to_csv('column_index_unique_artists.csv')  




   #print(data.columns.values)
   # pd.DataFrame(c).to_csv('column_index_unique_tracks_track_uri.csv')
    #pd.DataFrame(r).to_csv('row_index_playlist_pid.csv')   
    

   
   
'''
    print ("Saving Binary Data")
    data_csv = pd.DataFrame(data, dtype = np.int8)
    data_csv.columns = c
    data_csv.index = r
    data_csv.to_csv('binary_data_int8.csv')
    #print ("Saving meta data")
    #inf = pd.DataFrame(i)
    #inf.index = r
    #inf.to_csv('meta_data_playlist.csv')
    #print(data)
    #data = pd.read_csv('binary_data.csv', header=0, index_col=0, nrows = 218129, dtype = {str(range(15000)): np.int32})
    train = data_csv.sample(n=10000, random_state=99)
    test = data_csv.loc[~data_csv.index.isin(train.index), :]
    print("saving training")

    train.to_csv('train.csv')
    print("saving testing")

    test.to_csv('test.csv')

    arr = iter_loadtxt('binary_data_int8.csv',skiprows=1, dtype = np.int8) #change new_fp to the name of the file
    print(arr.shape)
'''


