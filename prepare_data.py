
import sys
import json
import time
import os
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
#from sklearn.model_selection import train_test_split


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
                column_index = np.append (column_index, get_tracks(playlist))
                info = parse_playlist_info(playlist)
                info_data.append(info)
                #break
    print(np.shape(row_index))
    #print (column_index)
    print("Tracks: ", np.shape(column_index))
    column_index = np.unique(column_index)  ## gets the unique tracks
    print("Unique Tracks: ", np.shape(column_index))
#    '''
    print ('......................................................')

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
                binary_data[count,:]=np.int8(float(generate_binary_data(playlist, column_index)))
                print(np.sum(binary_data[count,:]))

                count=count+1

 
    return binary_data, column_index, row_index,info_data

# get tracks in a playlist
def get_tracks(playlist):
    #print ('......................')
    tracks =[]
    for i, track in enumerate(playlist['tracks']):
        tracks.append(track['track_uri'])
    return tracks

# get meta_data of playlist
def parse_playlist_info(playlist):
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(playlist['modified_at'] / 1000))
    p ={'pid': playlist['pid'], 'collaborative': playlist['collaborative'],'name': playlist['name'],'last_modified': ts,'num_edits': playlist['num_edits'], 'num_followers': playlist['num_followers'],'num_artists': playlist['num_artists'],'num_albums': playlist['num_albums'],'num_tracks': playlist['num_tracks']}
    return p


## given the unique tracks and a playlist, checks the tracks in every playlist and the position corresponding to the track available in a playlist is set to 1

def generate_binary_data(playlist,unique_tracks):
    tracks = get_tracks(playlist)
    #print(np.shape(tracks))
    binary_list = np.zeros(len(unique_tracks))
    #print(np.shape(binary_list))
    #print(len(tracks))
    #print(len(unique_tracks))
    #print(np.where((tracks==unique_tracks))[0])
    #count = 0
    for t in tracks:
    #    print(np.where(unique_tracks==t))
        #print(np.where(unique_tracks == t)[0][0])
        binary_list[np.where(unique_tracks==t)[0]] = 1
        #count=count+1
   # print(count)
    #print(np.sum(binary_list))
    return binary_list

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

if __name__ == '__main__':

    path = '../data'
    data, c, r,i = process_playlists(path)
   #print(data.columns.values)
    pd.DataFrame(c).to_csv('column_index_unique_tracks_track_uri.csv')
    pd.DataFrame(r).to_csv('row_index_playlist_pid.csv')
    
    

   
   
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


