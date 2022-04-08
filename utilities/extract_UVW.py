import os
import numpy as np

PATH = './fields_single/'
DEST = './fields_UVW/'

def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def find_vorticity(U, V):
    W = ((cr(V, 1, 0)-cr(V, -1, 0)) - (cr(U, 0, 1)-cr(U, 0, -1)))
    return W

def load_fields(filename):
    data = np.load(filename)
    U = data['U']
    V = data['V']
    W = find_vorticity(U,V)
    return U, V, W

def save_fields(U, V, W, filename):
    np.savez(filename, U=U, V=V, W=W)




files = os.listdir(PATH)
nfiles = len(files)
for i,file in enumerate(sorted(files)):
    filename = PATH + file
    filename2 = DEST + file

    U, V, W = load_fields(filename)
    save_fields(U,V,W,filename2)
    
