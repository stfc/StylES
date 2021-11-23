import numpy as np
import os
import sys

from matplotlib import cm
from PIL import Image

sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from IO_functions import StyleGAN_load_fields

from LES_plot import *
from LES_functions  import *
from HIT_2D import N, L, uRef


sca  = 1
PATH = "../../../data/N256_1runs/uvw/"
DEST = "./single/"
os.system("rm -rf single")
os.system("mkdir single")
closePlot=False

files = os.listdir(PATH)
nfiles = len(files)
for i,file in enumerate(sorted(files)):
    filename = PATH + file
    filename2 = DEST + file

    if (filename.endswith('.npz')):

        # load numpy array
        nimg = np.zeros([N,N,3], dtype=DTYPE)
        img_in = StyleGAN_load_fields(filename)
        nimg[:,:,0] = img_in[-1][0,:,:]
        nimg[:,:,1] = img_in[-1][1,:,:]
        nimg[:,:,2] = img_in[-1][2,:,:]
        nimg = np.cast[DTYPE](nimg)

    elif (filename.endswith('.png')):

        nimg = Image.open(filename).convert('RGB')
        nimg = np.asarray(nimg)
        nimg = nimg/256.0

    U = nimg[:,:,0]*2*uRef - uRef
    V = nimg[:,:,1]*2*uRef - uRef

    #W = nimg[:,:,2]
    W = find_vorticity(U, V)

    print_fields(U, V, U, W, N, filename=DEST + "/plots_" + str(i) + ".png")

    if (i==nfiles-1):
        closePlot=True
    
    plot_spectrum(U, V, L, DEST + "/energy_spectrum_" + str(i) + ".txt", close=closePlot)

    # nimg = vor
    #nimg = W - vor
    # sca = (np.max(nimg)-np.min(nimg)) / max(np.max(W)-np.min(W), np.max(vor)-np.min(vor)) 
    # print( sca, np.max(W), np.min(W), np.max(vor), np.min(vor) )

    # nimg = ((nimg - np.min(nimg))/(np.max(nimg) - np.min(nimg))*255*sca).astype(np.uint8)

    # img = Image.fromarray(nimg, 'L')
    # img.save(filename2)

    print("done for ", file)

