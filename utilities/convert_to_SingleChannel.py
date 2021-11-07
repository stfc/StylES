import numpy as np
import os
import sys

from matplotlib import cm
from PIL import Image

sys.path.insert(0, '../LES_Solvers/')

from LES_functions import find_vorticity


sca  = 1
PATH = "./uvw2/"

files = os.listdir(PATH)
for file in files:
    filename = PATH + file
    filename2 = PATH + "diff_" + file
    img = Image.open(filename).convert('RGB')

    nimg = np.asarray(img)
    U = (nimg[:,:,0] - np.min(nimg[:,:,0]))/(np.max(nimg[:,:,0]) - np.min(nimg[:,:,0]))
    V = (nimg[:,:,1] - np.min(nimg[:,:,1]))/(np.max(nimg[:,:,1]) - np.min(nimg[:,:,1]))
    W = (nimg[:,:,2] - np.min(nimg[:,:,2]))/(np.max(nimg[:,:,2]) - np.min(nimg[:,:,2]))

    vor = find_vorticity(U, V)
    vor = (vor - np.min(vor))/(np.max(vor) - np.min(vor))
 
    nimg = W
    # nimg = W - vor
    # sca = (np.max(nimg)-np.min(nimg)) / max(np.max(W)-np.min(W), np.max(vor)-np.min(vor)) 
    # print( sca, np.max(W), np.min(W), np.max(vor), np.min(vor) )

    nimg = ((nimg - np.min(nimg))/(np.max(nimg) - np.min(nimg))*255*sca).astype(np.uint8)

    img = Image.fromarray(nimg, 'L')
    img.save(filename2)

    print("done for ", file)

