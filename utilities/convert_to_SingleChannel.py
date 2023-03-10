#----------------------------------------------------------------------------------------------
#
#    Copyright (C): 2022 UKRI-STFC (Hartree Centre)
#
#    Author: Jony Castagna, Francesca Schiavello
#
#    Licence: This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------------------------------
import numpy as np
import os
import sys

from matplotlib import cm
from PIL import Image

sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_plot import *
from LES_functions  import *
from HIT_2D import N, L


sca  = 1
PATH = "../../../data/HIT_2D/fields_5imgs/"
DEST = "./results_convertion_from_npz/"
os.system("rm -rf " + DEST)
os.system("mkdir " + DEST)
closePlot=False

files = os.listdir(PATH)
nfiles = len(files)
for i,file in enumerate(sorted(files)):
    filename = PATH + file
    filename2 = DEST + file

    if (filename.endswith('.npz')):

        # load numpy array
        nimg = np.zeros([N,N,3], dtype=DTYPE)
        U, V, P, _ = load_fields(filename)
        nimg[:,:,0] = U
        nimg[:,:,1] = V
        nimg[:,:,2] = P
        nimg = np.cast[DTYPE](nimg)

    elif (filename.endswith('.png')):

        nimg = Image.open(filename).convert('RGB')
        nimg = np.asarray(nimg)
        nimg = nimg/256.0

    U = nimg[:,:,0]
    V = nimg[:,:,1]

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

cmd = "mv Energy_spectrum.png " + DEST
os.system(cmd)