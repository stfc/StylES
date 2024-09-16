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
import matplotlib.pyplot as plt
import os
import glob
import imageio
import sys

from PIL import Image

sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_plot import *
from LES_functions  import *

N    = 512
L    = 50.176
sca  = 1
PATH = "../../../data/BOUT_runs/HW_3D/HW_N512x16x512_perX/fields_npz/"
DEST = "./results_convertion_from_npz/"
os.system("rm -rf " + DEST)
os.system("mkdir " + DEST)
closePlot=False

def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def find_vorticity_HW(V_DNS, DELX, DELY):
    # cP_DNS = (tr(V_DNS, 1, 0) - 2*V_DNS + tr(V_DNS,-1, 0))/(DELX**2) \
    #        + (tr(V_DNS, 0, 1) - 2*V_DNS + tr(V_DNS, 0,-1))/(DELY**2)
    cP_DNS = (-cr(V_DNS, 2, 0) + 16*cr(V_DNS, 1, 0) - 30*V_DNS + 16*cr(V_DNS,-1, 0) - cr(V_DNS,-2, 0))/(12*DELX**2) \
           + (-cr(V_DNS, 0, 2) + 16*cr(V_DNS, 0, 1) - 30*V_DNS + 16*cr(V_DNS, 0,-1) - cr(V_DNS, 0,-2))/(12*DELY**2)

    return cP_DNS


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

    W = nimg[:,:,2]
    # W = find_vorticity(U, V)
    #W = find_vorticity_HW(V, L/N*8, L/N*8)


    filename = DEST + "/plots_" + str(i).zfill(4) + ".png"
    print_fields_3(U, V, W, filename=filename, testcase='HW') #, \
                # Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

    if (i==nfiles-1):
        closePlot=True

    # filename = DEST + "/energy_spectrum_" + str(i) + ".png"
    # dVdx = (-cr(V, 2, 0) + 8*cr(V, 1, 0) - 8*cr(V, -1,  0) + cr(V, -2,  0))/(12.0*L/N*8)
    # dVdy = (-cr(V, 0, 2) + 8*cr(V, 0, 1) - 8*cr(V,  0, -1) + cr(V,  0, -2))/(12.0*L/N*8)
    # plot_spectrum_2d_3v(U, dVdx, dVdy, L, filename, label="DNS", close=closePlot)

    # nimg = vor
    #nimg = W - vor
    # sca = (np.max(nimg)-np.min(nimg)) / max(np.max(W)-np.min(W), np.max(vor)-np.min(vor)) 
    # print( sca, np.max(W), np.min(W), np.max(vor), np.min(vor) )

    # nimg = ((nimg - np.min(nimg))/(np.max(nimg) - np.min(nimg))*255*sca).astype(np.uint8)

    # img = Image.fromarray(nimg, 'L')
    # img.save(filename2)

    print("done for ", file)


anim_file = DEST + "/animation.gif"
filenames = glob.glob(DEST + "/*.png")
filenames = sorted(filenames)

with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)