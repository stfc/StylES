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
import os
import sys
import scipy as sc

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *

os.chdir('../')
from MSG_StyleGAN_tf2 import *
os.chdir('./utilities')



# local parameters
USE_DLATENTS   = True   # "LATENTS" consider also mapping, DLATENTS only synthesis
NINTER         = 100

# clean up and prepare folders
os.system("rm -rf results_checkStyles/plots")
os.system("rm -rf results_checkStyles/fields")
os.system("rm -rf results_checkStyles/uvw")
os.system("rm -rf results_checkStyles/energy")

os.system("rm -rf logs")

os.system("mkdir -p results_checkStyles/plots")
os.system("mkdir -p results_checkStyles/fields")
os.system("mkdir -p results_checkStyles/uvw")
os.system("mkdir -p results_checkStyles/energy")

N_DNS = 2**RES_LOG2
N_LES = 2**RES_LOG2_FIL


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)



tf.random.set_seed(3)
if (USE_DLATENTS):
    zlatent_1  = tf.random.uniform([1, LATENT_SIZE])
    zlatent_2  = tf.random.uniform([1, LATENT_SIZE])
    dlatents_1 = mapping(zlatent_1, training=False)
    dlatents_2 = mapping(zlatent_2, training=False)
else:
    latents_1 = tf.random.uniform([1, LATENT_SIZE])
    latents_2 = tf.random.uniform([1, LATENT_SIZE])


# average all styles
for ninter in range(NINTER):
    w2 = ninter/(NINTER-1)
    w1 = 1.0-w2
    if (USE_DLATENTS):
        dlatents = dlatents_1*w1 + dlatents_2*w2
    else:
        latents = latents_1*w1 + latents_2*w2
        dlatents = mapping(latents, training=False)


    # inference
    predictions = synthesis(dlatents, training=False)


    # write fields and energy spectra for each layer
    for kk in range(RES_LOG2, RES_LOG2+1):
        UVP_DNS = predictions[kk-2]
        res = 2**kk

        den_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        phi_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        vor_DNS_t = UVP_DNS[0, 2, :, :].numpy()
        
        filename = "results_checkStyles/plots/plots_lat_" + str(ninter) + "_res_" + str(res) + ".png"
        print_fields_3(den_DNS_t, phi_DNS_t, vor_DNS_t, res, filename)

        filename = "results_checkStyles/plots/vor_lat_" + str(ninter) + "_res_" + str(res) + ".png"
        print_fields_1(vor_DNS_t, filename)

        print("Interpolation step " + str(ninter+1) + " of " + str(NINTER))


# average single styles
cont = 0
listInterp = ["coarse", "medium", "fine"]
for glayer in listInterp:
    for ninter in range(NINTER):
        w2 = ninter/(NINTER-1)
        w1 = 1.0-w2
        if (USE_DLATENTS):
            if (glayer=="coarse"):
                subdl1 = dlatents_1[:,0:C_LAYERS,:]
                subdl2 = dlatents_2[:,0:C_LAYERS,:]
                extdl1 = dlatents_1[:,C_LAYERS:G_LAYERS,:]
                subdl  = subdl1*w1 + subdl2*w2
                dlatents = tf.concat([subdl, extdl1], axis=1)
            elif (glayer=="medium"):
                subdl1 = dlatents_1[:,C_LAYERS:M_LAYERS,:]
                subdl2 = dlatents_2[:,C_LAYERS:M_LAYERS,:]
                extdl1 = dlatents_1[:,0:C_LAYERS,:]
                extdl2 = dlatents_1[:,M_LAYERS:G_LAYERS,:]
                subdl  = subdl1*w1 + subdl2*w2
                dlatents = tf.concat([extdl1, subdl, extdl2], axis=1)
            elif (glayer=="fine"):
                subdl1 = dlatents_1[:,M_LAYERS:G_LAYERS,:]
                subdl2 = dlatents_2[:,M_LAYERS:G_LAYERS,:]
                extdl1 = dlatents_1[:,0:M_LAYERS,:]
                subdl  = subdl1*w1 + subdl2*w2
                dlatents = tf.concat([extdl1, subdl], axis=1)                
        else:
            print("Cannot interpolate on z styles!")
            exit()


        # inference
        predictions = synthesis(dlatents, training=False)


        # write fields and energy spectra for each layer
        UVP_DNS = predictions[RES_LOG2-2]

        den_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        phi_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        vor_DNS_t = UVP_DNS[0, 2, :, :].numpy()
            
        #filename = "results_checkStyles/plots/plots_style_" + str(glayer) + "_int_" + str(ninter) + ".png"
        #print_fields_3(den_DNS_t, phi_DNS_t, vor_DNS_t, N_DNS, filename)

        filename = "results_checkStyles/plots/vor_style_" + str(cont) + ".png"
        print_fields_1(vor_DNS_t, filename, Wmin=-1.0, Wmax=1.0)

        print("Interpolation step " + str(ninter+1) + " of " + str(NINTER) + " on style " + str(glayer))
        cont = cont+1


print ("Job completed successfully")
