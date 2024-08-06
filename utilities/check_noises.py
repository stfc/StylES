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

sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *
from HIT_2D import L

from MSG_StyleGAN_tf2 import *


# local parameters
NIP = 5   # number of interpolation points 
UMIN = -0.5
UMAX =  0.5
VMIN = -0.5
VMAX =  0.5
PMIN = -1.0
PMAX =  1.0
CMIN =  0.0
CMAX =  1.0
WMIN = -0.25
WMAX =  0.25


# clean up
os.system("rm -rf plots")
os.system("rm -rf uvw")
os.system("rm -rf energy")
os.system("mkdir plots")
os.system("mkdir uvw")
os.system("mkdir energy")


dir_log = 'logs/'
tf.random.set_seed(SEED)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  
P_DNS_t = np.zeros([OUTPUT_DIM, OUTPUT_DIM])
C_DNS_t = np.zeros([OUTPUT_DIM, OUTPUT_DIM])


# loading StyleGAN checkpoint and filter
checkpoint.restore(managerCheckpoint.latest_checkpoint)


# create variable synthesis model
latents      = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
wlatents     = layer_wlatent(latents)
dlatents     = wlatents(latents)
outputs      = synthesis(dlatents, training=False)
wl_synthesis  = tf.keras.Model(latents, outputs)
wl_synthesis2 = tf.keras.Model(latents, outputs)
wl_synthesis3 = tf.keras.Model(latents, outputs)

# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, '../' + CHKP_DIR_WL, max_to_keep=2)
checkpoint_wl.restore(managerCheckpoint_wl.latest_checkpoint)

checkpoint_wl2 = tf.train.Checkpoint(wl_synthesis2=wl_synthesis)
managerCheckpoint_wl2 = tf.train.CheckpointManager(checkpoint_wl2, '../' + CHKP_DIR_WL, max_to_keep=2)
checkpoint_wl2.restore(managerCheckpoint_wl.latest_checkpoint)




@tf.function
def find_step(latent, clatents):
    wl_synthesis.trainable_variables[st].assign(clatents)
    predictions = wl_synthesis3(latent, training=False)
    UVW_DNS     = predictions[RES_LOG2-2]

    return UVW_DNS



# Change style as interpolation between the 2 wlatent space
zlatent = tf.random.uniform([1, LATENT_SIZE])
dlatents  = mapping(zlatent, training=False)

for st in range(1,len(wl_synthesis.trainable_variables)):
    rand0 = wl_synthesis.trainable_variables[st]
    rand1 = wl_synthesis2.trainable_variables[st]
    closePlot = False
    for i in range(NIP):
        if (i==NIP):
            clatents = tf.convert_to_tensor(rand0)
            UVW_DNS = find_step(dlatents, clatents)
        else:
            clatents = tf.convert_to_tensor((1.-i/float(NIP-1))*rand0 + i/float(NIP-1)*rand1) 

            # if (st==0):
            #     nwlatents = tf.concat([clatents, wl_synthesis.trainable_variables[st+1:G_LAYERS]], 0)
            # elif (st==G_LAYERS-1):
            #     nwlatents = tf.concat([wl_synthesis.trainable_variables[0:st], clatents], 0)
            # else:
            #     nwlatents = tf.concat([wl_synthesis.trainable_variables[0:st], clatents, wl_synthesis.trainable_variables[st+1:G_LAYERS]], 0)

            UVW_DNS = find_step(dlatents, clatents)

            U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
            V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
            W_DNS_t = UVW_DNS[0, 2, :, :].numpy()

            filename = "plots/plots_sty_" + str(st) + "_lev_" + str(i) + ".png"
            print_fields(U_DNS_t, V_DNS_t, P_DNS_t, W_DNS_t, OUTPUT_DIM, filename)
            # Umin=UMIN, Umax=UMAX, Vmin=VMIN, Vmax=VMAX, Pmin=PMIN, Pmax=PMAX, Wmin=WMIN, Wmax=WMAX)

            filename = "energy/energy_spectrum_sty_" + str(st) + "_lev_" + str(i) + ".png"
            if (i == NIP-1):
                closePlot=True
            plot_spectrum_2d_3v(U_DNS_t, V_DNS_t, L, filename, closePlot)

        print("done for style " + str(st) + " i " + str(i))

