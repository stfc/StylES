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
import imageio
import glob

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
USE_DLATENTS = True   # "LATENTS" consider also mapping, DLATENTS only synthesis
NINTER       = 5
NLATS        = 1
PATH_ANIMAT  = "results_checkStyles/plots/"
N_DNS        = 2**RES_LOG2
N_LES        = 2**RES_LOG2-FIL
RS           = int(2**FIL)
  
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



# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)


# create filter model
x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
out     = gaussian_filter(x_in[0,0,:,:], rs=RS, rsca=RS)
gfilter = tf.keras.Model(inputs=x_in, outputs=out)


# load numpy array
U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_DNS)
U_DNS = np.cast[DTYPE](U_DNS)
V_DNS = np.cast[DTYPE](V_DNS)
P_DNS = np.cast[DTYPE](P_DNS)

# convert to tensor
U_DNS = tf.convert_to_tensor(U_DNS, dtype=DTYPE)
V_DNS = tf.convert_to_tensor(V_DNS, dtype=DTYPE)
P_DNS = tf.convert_to_tensor(P_DNS, dtype=DTYPE)

# save original DNS
U_DNS_org = tf.identity(U_DNS)
V_DNS_org = tf.identity(V_DNS)
P_DNS_org = tf.identity(P_DNS)

U_DNS_org = U_DNS_org[tf.newaxis,tf.newaxis,:,:]
V_DNS_org = V_DNS_org[tf.newaxis,tf.newaxis,:,:]
P_DNS_org = P_DNS_org[tf.newaxis,tf.newaxis,:,:]

imgA = tf.concat([U_DNS_org, V_DNS_org, P_DNS_org], axis=1)

# find filtered field
fU = gfilter(U_DNS_org)[0,0,:,:]
fV = gfilter(V_DNS_org)[0,0,:,:]
fP = gfilter(P_DNS_org)[0,0,:,:]

fimgA = tf.concat([fU[tf.newaxis,tf.newaxis,:,:], fV[tf.newaxis,tf.newaxis,:,:], fP[tf.newaxis,tf.newaxis,:,:]], axis=1)


tf.random.set_seed(0)
zlatents_1 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
zlatents_2 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)

dlatents_1 = mapping( zlatents_1, training=False)
dlatents_2 = mapping(-zlatents_1, training=False)

for nl in range(NLATS):
    
    tf.random.set_seed(nl+1)

    # average all styles
    for ninter in range(NINTER):
        w2 = ninter/(NINTER-1)
        w1 = 1.0-w2
        dlatents = w1*dlatents_1 + w2*dlatents_2
        

        # inference
        predictions = synthesis(dlatents, training=False)


        # write fields and energy spectra for each layer
        for kk in range(RES_LOG2, RES_LOG2+1):
            UVP_DNS = predictions[kk-2]
            res = 2**kk

            den_DNS_t = UVP_DNS[0, 0, :, :].numpy()
            phi_DNS_t = UVP_DNS[0, 1, :, :].numpy()
            vor_DNS_t = UVP_DNS[0, 2, :, :].numpy()
            
            filename = "results_checkStyles/plots/plots_" + str(nl).zfill(3) + "_inter_" + str(ninter).zfill(3) + "_res_" + str(res).zfill(3) + ".png"
            print_fields_3(den_DNS_t, phi_DNS_t, vor_DNS_t, N=res, filename=filename, \
                Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

            # filename = "results_checkStyles/plots/vort_" + str(nl).zfill(3) + "_inter_" + str(ninter).zfill(3) + "_res_" + str(res).zfill(3) + ".png"
            # print_fields_1(vor_DNS_t, filename)

            filename = "results_checkStyles/fields/fields_" + str(nl).zfill(3) + "_inter_" + str(ninter).zfill(3) + "_res_" + str(res).zfill(3)
            save_fields(0, den_DNS_t, phi_DNS_t, vor_DNS_t, filename=filename)

            print("Interpolation step " + str(ninter+1) + " of " + str(NINTER))

            
    # average single styles
    cont = 0
    listInterp = ["0coarse", "1medium", "2fine"]
    for glayer in listInterp:
        for ninter in range(NINTER):

            # linear
            w2 = ninter/(NINTER-1)

            # linear, but random for each point
            # w2 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)

            # # only 1 point at the time
            # wa = tf.zeros([ninter])
            # wb = tf.ones([1])
            # wc = tf.zeros([LATENT_SIZE-ninter-1])
            # w2 = tf.concat([wa, wb, wc],0)

            # find w1                        
            w1 = 1.0 - w2
            
            # for layer in synthesis.layers:
            #     if "layer_noise_constants" in layer.name:
            #         lname = layer.name
            #         ldx = int(lname.replace("layer_noise_constants",""))
            #         for variable in layer.trainable_variables:
            #             noise_DNS = layer.trainable_variables[0]*100.0
            #             layer.trainable_variables[0].assign(noise_DNS)
                            
            if (USE_DLATENTS):
                if (glayer=="0coarse"):
                    subdl1 = dlatents_1[:,0:C_LAYERS,:]
                    subdl2 = dlatents_2[:,0:C_LAYERS,:]
                    extdl1 = dlatents_1[:,C_LAYERS:G_LAYERS,:]
                    subdl  = subdl1*w1 + subdl2*w2
                    dlatents = tf.concat([subdl, extdl1], axis=1)
                elif (glayer=="1medium"):
                    subdl1 = dlatents_1[:,C_LAYERS:M_LAYERS,:]
                    subdl2 = dlatents_2[:,C_LAYERS:M_LAYERS,:]
                    extdl1 = dlatents_1[:,0:C_LAYERS,:]
                    extdl2 = dlatents_1[:,M_LAYERS:G_LAYERS,:]
                    subdl  = subdl1*w1 + subdl2*w2
                    dlatents = tf.concat([extdl1, subdl, extdl2], axis=1)
                elif (glayer=="2fine"):
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
                
            # filename = "results_checkStyles/plots/fields_" + str(nl).zfill(2) + "_" + str(glayer) + "_inter_" + str(ninter).zfill(3) + ".png"
            # print_fields_3(den_DNS_t, phi_DNS_t, vor_DNS_t, N=N_DNS, filename=filename)

            filename = "results_checkStyles/plots/vort_" + str(nl).zfill(2) + "_" + str(glayer) + "_inter_" + str(ninter).zfill(3) + ".png"
            print_fields_1(vor_DNS_t, filename, Wmin=-1.0, Wmax=1.0, legend=False)

            filename = "results_checkStyles/fields/fields_" + str(nl).zfill(2) + "_" + str(glayer) + "_inter_" + str(ninter).zfill(3)
            save_fields(0, den_DNS_t, phi_DNS_t, vor_DNS_t, filename=filename)

            print("Interpolation step " + str(ninter+1) + " of " + str(NINTER) + " on style " + str(glayer))
            cont = cont+1
            
    
    # # find new values
    # if (USE_DLATENTS):
    #     zlatent_1  = tf.identity(zlatent_2)
    #     dlatents_1 = tf.identity(dlatents_2)
    #     zlatent_2  = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
    #     dlatents_2 = mapping(zlatent_2, training=False)
    # else:
    #     latents_1  = tf.identity(latents_2)
    #     latents_2 =tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)


# #----------------------------- make animation
anim_file = './results_checkStyles/animation.gif'
filenames = glob.glob(PATH_ANIMAT + "*.png")
filenames = sorted(filenames)

with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)

print ("Job completed successfully")
