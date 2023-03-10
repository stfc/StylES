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
from HIT_2D import L

os.chdir('../')
from MSG_StyleGAN_tf2 import *
from IO_functions import StyleGAN_load_fields
from functions    import gaussian_kernel
os.chdir('./utilities')

from tensorflow.keras.applications.vgg16 import VGG16



# local parameters
USE_DLATENTS   = True   # "LATENTS" consider also mapping, DLATENTS only synthesis
NL             = 2        # number of different latent vectors randomly selected
LOAD_FIELD     = False       # load field from DNS solver (via restart.npz file)
FILE_REAL      = "../../../data/LES_Solver_10ksteps/fields/fields_run0_it10.npz"
WL_IRESTART    = True
NINTER         = 10

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

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(0)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  

N_DNS = 2**RES_LOG2
N_LES = 2**RES_LOG2_FIL
zero_DNS = cp.zeros([N_DNS,N_DNS], dtype=DTYPE)
SIG   = int(N_DNS/N_LES)  # Gaussian (tf and np) filter sigma
DW    = int(N_DNS/N_LES)  # downscaling factor
minMaxUVP = np.zeros((RES_LOG2-3,6), dtype="float32")
minMaxUVP[:,0] = 1.0
minMaxUVP[:,2] = 1.0
minMaxUVP[:,4] = 1.0


    
# define noise variances
inputVar1 = tf.constant(1.0, shape=[BATCH_SIZE, G_LAYERS-2], dtype=DTYPE)
inputVar2 = tf.constant(1.0, shape=[BATCH_SIZE, 2], dtype=DTYPE)
inputVariances = tf.concat([inputVar1,inputVar2],1)


# Download VGG16 model
VGG_model         = VGG16(input_shape=(OUTPUT_DIM, OUTPUT_DIM, NUM_CHANNELS), include_top=False, weights='imagenet')
VGG_features_list = [layer.output for layer in VGG_model.layers]
VGG_extractor     = tf.keras.Model(inputs=VGG_model.input, outputs=VGG_features_list)


# loading StyleGAN checkpoint and filter
checkpoint.restore(managerCheckpoint_wl.latest_checkpoint)


# create variable synthesis model
if (USE_DLATENTS):
    dlatents         = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
    tminMaxUVP       = tf.keras.Input(shape=[6], dtype="float32")
    wlatents         = layer_wlatent(dlatents)
    ndlatents        = wlatents(dlatents)
    noutputs         = synthesis([ndlatents, inputVariances], training=False)
    rescale          = layer_rescale(name="layer_rescale")
    outputs, UVP_DNS = rescale(noutputs, tminMaxUVP)
    wl_synthesis     = tf.keras.Model(inputs=[dlatents, tminMaxUVP], outputs=[outputs, UVP_DNS])
    # wl_synthesis_0   = tf.keras.Model(inputs=[dlatents, tminMaxUVP], outputs=[outputs, UVP_DNS])
    # wl_synthesis_1   = tf.keras.Model(inputs=[dlatents, tminMaxUVP], outputs=[outputs, UVP_DNS])
else:
    latents          = tf.keras.Input(shape=[LATENT_SIZE])
    tminMaxUVP       = tf.keras.Input(shape=[6], dtype="float32")
    wlatents         = layer_wlatent(latents)
    nlatents         = wlatents(latents)
    dlatents         = mapping(nlatents)
    noutputs         = synthesis([dlatents, inputVariances], training=False)
    rescale          = layer_rescale(name="layer_rescale")
    outputs, UVP_DNS = rescale(noutputs, tminMaxUVP)
    wl_synthesis     = tf.keras.Model(inputs=[latents, tminMaxUVP],  outputs=[outputs, UVP_DNS])
    # wl_synthesis_0   = tf.keras.Model(inputs=[latents, tminMaxUVP],  outputs=[outputs, UVP_DNS])
    # wl_synthesis_1   = tf.keras.Model(inputs=[dlatents, tminMaxUVP], outputs=[outputs, UVP_DNS])

# define learnin rate schedule and optimizer
if (lrDNS_POLICY=="EXPONENTIAL"):
    lr_schedule  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lrDNS,
        decay_steps=lrDNS_STEP,
        decay_rate=lrDNS_RATE,
        staircase=lrDNS_EXP_ST)
elif (lrDNS_POLICY=="PIECEWISE"):
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lrDNS_BOUNDS, lrDNS_VALUES)
opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)


# define checkpoint
wl_checkpoint_0 = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
wl_checkpoint_1 = tf.train.Checkpoint(wl_synthesis=wl_synthesis)






for k in range(NL):

    if (k==0):
        WL_CHKP_DIR    = "./wl_checkpoints_it10"
        WL_CHKP_PREFIX = os.path.join(WL_CHKP_DIR, "ckpt")
        status = wl_checkpoint_0.restore(tf.train.latest_checkpoint(WL_CHKP_DIR))
        print(status)
    else:
        WL_CHKP_DIR    = "./wl_checkpoints_it4001"
        WL_CHKP_PREFIX = os.path.join(WL_CHKP_DIR, "ckpt")
        status = wl_checkpoint_1.restore(tf.train.latest_checkpoint(WL_CHKP_DIR))
        print(status)


    # add latent space to trainable variables
    list_DNS_trainable_variables = []
    list_LES_trainable_variables = []
    # for variable in wl_synthesis.layer[wlatents.trainable_variables:
    #     list_DNS_trainable_variables.append(variable)
    #     if "LES" in variable.name:
    #         list_LES_trainable_variables.append(variable)

    if (k==0):
        weights_0 = []
        for layer in wl_synthesis.layers:
            if ("layer_wlatent" or "input_noise") in layer.name:
                weights_0.append(layer.get_weights())
    else:
        weights_1 = []
        for layer in wl_synthesis.layers:
            if ("layer_wlatent" or "input_noise") in layer.name:
                weights_1.append(layer.get_weights())

        for ninter in range(NINTER):
            w1 = ninter/(NINTER-1)
            w2 = 1.0-w1
            cont=0
            for layer in wl_synthesis.layers:
                if ("layer_wlatent" or "input_noise") in layer.name:
                    weights_DNS = w1*weights_1[cont][0] + w2*weights_0[cont][0]
                    weights_LES = w1*weights_1[cont][1] + w2*weights_0[cont][1]
                    layer.set_weights([weights_DNS, weights_LES])
                    cont =cont+1

            tf.random.set_seed(0)

            # find DNS and LES fields from random input
            tminMaxUVP = tf.convert_to_tensor(minMaxUVP[RES_LOG2-4,:][np.newaxis,:], dtype="float32")
            if (k==0):
                if (USE_DLATENTS):
                    zlatent              = tf.random.uniform([1, LATENT_SIZE])
                    dlatents             = mapping(zlatent, training=False)
                    predictions, UVP_DNS = wl_synthesis([dlatents, tminMaxUVP], training=False)
                else:
                    latents              = tf.random.uniform([1, LATENT_SIZE])
                    predictions, UVP_DNS = wl_synthesis([latents, tminMaxUVP], training=False)
            else:
                if (USE_DLATENTS):
                    zlatent              = tf.random.uniform([1, LATENT_SIZE])
                    dlatents             = mapping(zlatent, training=False)
                    predictions, UVP_DNS = wl_synthesis([dlatents, tminMaxUVP], training=False)
                else:
                    latents              = tf.random.uniform([1, LATENT_SIZE])
                    predictions, UVP_DNS = wl_synthesis([latents, tminMaxUVP], training=False)


            # write fields and energy spectra for each layer
            closePlot=False
            for kk in range(RES_LOG2, RES_LOG2+1):
                UVP_DNS = predictions[kk-2]
                res = 2**kk

                U_DNS_t = UVP_DNS[0, 0, :, :].numpy()
                V_DNS_t = UVP_DNS[0, 1, :, :].numpy()
                P_DNS_t = UVP_DNS[0, 2, :, :].numpy()
                
                U_DNS_t = two*(U_DNS_t - np.min(U_DNS_t))/(np.max(U_DNS_t) - np.min(U_DNS_t)) - one
                V_DNS_t = two*(V_DNS_t - np.min(V_DNS_t))/(np.max(V_DNS_t) - np.min(V_DNS_t)) - one
                P_DNS_t = two*(P_DNS_t - np.min(P_DNS_t))/(np.max(P_DNS_t) - np.min(P_DNS_t)) - one

                U_DNS_t = (U_DNS_t+one)*(minMaxUVP[kk-4,0]-minMaxUVP[kk-4,1])/two + minMaxUVP[kk-4,1]
                V_DNS_t = (V_DNS_t+one)*(minMaxUVP[kk-4,2]-minMaxUVP[kk-4,3])/two + minMaxUVP[kk-4,3]
                P_DNS_t = (P_DNS_t+one)*(minMaxUVP[kk-4,4]-minMaxUVP[kk-4,5])/two + minMaxUVP[kk-4,5]

                W_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)

                filename = "results_checkStyles/plots/plots_lat_" + str(ninter) + "_res_" + str(res) + ".png"
                print_fields_1(W_DNS_t, filename)

                filename = "results_checkStyles/fields/fields_lat_" + str(ninter) + "_res_" + str(res) + ".npz"
                save_fields(0, U_DNS_t, V_DNS_t, P_DNS_t, zero_DNS, zero_DNS, W_DNS_t, filename)

                filename = "results_checkStyles/energy/energy_spectrum_lat_" + str(ninter) + "_res_" + str(res) + ".txt"
                if (kk==RES_LOG2):
                    closePlot=True
                plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

                print("From GAN spectrum at resolution " + str(res))


    os.system("mv Energy_spectrum.png results_checkStyles/energy/Energy_spectrum_fromGAN.png")

    print ("done lantent " + str(k))
