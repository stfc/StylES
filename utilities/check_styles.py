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
CHECK_FILTER = False


dir_log = 'logs/'
tf.random.set_seed(1)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  
P_DNS_t = np.zeros([OUTPUT_DIM, OUTPUT_DIM])
C_DNS_t = np.zeros([OUTPUT_DIM, OUTPUT_DIM])


# loading StyleGAN checkpoint and filter
checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))
mapping_ave.trainable = False
synthesis_ave.trainable = False


# create variable synthesis model
latents      = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
wlatents     = layer_wlatent(latents)
dlatents     = wlatents(latents)
outputs      = synthesis_ave(dlatents, training=False)
wl_synthesis = tf.keras.Model(latents, outputs)


# find first wlatent space
tf.random.set_seed(1)
input_random0 = tf.random.uniform([1, LATENT_SIZE], dtype=DTYPE)
wlatents0     = mapping_ave(input_random0, training=False)


# find second wlatent space
tf.random.set_seed(2)
input_random1 = tf.random.uniform([1, LATENT_SIZE], dtype=DTYPE)
wlatents1     = mapping_ave(input_random1, training=False)


# Change style as interpolation between the 2 wlatent space
for st in range(G_LAYERS):
    rand0 = wlatents0[:, st:st+1, :]
    rand1 = wlatents1[:, st:st+1, :]
    closePlot = False
    for i in range(NIP):
        clatents = (1.-i/float(NIP-1))*rand0 + i/float(NIP-1)*rand1 

        if (st==0):
            nwlatents = tf.concat([clatents, wlatents0[:, st+1:G_LAYERS, :]], 1)
        elif (st==G_LAYERS-1):
            nwlatents = tf.concat([wlatents0[:, 0:st, :], clatents], 1)
        else:
            nwlatents = tf.concat([wlatents0[:, 0:st, :], clatents, wlatents0[:, st+1:G_LAYERS, :]], 1)

        predictions = synthesis_ave(nwlatents, training=False)
        UVW_DNS     = predictions[RES_LOG2-2]
        U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
        W_DNS_t = UVW_DNS[0, 2, :, :].numpy()

        filename = "plots/plots_sty_" + str(st) + "_lev_" + str(i) + ".png"
        print_fields(U_DNS_t, V_DNS_t, P_DNS_t, W_DNS_t, OUTPUT_DIM, filename)
        # Umin=UMIN, Umax=UMAX, Vmin=VMIN, Vmax=VMAX, Pmin=PMIN, Pmax=PMAX, Wmin=WMIN, Wmax=WMAX)

        filename = "energy/energy_spectrum_sty_" + str(st) + "_lev_" + str(i) + ".txt"
        if (i == NIP-1):
            closePlot=True
        plot_spectrum(U_DNS_t, V_DNS_t, L, filename, closePlot)

        if (CHECK_FILTER):
    
            UVW     = filter(UVW_DNS, training=False)
            UVW_LES = predictions[RES_LOG2_FIL-2]
            U       = UVW_LES[0, 0, :, :].numpy()
            V       = UVW_LES[0, 1, :, :].numpy()
            resFil  =          tf.reduce_mean(tf.math.squared_difference(UVW[0,0,:,:], U))
            resFil  = resFil + tf.reduce_mean(tf.math.squared_difference(UVW[0,1,:,:], V))
            resFil  = resFil*4/(2*OUTPUT_DIM*OUTPUT_DIM)
            print("Differences between actual filter and trained filter {0:6.3e}".format(resFil.numpy()))

        print("done for style " + str(st) + " i " + str(i))

