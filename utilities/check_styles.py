import os
import sys

sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *
from LES_functions  import *
from LES_plot       import *

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *
from PIL import Image




# clean up, declarations and initialization
NIP = 10   # number of interpolation points 
os.system("rm plots_it*")



# loading StyleGAN checkpoint and filter
checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))
mapping_ave.trainable = False
synthesis_ave.trainable = False


# create variable synthesis model
latents      = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
wlatents     = layer_wlatent(latents)
nlatents     = wlatents(latents)
outputs      = synthesis_ave(nlatents, training=False)
wl_synthesis = tf.keras.Model(latents, outputs)


# # print different fields (to check quality and find 2 different seeds)
# for k in range(100):
#     tf.random.set_seed(k)
#     input_random = tf.random.uniform([1, LATENT_SIZE], dtype=DTYPE)
#     wlatents     = mapping_ave(input_random, training=False)
#     predictions  = wl_synthesis(wlatents, training=False)
#     UVW_DNS      = predictions[RES_LOG2-2]
#     U_DNS = UVW_DNS[0, 0, :, :].numpy()
#     V_DNS = UVW_DNS[0, 1, :, :].numpy()
#     P_DNS = UVW_DNS[0, 2, :, :].numpy()
#     filename = "styles_" + str(k)
#     print_fields_1(U_DNS, V_DNS, 0, N, name=filename)
#     print ("seed " + str(k))
# exit()


# find first wlatent space
tf.random.set_seed(1)
input_random0 = tf.random.uniform([1, LATENT_SIZE], dtype=DTYPE)
wlatents0     = mapping_ave(input_random0, training=False)


# find second wlatent space
tf.random.set_seed(9)
input_random1 = tf.random.uniform([1, LATENT_SIZE], dtype=DTYPE)
wlatents1     = mapping_ave(input_random1, training=False)


# Change style as interpolation between the 2 wlatent space
cont = 0
for st in range(G_LAYERS):
    rand0 = wlatents0[:, st:st+1, :]
    rand1 = wlatents1[:, st:st+1, :]
    for i in range(NIP):
        clatents = (1.-i/float(NIP-1))*rand0 + i/float(NIP-1)*rand1 

        if (st==0):
            nwlatents = tf.concat([clatents, wlatents0[:, st+1:G_LAYERS, :]], 1)
        elif (st==G_LAYERS-1):
            nwlatents = tf.concat([wlatents0[:, 0:st, :], clatents], 1)
        else:
            nwlatents = tf.concat([wlatents0[:, 0:st, :], clatents, wlatents0[:, st+1:G_LAYERS, :]], 1)

        predictions = wl_synthesis(nwlatents, training=False)
        UVW_DNS     = predictions[RES_LOG2-2]
        U_DNS = UVW_DNS[0, 0, :, :].numpy()
        V_DNS = UVW_DNS[0, 1, :, :].numpy()
        filename = "styles_" + str(st) + "_" + str(i) + "_" + str(cont)
        print_fields_1(U_DNS, V_DNS, 0, N, name=filename)
        cont=cont+1

        print("done for style " + str(st) + " i " + str(i))


# print fields from second wlantent space
predictions   = wl_synthesis(wlatents1, training=False)
UVW_DNS       = predictions[RES_LOG2-2]

U_DNS = UVW_DNS[0, 0, :, :].numpy()
V_DNS = UVW_DNS[0, 1, :, :].numpy()
P_DNS = UVW_DNS[0, 2, :, :].numpy()

print_fields_1(U_DNS, V_DNS, 0, N, name="styles_new")
