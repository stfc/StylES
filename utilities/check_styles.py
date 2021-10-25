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
UMIN = -2.0
UMAX =  2.0
VMIN = -2.0
VMAX =  2.0
PMIN = -1000.0
PMAX =  1000.0
CMIN =  0.0
CMAX =  1.0
WMIN = -200.0
WMAX =  200.0


# clean up
os.system("rm Energy_spectrum*")
os.system("rm -rf log*")
os.system("rm plots_it*")
CHECK_FILTER = False
LOAD_FIELD = False
dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(1)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  


# loading StyleGAN checkpoint and filter
checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))
mapping_ave.trainable = False
synthesis_ave.trainable = False


# create variable synthesis model
latents      = tf.keras.Input(shape=[LATENT_SIZE])
wlatents     = layer_wlatent(latents)
nlatents     = wlatents(latents)
dlatents     = mapping_ave(nlatents)
outputs      = synthesis_ave(dlatents, training=False)
wl_synthesis = tf.keras.Model(latents, outputs)


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


# define step for finding latent space
@tf.function
def find_latent_step(dlatents, U_DNS, V_DNS):
    with tf.GradientTape() as tape_DNS:
        predictions = wl_synthesis(dlatents, training=False)
        UV_DNS      = predictions[RES_LOG2-2]

        resDNS = tf.math.reduce_mean(tf.math.squared_difference(UV_DNS[0,0,:,:], U_DNS)) \
               + tf.math.reduce_mean(tf.math.squared_difference(UV_DNS[0,1,:,:], V_DNS))
        resDNS = resDNS*iOUTDIM22

        gradients_DNS  = tape_DNS.gradient(resDNS, wl_synthesis.trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, wl_synthesis.trainable_variables))

    return dlatents, predictions, resDNS


# print different fields (to check quality and find 2 different seeds)
for k in range(100):
    
    # load initial flow
    if (LOAD_FIELD):

        # load DNS and LES fields from a given field (restart.npz file)
        U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, totTime = load_fields()
        print_fields_1(U_DNS, V_DNS, 0, N, name="DNS_org", Wmin=WMIN, Wmax=WMAX)

        # start iteration search latent space
        itDNS        = 0
        resDNS       = large
        latent = tf.random.uniform([1, LATENT_SIZE])
        tU_DNS = tf.convert_to_tensor(U_DNS)
        tV_DNS = tf.convert_to_tensor(V_DNS)
        tstart = time.time()
        while (resDNS>tollDNS and itDNS<maxItDNS):
            latent, predictions, resDNS = find_latent_step(latent, tU_DNS, tV_DNS)
            UVW_DNS = predictions[RES_LOG2-2]

            # write values in tensorboard
            lr = lr_schedule(itDNS)
            with train_summary_writer.as_default():
                tf.summary.scalar("residuals", resDNS, step=itDNS)
                tf.summary.scalar("lr", lr, step=itDNS)

            # print the fields 
            if (itDNS%100 == 0):
                tend = time.time()
                print("DNS iterations:  time {0:3f}   it {1:3d}  residuals {2:3e}  lr {3:3e} ".format(tend-tstart, itDNS, resDNS.numpy(), lr))
                U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
                V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
                print_fields_1(U_DNS_t, V_DNS_t, 0, N, name="DNSfromDNS", Wmin=WMIN, Wmax=WMAX)

            itDNS = itDNS+1

    else:

        # find DNS and LES fields from random input 
        tf.random.set_seed(k)
        latent       = tf.random.uniform([1, LATENT_SIZE])
        predictions  = wl_synthesis(latent, training=False)


    # write fields and energy spectra
    for kk in range(2, RES_LOG2-1):
        UVW_DNS = predictions[kk]
        res = 2**(kk+2)
        U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVW_DNS[0, 1, :, :].numpy()

        filename = "styles_" + str(k) + "_res_" + str(res)
        print_fields(U_DNS_t, V_DNS_t, U_DNS_t, U_DNS_t, 0, res, name=filename, \
            Umin=UMIN, Umax=UMAX, Vmin=VMIN, Vmax=VMAX, Pmin=PMIN, Pmax=PMAX, Wmin=WMIN, Wmax=WMAX)
        plot_spectrum(U_DNS_t, V_DNS_t, L, k, res, OUTPUT_DIM, name=filename)
    print ("done seed " + str(k))
exit()

# find first wlatent space
tf.random.set_seed(14)
input_random0 = tf.random.uniform([1, LATENT_SIZE], dtype=DTYPE)
wlatents0     = mapping_ave(input_random0, training=False)


# find second wlatent space
tf.random.set_seed(9)
input_random1 = tf.random.uniform([1, LATENT_SIZE], dtype=DTYPE)
wlatents1     = mapping_ave(input_random1, training=False)


# Change style as interpolation between the 2 wlatent space
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

        if (CHECK_FILTER):
            UVW_DNS = predictions[RES_LOG2-2]
            UVW     = filter(UVW_DNS, training=False)
            UVW_LES = predictions[RES_LOG2-3]
            U       = UVW_LES[0, 0, :, :].numpy()
            V       = UVW_LES[0, 1, :, :].numpy()
            resFil  =          tf.reduce_mean(tf.math.squared_difference(UVW[0,0,:,:], U))
            resFil  = resFil + tf.reduce_mean(tf.math.squared_difference(UVW[0,1,:,:], V))
            resFil  = resFil*4/(2*OUTPUT_DIM*OUTPUT_DIM)
            print("Differences between actual filter and trained filter {0:6.3e}".format(resFil.numpy()))
        else:
            filename = "styles_" + str(st) + "_" + str(i)
            print_fields_1(U_DNS, V_DNS, 0, N, name=filename, vmin=VMIN, vmax=VMAX)

        print("done for style " + str(st) + " i " + str(i))


# print fields from second wlantent space
predictions   = wl_synthesis(wlatents1, training=False)
UVW_DNS       = predictions[RES_LOG2-2]

U_DNS = UVW_DNS[0, 0, :, :].numpy()
V_DNS = UVW_DNS[0, 1, :, :].numpy()
P_DNS = UVW_DNS[0, 2, :, :].numpy()

print_fields_1(U_DNS, V_DNS, 0, N, name="styles_new", vmin=VMIN, vmax=VMAX)
