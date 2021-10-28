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


# local flags
CHECK      = "LATENTS"   # "LATENTS" consider also mapping, DLATENTS only synthetis
LOAD_FIELD = False       # load field from DNS solver (via restart.npz file)
NL         = 1         # number of different latent vectors randomly selected


# local parameters
UMIN = -1.0
UMAX =  1.0
VMIN = -1.0
VMAX =  1.0
PMIN = -1000.0
PMAX =  1000.0
CMIN =  0.0
CMAX =  1.0
WMIN = -500.0
WMAX =  500.0


# clean up
os.system("rm Energy_spectrum*")
os.system("rm -rf log*")
os.system("rm plots_it*")
dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(1)
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  # 2 because we sum U and V residuals  


# loading StyleGAN checkpoint and filter
checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))
mapping_ave.trainable = False
synthesis_ave.trainable = False


# create variable synthesis model
if (CHECK=="LATENTS"):
    latents      = tf.keras.Input(shape=[LATENT_SIZE])
    wlatents     = layer_wlatent(latents)
    nlatents     = wlatents(latents)
    dlatents     = mapping_ave(nlatents)
    outputs      = synthesis_ave(dlatents, training=False)
    wl_synthesis = tf.keras.Model(latents, outputs)
else:
    dlatents     = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
    wlatents     = layer_wlatent(dlatents)
    ndlatents    = wlatents(dlatents)
    outputs      = synthesis_ave(ndlatents, training=False)
    wl_synthesis = tf.keras.Model(dlatents, outputs)


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
def find_latent_step(latent, U_DNS, V_DNS):
    with tf.GradientTape() as tape_DNS:
        predictions = wl_synthesis(latent, training=False)
        UV_DNS      = predictions[RES_LOG2-2]

        resDNS = tf.math.reduce_mean(tf.math.squared_difference(UV_DNS[0,0,:,:], U_DNS)) \
               + tf.math.reduce_mean(tf.math.squared_difference(UV_DNS[0,1,:,:], V_DNS))
        resDNS = resDNS*iOUTDIM22

        gradients_DNS  = tape_DNS.gradient(resDNS, wl_synthesis.trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, wl_synthesis.trainable_variables))

    return latent, predictions, resDNS


# print different fields (to check quality and find 2 different seeds)
for k in range(NL):
    
    # load initial flow
    if (LOAD_FIELD):

        # load DNS and LES fields from a given field (restart.npz file)
        U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, totTime = load_fields()
        print_fields_1(U_DNS, V_DNS, 0, N, name="DNS_org.png", Wmin=WMIN, Wmax=WMAX)

        # start iteration search latent space
        itDNS        = 0
        resDNS       = large
        if (CHECK=="LATENTS"):
            latent = tf.random.uniform([1, LATENT_SIZE])
        else:
            latent = tf.random.uniform([1, G_LAYERS, LATENT_SIZE])
    
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
                print_fields_1(U_DNS_t, V_DNS_t, N, name="DNSfromDNS_it{0:d}".format(itDNS) + ".png", Wmin=WMIN, Wmax=WMAX)

            itDNS = itDNS+1

    else:

        # find DNS and LES fields from random input 
        tf.random.set_seed(k)
        if (CHECK=="LATENTS"):
            latents      = tf.random.uniform([1, LATENT_SIZE])
            predictions  = wl_synthesis(latents, training=False)
        else:
            dlatents     = tf.random.uniform([1, G_LAYERS, LATENT_SIZE])
            predictions  = wl_synthesis(dlatents, training=False)


    # write fields and energy spectra
    closePlot=False
    for kk in range(2, RES_LOG2-1):
        UVW_DNS = predictions[kk]
        res = 2**(kk+2)
        U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVW_DNS[0, 1, :, :].numpy()

        filename = "plots_lat_" + str(k) + "_res_" + str(res)
        print_fields_2(U_DNS_t, V_DNS_t, res, filename)

        filename = "energy_spectrum_lat_" + str(k) + "_res_" + str(res)
        if (kk== RES_LOG2-2):
            closePlot=True
        plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

    print ("done lantent " + str(k))
