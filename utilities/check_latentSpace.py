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
from statistics import mean
import sys
from zipfile import LargeZipFile
import scipy as sc
import matplotlib.pyplot as plt

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from matplotlib.ticker import FormatStrFormatter
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
NL          = 1     # number of different latent vectors randomly selected
TUNE_NOISE  = False
LOAD_FIELD  = True       # load field from DNS solver (via restart.npz file)
RESTART_WL  = True

if (TESTCASE=='HIT_2D'):
    FILE_REAL_PATH  = "../LES_Solvers/fields/"
elif (TESTCASE=='HW'):
    FILE_REAL_PATH  = "/archive/jcastagna/Fields/HW/fields_N256_reconstruction/"
elif (TESTCASE=='mHW'):
    FILE_REAL_PATH  = "/archive/jcastagna/Fields/mHW/mHW_N512_reconstruction/"

CHKP_DIR_WL = "./checkpoints_wl"
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2_FIL
N2_DNS      = int(N_DNS/2)
N2_LES      = int(N_LES/2)
tollDNS     = 1.0e-5

zero_DNS = np.zeros([N_DNS, N_DNS], dtype=DTYPE)
zero_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)


# clean up and prepare folders
os.system("rm results_latentSpace/*.png")
os.system("rm -rf results_latentSpace/plots")
os.system("rm -rf results_latentSpace/fields")
os.system("rm -rf results_latentSpace/uvw")
os.system("rm -rf results_latentSpace/energy")
if (LOAD_FIELD):
    os.system("rm -rf results_latentSpace/plots_org")
    os.system("rm -rf results_latentSpace/fields_org")
    os.system("rm -rf results_latentSpace/uvw_org")
    os.system("rm -rf results_latentSpace/energy_org")
os.system("rm -rf logs")

os.system("mkdir -p results_latentSpace/plots")
os.system("mkdir -p results_latentSpace/fields")
os.system("mkdir -p results_latentSpace/uvw")
os.system("mkdir -p results_latentSpace/energy")
os.system("mkdir -p results_latentSpace/plots_org/")
os.system("mkdir -p results_latentSpace/fields_org")
os.system("mkdir -p results_latentSpace/uvw_org")
os.system("mkdir -p results_latentSpace/energy_org")

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(SEED_RESTART)




# loading StyleGAN checkpoint
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=1)
checkpoint.restore(managerCheckpoint.latest_checkpoint)

if managerCheckpoint.latest_checkpoint:
    print("StyleGAN restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing StyleGAN from scratch.")

time.sleep(3)




# create variable synthesis model
layer_LES = layer_wlatent_mLES()

w0           = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
w1           = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
w            = layer_LES(w0, w1)
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=[w0, w1], outputs=outputs)



# define optimizer for DNS search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_DNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_DNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_DNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_DNS)



# define checkpoints wl_synthesis and filter
checkpoint_wl = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)




# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS    = []

for variable in layer_LES.trainable_variables:
    ltv_DNS.append(variable)

print("\n DNS variables:")
for variable in ltv_DNS:
    print(variable.name)


time.sleep(3)





#---------------------------------------------- LES tf.functions---------------------------------
@tf.function
def step_find_latents_DNS(w0, w1, imgA, fimgA, ltv):
    with tf.GradientTape() as tape_DNS:

        # find predictions
        predictions = wl_synthesis([w0, w1], training=False)
        UVP_DNS = predictions[RES_LOG2-2]
        UVP_LES = predictions[RES_LOG2-FIL-2]

        # normalize
        U_DNS = UVP_DNS[0, 0, :, :]
        V_DNS = UVP_DNS[0, 1, :, :]
        P_DNS = UVP_DNS[0, 2, :, :]

        U_DNS = 2.0*(U_DNS - tf.math.reduce_min(U_DNS))/(tf.math.reduce_max(U_DNS) - tf.math.reduce_min(U_DNS)) - 1.0
        V_DNS = 2.0*(V_DNS - tf.math.reduce_min(V_DNS))/(tf.math.reduce_max(V_DNS) - tf.math.reduce_min(V_DNS)) - 1.0
        P_DNS = 2.0*(P_DNS - tf.math.reduce_min(P_DNS))/(tf.math.reduce_max(P_DNS) - tf.math.reduce_min(P_DNS)) - 1.0

        # convert back to 1 tensor
        U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
        V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
        P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]

        UVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], 1)

        # filter        
        fUVP_DNS = filter[FIL-1](UVP_DNS, training=False)

        # find residuals
        resDNS = tf.math.reduce_mean(tf.math.squared_difference(UVP_DNS, imgA))
        resLES = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, fimgA)) + \
                 tf.math.reduce_mean(tf.math.squared_difference(UVP_LES, fimgA))

        resREC = resDNS + 0.0*resLES

        # aply gradients
        gradients_DNS = tape_DNS.gradient(resREC, ltv)
        opt_DNS.apply_gradients(zip(gradients_DNS, ltv))

        # find filter loss
        loss_fil    = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))

    return resREC, resLES, resDNS, UVP_DNS, loss_fil



# set z
if (RESTART_WL):
    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    data      = np.load("results_latentSpace/z0.npz")
    z0        = data["z0"]
    w1        = data["w1"]
    mLES      = data["mLES"]
    noise_DNS = data["noise_DNS"]

    # convert to TensorFlow tensors            
    z0        = tf.convert_to_tensor(z0)
    w0        = mapping(z0, training=False) 
    w1        = tf.convert_to_tensor(w1)
    mLES      = tf.convert_to_tensor(mLES)
    noise_DNS = tf.convert_to_tensor(noise_DNS)

    # assign kDNS
    layer_LES.trainable_variables[0].assign(mLES)

    # assign variable noise
    it=0
    for layer in synthesis.layers:
        if "layer_noise_constants" in layer.name:
            layer.trainable_variables[0].assign(noise_DNS[it])
            it=it+1
else:    
    z0 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
    z1 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART+1)
    w0 = mapping(z0, training=False)
    w1 = mapping(z1, training=False)

# save old w
wto = tf.identity(w0)


# print different fields (to check quality and find 2 different seeds)
for k in range(NL):
    
    # load initial flow
    tf.random.set_seed(k)
    if (LOAD_FIELD):

        # load initial flow
        if (TESTCASE=='HIT_2D'):
            tail = str(int(k*100+10000))
            FILE_REAL = FILE_REAL_PATH + "fields_run0_it" + tail + ".npz"

        if (TESTCASE=='HW'):
            tail = str(int(k+200))
            FILE_REAL = FILE_REAL_PATH + "fields_run11_time" + tail + ".npz"

        if (TESTCASE=='mHW'):
            tail = str(int(k+200))
            FILE_REAL = FILE_REAL_PATH + "fields_run1000_time" + tail + ".npz"


        # load numpy array
        U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_REAL)
        U_DNS = np.cast[DTYPE](U_DNS)
        V_DNS = np.cast[DTYPE](V_DNS)
        P_DNS = np.cast[DTYPE](P_DNS)

        if (TESTCASE=='HIT_2D'):
            P_DNS = find_vorticity(U_DNS, V_DNS)

        filename = "results_latentSpace/fields_org/fields_lat_" + str(k) + "_res_" + str(N_DNS) + ".npz"
        save_fields(0, U_DNS, V_DNS, P_DNS, zero_DNS, zero_DNS, zero_DNS, filename)


        # normalize
        U_DNS_org = 2.0*(U_DNS - np.min(U_DNS))/(np.max(U_DNS) - np.min(U_DNS)) - 1.0
        V_DNS_org = 2.0*(V_DNS - np.min(V_DNS))/(np.max(V_DNS) - np.min(V_DNS)) - 1.0
        P_DNS_org = 2.0*(P_DNS - np.min(P_DNS))/(np.max(P_DNS) - np.min(P_DNS)) - 1.0

        # print plots
        filename = "results_latentSpace/plots_org/Plots_DNS_org_" + tail +".png"
        print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N_DNS, filename, \
        Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

        # print spectrum
        filename = "results_latentSpace/energy_org/energy_spectrum_org_" + str(k) + ".txt"
        closePlot=True
        plot_spectrum(U_DNS_org, V_DNS_org, L, filename, close=closePlot)
        
        os.system("mv Energy_spectrum.png results_latentSpace/energy_org/Energy_spectrum_org.png")


        # preprare target image
        U_DNS_t = U_DNS_org[:,:]
        V_DNS_t = V_DNS_org[:,:]
        P_DNS_t = P_DNS_org[:,:]

        tU_DNS = tf.convert_to_tensor(U_DNS_t)
        tV_DNS = tf.convert_to_tensor(V_DNS_t)
        tP_DNS = tf.convert_to_tensor(P_DNS_t)

        U_DNS = tU_DNS[np.newaxis,np.newaxis,:,:]
        V_DNS = tV_DNS[np.newaxis,np.newaxis,:,:]
        P_DNS = tP_DNS[np.newaxis,np.newaxis,:,:]

        imgA  = tf.concat([U_DNS, V_DNS, P_DNS], 1)
        fimgA = filter[FIL-1](imgA)

       
        # start research on the latent space
        it = 0
        resREC = large
        tstart = time.time()
        while (resREC>tollDNS and it<lr_DNS_maxIt):

            lr = lr_schedule_DNS(it)
            resREC, resLES, resDNS, UVP_DNS, loss_fil = step_find_latents_DNS(w0, w1, imgA, fimgA, ltv_DNS)

            mLES = layer_LES.trainable_variables[0]
            if (tf.reduce_min(mLES)<0 or tf.reduce_max(mLES)>1):
                print("Find new w1...")
                wa = mLESo*w0[:,0:M_LAYERS,:] + (1.0-mLESo)*w1[:,0:M_LAYERS,:]
                wb = wa[:,M_LAYERS-1:M_LAYERS,:]
                wb = tf.tile(wb, [1,G_LAYERS-M_LAYERS,1])
                wa = wa[:,0:M_LAYERS,:]
                wt = tf.concat([wa,wb], axis=1)
                w1 = 2*wt - wto
                w0 = tf.identity(wto)
                mLESn = tf.fill((M_LAYERS, LATENT_SIZE), 0.5)
                mLESn = tf.cast(mLESn, dtype=DTYPE)
                layer_LES.trainable_variables[0].assign(mLESn)
            else:
                mLESo = tf.identity(mLES)

            # print residuals and fields
            if (it%100==0):

                # separate DNS fields from GAN
                U_DNS = UVP_DNS[0, 0, :, :].numpy()
                V_DNS = UVP_DNS[0, 1, :, :].numpy()
                P_DNS = UVP_DNS[0, 2, :, :].numpy()

                # print residuals
                tend = time.time()
                print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
                    .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

                # write losses to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('resREC',       resREC,       step=it)
                    tf.summary.scalar('loss_fil',     loss_fil,     step=it)                    
                    tf.summary.scalar('lr',           lr,           step=it)

                if (it%1000==0):

                    # filename = "z_latent.png"
                    # plt.plot(z.numpy()[0,:])
                    # plt.savefig(filename)
                    # plt.close()

                    filename = "results_latentSpace/plots/Plots_DNS_fromGAN.png"
                    # filename = "results_latentSpace/plots/Plots_DNS_fromGAN_" + str(it) + ".png"
                    print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename, \
                            Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)


            it = it+1


        # print final residuals
        lr = lr_schedule_DNS(it)
        tend = time.time()
        print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
            .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))


    else:

        # find DNS and LES fields from random input
        if (k>=0):
            z0 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
            z1 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART+1)
            w0 = mapping(z0, training=False)
            w1 = mapping(z1, training=False)
        predictions = wl_synthesis([w0, w1], training=False)
        UVP_DNS     = predictions[RES_LOG2-2]      




    # print spectrum from filter
    predictions = wl_synthesis([w0, w1], training=False)
    UVP_DNS = predictions[RES_LOG2-2]

    UVP_LES = filter[FIL-1](UVP_DNS, training=False)
    res = 2**(RES_LOG2-FIL)
    U_t = UVP_LES[0, 0, :, :].numpy()
    V_t = UVP_LES[0, 1, :, :].numpy()
    P_t = UVP_LES[0, 2, :, :].numpy()

    filename = "results_latentSpace/plots/plots_fil_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(U_t, V_t, P_t, res, filename, TESTCASE)

    filename = "results_latentSpace/fields/fields_fil_lat_" + str(k) + "_res_" + str(res) + ".npz"
    save_fields(0, U_t, V_t, P_t, zero_LES, zero_LES, zero_LES, filename)

    filename = "results_latentSpace/energy/energy_spectrum_fil_lat_" + str(k) + "_res_" + str(res) + ".txt"
    closePlot=True
    plot_spectrum(U_t, V_t, L, filename, close=closePlot)
    
    os.system("mv Energy_spectrum.png results_latentSpace/energy/Energy_spectrum_filtered.png")




    # write fields and energy spectra for each layer
    closePlot=False
    for kk in range(4, RES_LOG2+1):
        UVP_DNS = predictions[kk-2]
        res = 2**kk

        U_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        P_DNS_t = UVP_DNS[0, 2, :, :].numpy()
        
        filename = "results_latentSpace/plots/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
        print_fields_3(U_DNS_t, V_DNS_t, P_DNS_t, res, filename, TESTCASE)

        filename = "results_latentSpace/fields/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
        save_fields(0, U_DNS_t, V_DNS_t, P_DNS_t, zero_DNS, zero_DNS, zero_DNS, filename)

        filename = "results_latentSpace/energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".txt"
        if (kk==RES_LOG2):
            closePlot=True
        plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot)

        print("From GAN spectrum at resolution " + str(res))


    print_fields_1(P_DNS_t, "results_latentSpace/plots/Plots_DNS_fromGAN.png", legend=False)
    os.system("mv Energy_spectrum.png results_latentSpace/energy/Energy_spectrum_fromGAN.png")

    print ("done latent " + str(k))
