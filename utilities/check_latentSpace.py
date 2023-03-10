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
NL          = 3     # number of different latent vectors randomly selected
TUNE_NOISE  = True
LOAD_FIELD  = False       # load field from DNS solver (via restart.npz file)
NITEZ       = 1000
RESTART_WL  = False

if (TESTCASE=='HIT_2D'):
    FILE_REAL_PATH  = "../../../data/HIT_2D_reconstruction/fields/"
elif (TESTCASE=='HW'):
    FILE_REAL_PATH  = "../../../data/HW/HW_reconstruction/fields/"
elif (TESTCASE=='mHW'):
    FILE_REAL_PATH  = "../../../data/mHW/mHW_N512_reconstruction/fields/"

CHKP_DIR_WL = "./checkpoints_wl"
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2_FIL
N2_DNS      = int(N_DNS/2)
N2_LES      = int(N_LES/2)
tollDNS     = 1.0e-4

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
layer_LES = layer_wlatent_LES()
layer_DNS = layer_wlatent_DNS()

zlatents     = tf.keras.Input(shape=([LATENT_SIZE]), dtype=DTYPE)
wlatents     = mapping(zlatents)
wlatents_LES = layer_LES(wlatents)
wlatents_DNS = layer_DNS(wlatents_LES)
outputs      = synthesis(wlatents_DNS, training=False)
wl_synthesis = tf.keras.Model(inputs=zlatents, outputs=outputs)



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

# for variable in layer_DNS.trainable_variables:
#     ltv_DNS.append(variable)


print("\n DNS variables:")
for variable in ltv_DNS:
    print(variable.name)


time.sleep(3)





#---------------------------------------------- LES tf.functions---------------------------------
@tf.function
def step_find_latents_DNS(latents, imgA, fimgA, ltv):
    with tf.GradientTape() as tape_DNS:

        # find predictions
        predictions = wl_synthesis(latents, training=False)
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

        resREC = resDNS + resLES

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
    data = np.load("results_reconstruction/zlatents.npz")
    zlatents = data["zlatents"]
else:
    zlatents = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)


# print different fields (to check quality and find 2 different seeds)
for k in range(NL):
    
    # load initial flow
    tf.random.set_seed(k)
    if (LOAD_FIELD):

        # load initial flow
        if (TESTCASE=='HIT_2D'):
            tail = str(int(k*100+6100))
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

        
        # find a closer z latent space
        if (not RESTART_WL):
            minDiff = large
            for i in range(NITEZ):

                # find new fields
                zlatents_new = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
                wlatents  = mapping(zlatents_new, training=False)

                # filename = "z_latent.png"
                # plt.plot(z.numpy()[0,:])
                # plt.savefig(filename)
                # plt.close()

                #exit()

                predictions = synthesis(wlatents, training=False)
                UVP_DNS = predictions[RES_LOG2-2]
                fUVP_DNS = filter[FIL-1](UVP_DNS)

                # find difference with target image
                diff = tf.math.reduce_mean(tf.math.squared_difference(UVP_DNS[0,0,N2_DNS,:], imgA[0,0,N2_DNS,:]))

                # swap and plot if found a new minimum
                if (diff < minDiff):
                    minDiff = diff
                    zlatents = zlatents_new

                    plt.plot(UVP_DNS[0,0,N2_DNS,:], label=str(i) + " " + str(minDiff.numpy()))
                    plt.legend()
                    plt.savefig("results_latentSpace/findz.png")

                    filename = "results_latentSpace/findz_fields_diff.png"
                    print_fields_3_diff(P_DNS_org, UVP_DNS[0,2,:,:], P_DNS_org-UVP_DNS[0,2,:,:], N_DNS, filename, \
                    Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

                    print("Find new z at iteration " + str(i) + " with diff ", minDiff.numpy())
                else:
                    if ((i%100)==0):
                        print("Looking for closer z... iteration " + str(i) + " of " + str(NITEZ))


        # start research on the latent space
        it = 0
        resREC = large
        tstart = time.time()
        while (resREC>tollDNS and it<lr_DNS_maxIt):
            lr = lr_schedule_DNS(it)
            resREC, resLES, resDNS, UVP_DNS, loss_fil = step_find_latents_DNS(zlatents, imgA, fimgA, ltv_DNS)


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

                if (it%100==0):

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
            zlatents    = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=k)
            wlatents    = mapping(zlatents, training=False)
        predictions = synthesis(wlatents, training=False)
        UVP_DNS     = predictions[RES_LOG2-2]      




    # save checkpoint for wl_synthesis and zlatents
    managerCheckpoint_wl.save()
    np.savez("results_latentSpace/zlatents.npz", zlatents=zlatents.numpy())




    # print spectrum from filter
    predictions = wl_synthesis(zlatents, training=False)
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
