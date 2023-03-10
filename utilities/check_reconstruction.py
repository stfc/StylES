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
from math import floor, log10

os.chdir('../')
from MSG_StyleGAN_tf2 import *
from IO_functions import StyleGAN_load_fields
from functions    import gaussian_kernel
os.chdir('./utilities')

from tensorflow.keras.applications.vgg16 import VGG16



# local parameters
TUNE_NOISE    = True
NITEZ         = 0   # number of attempts to find a closer z. When restart from a GAN field, use NITEZ=0
RESTART_WL    = False
RELOAD_FREQ   = 10
CHKP_DIR_WL   = "./checkpoints_wl"
N_DNS         = 2**RES_LOG2
N_LES         = 2**(RES_LOG2-FIL)
N2_DNS        = int(N_DNS/2)
N2_LES        = int(N_LES/2)
tollLESValues = [1.0e-2, 1.0e-3, 1.0e-4]
zero_DNS      = np.zeros([N_DNS, N_DNS], dtype=DTYPE)

if (TESTCASE=='HIT_2D'):
    FILE_PATH  = "../LES_Solvers/fields/"
    NL         = 101     # number of different latent vectors randomly selected
    if (NITEZ==0):
        t0_label = 'step 0'
        tf_label = 'step 10k'
    else:
        t0_label = r'545 $\tau_E$'
        tf_label = r'1818 $\tau_E$'
elif (TESTCASE=='HW'):
    FILE_PATH  = "../../../data/HW/HW_reconstruction/fields/"
    NL         = 100     # number of different latent vectors randomly selected
    t0_label = r'200 $\omega^{-1}_{ci}$'
    tf_label = r'10000 $\omega^{-1}_{ci}$'
elif (TESTCASE=='mHW'):
    FILE_PATH  = "../../../data/mHW/mHW_N512_reconstruction/fields/"
    NL         = 100     # number of different latent vectors randomly selected
    t0_label = r'200 $\omega^{-1}_{ci}$'
    tf_label = r'300 $\omega^{-1}_{ci}$'




# clean up and prepare folders
os.system("rm -rf results_reconstruction/plots")
os.system("rm -rf results_reconstruction/fields")
os.system("rm -rf results_reconstruction/uvw")
os.system("rm -rf results_reconstruction/energy")
os.system("rm -rf results_reconstruction/plots_org")
os.system("rm -rf results_reconstruction/fields_org")
os.system("rm -rf results_reconstruction/uvw_org")
os.system("rm -rf results_reconstruction/energy_org")
os.system("rm -rf logs")

os.system("mkdir -p results_reconstruction/plots")
os.system("mkdir -p results_reconstruction/fields")
os.system("mkdir -p results_reconstruction/uvw")
os.system("mkdir -p results_reconstruction/energy")
os.system("mkdir -p results_reconstruction/plots_org/")
os.system("mkdir -p results_reconstruction/fields_org")
os.system("mkdir -p results_reconstruction/uvw_org")
os.system("mkdir -p results_reconstruction/energy_org")

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

zlatents     = tf.keras.Input(shape=([LATENT_SIZE]), dtype=DTYPE)
wlatents     = mapping(zlatents)
wlatents_LES = layer_LES(wlatents)
outputs      = synthesis(wlatents_LES, training=False)
wl_synthesis = tf.keras.Model(inputs=zlatents, outputs=outputs)



# define optimizer for LES search
if (lr_LES_POLICY=="EXPONENTIAL"):
    lr_schedule_LES  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_LES,
        decay_steps=lr_LES_STEP,
        decay_rate=lr_LES_RATE,
        staircase=lr_LES_EXP_ST)
elif (lr_LES_POLICY=="PIECEWISE"):
    lr_schedule_LES = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_LES_BOUNDS, lr_LES_VALUES)
opt_LES = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_LES)



# define checkpoints wl_synthesis and filter
checkpoint_wl = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)




# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    ltv_LES = []

for variable in layer_LES.trainable_variables:
    ltv_DNS.append(variable)
    ltv_LES.append(variable)


print("\n DNS variables:")
for variable in ltv_DNS:
    print(variable.name)

print("\n LES variables:")
for variable in ltv_LES:
    print(variable.name)

time.sleep(3)




#---------------------------------------------- functions---------------------------------
@tf.function
def step_find_latents_LES(latents, fimgA, ltv):
    with tf.GradientTape() as tape_LES:

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
        resDNS = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, fimgA))
        resLES = tf.math.reduce_mean(tf.math.squared_difference(UVP_LES, fimgA))

        resREC = resDNS + resLES

        # aply gradients
        gradients_LES = tape_LES.gradient(resREC, ltv)
        opt_LES.apply_gradients(zip(gradients_LES, ltv))

        # find filter loss
        loss_fil    = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))

    return resREC, resLES, resDNS, UVP_DNS, loss_fil




#---------------------------------------------- MAIN LOOP---------------------------------
ltoll = len(tollLESValues)
totTime = np.zeros((NL), dtype=DTYPE)

velx_DNS = np.zeros((ltoll,2,NL), dtype=DTYPE)
vely_DNS = np.zeros((ltoll,2,NL), dtype=DTYPE)
vort_DNS = np.zeros((ltoll,2,NL), dtype=DTYPE)

velx_LES = np.zeros((ltoll,2,NL), dtype=DTYPE)
vely_LES = np.zeros((ltoll,2,NL), dtype=DTYPE)
vort_LES = np.zeros((ltoll,2,NL), dtype=DTYPE)

c_velx = np.zeros((2,2,N_DNS), dtype=DTYPE)
c_vely = np.zeros((2,2,N_DNS), dtype=DTYPE)
c_vort = np.zeros((2,2,N_DNS), dtype=DTYPE)

spectra = np.zeros((2,2,2,N_DNS), dtype=DTYPE)

fig, axs = plt.subplots(1, 3, figsize=(20,10))
fig.subplots_adjust(hspace=0.25)
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
if (NITEZ==0):
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
else:
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

colors = ['k','r','b','g','y']


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
    zlatents = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)


# save checkpoint for wl_synthesis
managerCheckpoint_wl.save()


# start main loop
tstart = time.time()
for tv, tollLES in enumerate(tollLESValues):

    for k in range(NL):
        
        # load initial flow
        if (TESTCASE=='HIT_2D'):
            if (NITEZ==0):
                tail = str(int(k*100))
            else:
                tail = str(int(k*100 + 6100))
            FILENAME = FILE_PATH + "fields_run0_it" + tail + ".npz"

        if (TESTCASE=='HW'):
            tail = str(int(k+200))
            FILENAME = FILE_PATH + "fields_run11_time" + tail + ".npz"

        if (TESTCASE=='mHW'):
            tail = str(int(k+200))
            FILENAME = FILE_PATH + "fields_run1000_time" + tail + ".npz"


        # load numpy array
        U_DNS, V_DNS, P_DNS, totTime[k] = load_fields(FILENAME)
        U_DNS = np.cast[DTYPE](U_DNS)
        V_DNS = np.cast[DTYPE](V_DNS)
        P_DNS = np.cast[DTYPE](P_DNS)

        if (TESTCASE=='HIT_2D'):
            if (NITEZ==0):
                totTime[k] =  k*100  # only in case we used a restart from StyleGAN!
            P_DNS = find_vorticity(U_DNS, V_DNS)

        # normalize
        U_DNS_org = 2.0*(U_DNS - np.min(U_DNS))/(np.max(U_DNS) - np.min(U_DNS)) - 1.0
        V_DNS_org = 2.0*(V_DNS - np.min(V_DNS))/(np.max(V_DNS) - np.min(V_DNS)) - 1.0
        P_DNS_org = 2.0*(P_DNS - np.min(P_DNS))/(np.max(P_DNS) - np.min(P_DNS)) - 1.0

        # print plots
        if (tv==-1):
            filename = "results_reconstruction/plots_org/Plots_DNS_org_" + tail +".png"
            print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N_DNS, filename, \
            Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)


        # save centerline for DNS values
        velx_DNS[tv,0,k] = U_DNS_org[N2_DNS, N2_DNS]
        vely_DNS[tv,0,k] = V_DNS_org[N2_DNS, N2_DNS]
        vort_DNS[tv,0,k] = P_DNS_org[N2_DNS, N2_DNS]

        # save centerline for initial and final values
        if ((tv==len(tollLESValues)-1) and (k==0)):
            c_velx[0,0,:] = U_DNS_org[N2_DNS, :]
            c_vely[0,0,:] = V_DNS_org[N2_DNS, :]
            c_vort[0,0,:] = P_DNS_org[N2_DNS, :]

            # find spectrum
            spectra[0,0,:,:] = plot_spectrum_noPlots(U_DNS_org, V_DNS_org, L)

        if ((tv==len(tollLESValues)-1) and (k==(NL-1))):
            c_velx[0,1,:] = U_DNS_org[N2_DNS, :]
            c_vely[0,1,:] = V_DNS_org[N2_DNS, :]
            c_vort[0,1,:] = P_DNS_org[N2_DNS, :]

            # find spectrum
            spectra[0,1,:,:] = plot_spectrum_noPlots(U_DNS_org, V_DNS_org, L)

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

        if (tv==0 and k==0):
            print("LES resolution is " + str(fimgA.shape[2]) + "x" + str(fimgA.shape[3]))

        if (k%RELOAD_FREQ==0):
            checkpoint_wl.restore(managerCheckpoint_wl.latest_checkpoint)


        # find a closer z latent space
        if (k==0 and (not RESTART_WL) and NITEZ>0):

            checkpoint_wl.restore(managerCheckpoint_wl.latest_checkpoint)

            if (tv==0):
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

                        #plt.plot(UVP_DNS[0,0,N2_DNS,:], label=str(i) + " " + str(minDiff.numpy()))
                        #plt.legend()
                        #plt.savefig("results_reconstruction/findz.png")

                        filename = "results_reconstruction/findz_fields_diff.png"
                        #print_fields_3_diff(P_DNS_org, UVP_DNS[0,2,:,:], P_DNS_org-UVP_DNS[0,2,:,:], N_DNS, filename, \
                        #Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

                        print("Find new z at iteration " + str(i) + " with diff ", minDiff.numpy())
                    else:
                        if ((i%100)==0):
                            print("Looking for closer z... iteration " + str(i) + " of " + str(NITEZ))

                np.savez("results_reconstruction/zlatents.npz", zlatents=zlatents.numpy())

            else:
                    
                data = np.load("results_reconstruction/zlatents.npz")
                zlatents = data["zlatents"]


        # start research on the latent space
        it = 0
        resREC = large
        opt_LES.initial_learning_rate = lr_LES      # reload initial learning rate
        while (resREC>tollLES and it<lr_LES_maxIt):                
            lr = lr_schedule_LES(it)
            resREC, resLES, resDNS, UVP_DNS, loss_fil = step_find_latents_LES(zlatents, fimgA, ltv_LES)


            # print residuals and fields
            if ((it%100==0 and it!=0) or (it%100==0 and k==0)):

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
                    tf.summary.scalar('resREC',   resREC,   step=it)
                    tf.summary.scalar('resDNS',   resDNS,   step=it)
                    tf.summary.scalar('resLES',   resLES,   step=it)
                    tf.summary.scalar('loss_fil', loss_fil, step=it)
                    tf.summary.scalar('lr',       lr,       step=it)

                if (it%1000==-1):

                    filename = "results_reconstruction/plots/Plots_DNS_fromGAN.png"
                    # filename = "results_reconstruction/plots/Plots_DNS_fromGAN_" + str(it) + ".png"
                    print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename, \
                            Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

            it = it+1


        # print final residuals
        lr = lr_schedule_LES(it)
        tend = time.time()
        print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
            .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))


        # separate DNS fields from GAN
        U_DNS = UVP_DNS[0, 0, :, :].numpy()
        V_DNS = UVP_DNS[0, 1, :, :].numpy()
        P_DNS = UVP_DNS[0, 2, :, :].numpy()

        # save fields
        if ((k==0) or k==(NL-1)):
            filename = "results_reconstruction/fields/fields_tv" + str(tv) + "_k" + str(k) + ".npz"
            save_fields(0, U_DNS, V_DNS, P_DNS, zero_DNS, zero_DNS, zero_DNS, filename)


        # save centerline DNS from GAN values
        velx_DNS[tv,1,k] = U_DNS[N2_DNS, N2_DNS]
        vely_DNS[tv,1,k] = V_DNS[N2_DNS, N2_DNS]
        vort_DNS[tv,1,k] = P_DNS[N2_DNS, N2_DNS]

        if ((tv==len(tollLESValues)-1) and (k==0)):
            c_velx[1,0,:] = U_DNS[N2_DNS, :]
            c_vely[1,0,:] = V_DNS[N2_DNS, :]
            c_vort[1,0,:] = P_DNS[N2_DNS, :]

            # find spectrum
            spectra[1,0,:,:] = plot_spectrum_noPlots(U_DNS, V_DNS, L)

        if ((tv==len(tollLESValues)-1) and (k==(NL-1))):
            c_velx[1,1,:] = U_DNS[N2_DNS, :]
            c_vely[1,1,:] = V_DNS[N2_DNS, :]
            c_vort[1,1,:] = P_DNS[N2_DNS, :]

            # find spectrum
            spectra[1,1,:,:] = plot_spectrum_noPlots(U_DNS, V_DNS, L)

        if (tv==len(tollLESValues)-1):

            # print fields
            # filename = "results_reconstruction/plots/Plots_DNS_fromGAN_" + tail + "_" + str(tv) + ".png"
            # print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename, \
            # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

            # filename = "results_reconstruction/plots/Plots_DNS_diffs_" + tail + "_" + str(tv) + ".png"
            # print_fields_3_diff(P_DNS_org, P_DNS, tf.math.abs(P_DNS_org-P_DNS), N_DNS, filename, \
            # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=0.0, Pmax=1.0)

            # separate DNS fields from GAN
            predictions = wl_synthesis(zlatents, training=False)
            UVP_LES = predictions[RES_LOG2-FIL-2]

            U_LES = UVP_LES[0, 0, :, :].numpy()
            V_LES = UVP_LES[0, 1, :, :].numpy()
            P_LES = UVP_LES[0, 2, :, :].numpy()

            filename = "results_reconstruction/plots/Plots_LES_diffs_" + tail + "_" + str(tv) + ".png"
            print_fields_2(P_LES, P_DNS, N_DNS, filename, Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0)

            filename = "results_reconstruction/plots/Plots_DNS_diffs_" + tail + "_" + str(tv) + ".png"
            print_fields_4_diff(P_DNS_org, P_LES, P_DNS, tf.math.abs(P_DNS_org-P_DNS), N_DNS, filename, \
            Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=0.0, Pmax=1.0)


    # plot values
    if (tv==0):
        lineColor = colors[tv]
        stollLES = "{:.1e}".format(tollLESValues[tv])

        if (TESTCASE=='HIT_2D'):
            ax1.plot(totTime[:], velx_DNS[tv,0,:], color=lineColor, label=r'DNS')
            ax2.plot(totTime[:], vely_DNS[tv,0,:], color=lineColor, label=r'DNS')
            ax3.plot(totTime[:], vort_DNS[tv,0,:], color=lineColor, label=r'DNS')
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            ax1.plot(totTime[:], velx_DNS[tv,0,:], color=lineColor, label=r'DNS')
            ax2.plot(totTime[:], vely_DNS[tv,0,:], color=lineColor, label=r'DNS')
            ax3.plot(totTime[:], vort_DNS[tv,0,:], color=lineColor, label=r'DNS')

    lineColor = colors[tv+1]
    exponent = int(floor(log10(abs(tollLESValues[tv]))))
    stollLES = r"$10^{{{0:d}}}$".format(exponent, 1)

    velx_norm = np.sum(np.sqrt((velx_DNS[tv,0,:] - velx_DNS[tv,1,:])**2))
    vely_norm = np.sum(np.sqrt((vely_DNS[tv,0,:] - vely_DNS[tv,1,:])**2))
    vort_norm = np.sum(np.sqrt((vort_DNS[tv,0,:] - vort_DNS[tv,1,:])**2))

    ax1.plot(totTime[:], velx_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'StylES, $\epsilon_{REC}$=' + stollLES + ',   $\| \| \cdot \| \|_2$ = {:.2e}'.format(velx_norm))
    ax2.plot(totTime[:], vely_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'StylES, $\epsilon_{REC}$=' + stollLES + ',   $\| \| \cdot \| \|_2$ = {:.2e}'.format(vely_norm))
    ax3.plot(totTime[:], vort_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'StylES, $\epsilon_{REC}$=' + stollLES + ',   $\| \| \cdot \| \|_2$ = {:.2e}'.format(vort_norm))


    # save centerline values on file
    np.savez("results_reconstruction/uvw_vs_time.npz", totTime=totTime, \
        U_DNS=velx_DNS, V_DNS=vely_DNS, W_DNS=vort_DNS, \
        U_LES=velx_LES, V_LES=vely_LES, W_LES=vort_LES)


    ax1.legend(frameon=False, prop={'size': 11})
    ax2.legend(frameon=False, prop={'size': 11})
    ax3.legend(frameon=False, prop={'size': 11})

    ax1.set_xlabel(r'steps', fontsize=22)
    ax2.set_xlabel(r'steps', fontsize=22)
    ax3.set_xlabel(r'steps', fontsize=22)
    
    if (TESTCASE=='HIT_2D'):
        ax1.set_ylabel(r'$u$',      fontsize=22)
        ax2.set_ylabel(r'$v$',      fontsize=22)
        ax3.set_ylabel(r'$\omega$', fontsize=22)
    else:
        ax1.set_ylabel(r'$n$',      fontsize=22)
        ax2.set_ylabel(r'$\phi$',   fontsize=22)
        ax3.set_ylabel(r'$\omega$', fontsize=22)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(18)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(18)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(18)
    
    fig.tight_layout(pad=5.0)
    
    plt.savefig("results_reconstruction/uvw_vs_time.png", bbox_inches='tight', pad_inches=0.05)


# save checkpoint for wl_synthesis and zlatents
managerCheckpoint_wl.save()
if (not RESTART_WL):
    zlatents=zlatents.numpy()
np.savez("results_reconstruction/zlatents.npz", zlatents=zlatents)


#------------------- plot centerline profiles
plt.close()

x = np.linspace(0, L, N_DNS)
c_fig, c_axs = plt.subplots(2, 3, figsize=(20,10))
c_fig.subplots_adjust(hspace=0.25)
cax1 = c_axs[0,0]
cax2 = c_axs[0,1]
cax3 = c_axs[0,2]
cax4 = c_axs[1,0]
cax5 = c_axs[1,1]
cax6 = c_axs[1,2]

# first step
k=0
lineColor = 'k'
cax1.plot(x, c_velx[0,0,:], color=lineColor, label=r'DNS $u$ at ' + t0_label)
cax2.plot(x, c_vely[0,0,:], color=lineColor, label=r'DNS $v$ at ' + t0_label)
cax3.plot(x, c_vort[0,0,:], color=lineColor, label=r'DNS $\omega$ ' + t0_label)

lineColor = 'r'
cax1.plot(x, c_velx[1,0,:], color=lineColor, label=r'StylES $u$ at ' + t0_label)
cax2.plot(x, c_vely[1,0,:], color=lineColor, label=r'StylES $v$ at ' + t0_label)
cax3.plot(x, c_vort[1,0,:], color=lineColor, label=r'StylES $\omega$ at ' + t0_label)

# final step
k = 1
lineColor = 'k'
cax4.plot(x, c_velx[0,1,:], color=lineColor, linestyle='dotted', label=r'DNS $u$ at ' + tf_label)
cax5.plot(x, c_vely[0,1,:], color=lineColor, linestyle='dotted', label=r'DNS $v$ at ' + tf_label)
cax6.plot(x, c_vort[0,1,:], color=lineColor, linestyle='dotted', label=r'DNS $\omega$ at ' + tf_label)

lineColor = 'r'
cax4.plot(x, c_velx[1,1,:], color=lineColor, linestyle='dotted', label=r'StylES $u$ at ' + tf_label)
cax5.plot(x, c_vely[1,1,:], color=lineColor, linestyle='dotted', label=r'StylES $v$ at ' + tf_label)
cax6.plot(x, c_vort[1,1,:], color=lineColor, linestyle='dotted', label=r'StylES $\omega$ at ' + tf_label)

cax1.set_xlabel("x", fontsize=14)
cax2.set_xlabel("x", fontsize=14)
cax3.set_xlabel("x", fontsize=14)

cax4.set_xlabel("x", fontsize=14)
cax5.set_xlabel("x", fontsize=14)
cax6.set_xlabel("x", fontsize=14)

cax1.legend()
cax2.legend()
cax3.legend()

cax4.legend()
cax5.legend()
cax6.legend()

plt.savefig("results_reconstruction/uvw_vs_x.png", bbox_inches='tight', pad_inches=0.05)
plt.close()

np.savez("results_reconstruction/uvw_vs_x.npz", totTime=totTime, U=c_velx, V=c_vely, W=c_vort)


#------------------- plot energy spectra
plt.close()

plt.plot(spectra[0,0,0,:], spectra[0,0,1,:], color='k',                     linewidth=0.5, label=r'DNS at ' + t0_label)
plt.plot(spectra[0,1,0,:], spectra[0,1,1,:], color='r',                     linewidth=0.5, label=r'DNS at ' + tf_label)
plt.plot(spectra[1,0,0,:], spectra[1,0,1,:], color='k', linestyle='dotted', linewidth=0.5, label=r'StylES at ' + t0_label)
plt.plot(spectra[1,1,0,:], spectra[1,1,1,:], color='r', linestyle='dotted', linewidth=0.5, label=r'StylES at ' + tf_label)

plt.xscale("log")
plt.yscale("log")
plt.xlim(xLogLim)
plt.ylim(yLogLim)
plt.xlabel("k")
plt.ylabel("E")
plt.legend()

plt.savefig("results_reconstruction/Energy_spectrum.png", pad_inches=0.5)

np.savez("results_reconstruction/uvw_vs_time.npz", spectra=spectra)