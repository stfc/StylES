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
from matplotlib.ticker import FuncFormatter

os.chdir('../')
from MSG_StyleGAN_tf2 import *
from IO_functions import StyleGAN_load_fields
from functions    import gaussian_kernel
os.chdir('./utilities')




#------------------------------------------- set local parameters
TUNE_NOISE    = True
NITEZ         = 0   # number of attempts to find a closer z. When restart from a GAN field, use NITEZ=0
RESTART_WL    = False
RELOAD_FREQ   = 100000
CHKP_DIR_WL   = "./checkpoints_wl"
N_DNS         = 2**RES_LOG2
N_LES         = 2**(RES_LOG2-FIL)
N2_DNS        = int(N_DNS/2)
N2_LES        = int(N_LES/2)
<<<<<<< HEAD
tollLESValues = [1.0e-2, 1.0e-3, 1.0e-4]
=======
tollLESValues = [1.0e-1, 1.0e-2, 1.0e-3]
zero_DNS      = np.zeros([N_DNS, N_DNS], dtype=DTYPE)
>>>>>>> main

if (TESTCASE=='HIT_2D'):
    FILE_PATH = "../LES_Solvers/fields/"
    NL        = 101     # number of different latent vectors randomly selected
    t0_label  = 'step 0'
    tf_label  = 'step 10k'
    from HIT_2D import L, N
elif (TESTCASE=='HW'):
    FILE_PATH  = "../bout_interfaces/results_bout/fields/"
    NL       = 101     # number of different latent vectors randomly selected
    t0_label = r'200 $\omega^{-1}_{ci}$'
    tf_label = r'10000 $\omega^{-1}_{ci}$'
    L        = 50.176
elif (TESTCASE=='mHW'):
    FILE_PATH  = "../../../data/mHW/mHW_N512_reconstruction/fields/"
    NL       = 101     # number of different latent vectors randomly selected
    t0_label = r'200 $\omega^{-1}_{ci}$'
    tf_label = r'300 $\omega^{-1}_{ci}$'
    L        = 50.176

DELX = L/N_DNS
DELY = L/N_DNS


#------------------------------------------- initialization
os.system("rm -rf results_reconstruction/plots")
os.system("rm -rf results_reconstruction/fields")
os.system("rm -rf results_reconstruction/energy")
os.system("rm -rf results_reconstruction/plots_org")
os.system("rm -rf results_reconstruction/fields_org")
os.system("rm -rf results_reconstruction/energy_org")
os.system("rm -rf logs")

os.system("mkdir -p results_reconstruction/plots")
os.system("mkdir -p results_reconstruction/fields")
os.system("mkdir -p results_reconstruction/energy")
os.system("mkdir -p results_reconstruction/plots_org/")
os.system("mkdir -p results_reconstruction/fields_org")
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
layer_LES = layer_wlatent_mLES()

w0           = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
w1           = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
w            = layer_LES(w0, w1)
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=[w0, w1], outputs=outputs)


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
checkpoint_wl        = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)



# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_LES = []

for variable in layer_LES.trainable_variables:
    ltv_LES.append(variable)

print("\n LES variables:")
for variable in ltv_LES:
    print(variable.name)

time.sleep(3)




#---------------------------------------------- functions---------------------------------
def kilos(x, pos):
    'The two args are the value and tick position'
    return '%1dk' % (x*1e-3)


<<<<<<< HEAD
=======
@tf.function
def step_find_latents_LES(w0, w1, fimgA, ltv):
    with tf.GradientTape() as tape_LES:

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
>>>>>>> main

#------------------------------------------- loop over all tollerances
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
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

colors = ['k','r','b','g','y']

formatter_kilos = FuncFormatter(kilos)


# start main loop
tstart = time.time()
for tv, tollLES in enumerate(tollLESValues):

    for k in range(NL):
        
        #-------------- load initial flow
        if (TESTCASE=='HIT_2D'):
            tail = str(int(k*100))
            totTime[k] = k*100
            FILE_REAL = FILE_PATH + "fields_run0_it" + tail + ".npz"

        if (TESTCASE=='HW'):
            tail = str(int(k)).zfill(5)
            totTime[k] = k
            FILE_REAL = FILE_PATH + "fields_time" + tail + ".npz"

        if (TESTCASE=='mHW'):
            tail = str(int(k+200))
            totTime[k] = k+200
            FILE_REAL = FILE_PATH + "fields_run1000_time" + tail + ".npz"


        #-------------- load original field
        if (FILE_REAL.endswith('.npz')):

            # load numpy array
            U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_REAL)
            U_DNS = np.cast[DTYPE](U_DNS)
            V_DNS = np.cast[DTYPE](V_DNS)
            P_DNS = np.cast[DTYPE](P_DNS)

        elif (FILE_REAL.endswith('.png')):

            # load image
            orig = Image.open(FILE_REAL).convert('RGB')

            # convert to black and white, if needed
            if (NUM_CHANNELS==1):
                orig = orig.convert("L")

            # remove white spaces
            #orig = trim(orig)

            # resize images
            orig = orig.resize((OUTPUT_DIM,OUTPUT_DIM))

            # convert to numpy array
            orig = np.asarray(orig, dtype=DTYPE)
            orig = orig/255.0

            U_DNS = orig[:,:,0]
            V_DNS = orig[:,:,1]
            P_DNS = orig[:,:,2]


        # find vorticity
        if (TESTCASE=='HIT_2D'):
<<<<<<< HEAD
=======
            if (NITEZ==0):
                totTime[k] = k*100  # only in case we used a restart from StyleGAN!
>>>>>>> main
            P_DNS = find_vorticity(U_DNS, V_DNS)
            cP_DNS = find_vorticity(U_DNS, V_DNS)
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            # cP_DNS = (tr(V_DNS, 1, 0) - 2*V_DNS + tr(V_DNS, -1, 0))/(DELX**2) \
            #            + (tr(V_DNS, 0, 1) - 2*V_DNS + tr(V_DNS, 0, -1))/(DELY**2)
            cP_DNS = (-tr(V_DNS, 2, 0) + 16*tr(V_DNS, 1, 0) - 30*V_DNS + 16*tr(V_DNS,-1, 0) - tr(V_DNS,-2, 0))/(12*DELX**2) \
                       + (-tr(V_DNS, 0, 2) + 16*tr(V_DNS, 0, 1) - 30*V_DNS + 16*tr(V_DNS, 0,-1) - tr(V_DNS, 0,-2))/(12*DELY**2)

        # normalize
        U_min = np.min(U_DNS)
        U_max = np.max(U_DNS)
        V_min = np.min(V_DNS)
        V_max = np.max(V_DNS)
        P_min = np.min(P_DNS)
        P_max = np.max(P_DNS)

        UVP_minmax = np.asarray([U_min, U_max, V_min, V_max, P_min, P_max])
        UVP_minmax = tf.convert_to_tensor(UVP_minmax)   
        
        U_DNS_org = 2.0*(U_DNS - np.min(U_DNS))/(np.max(U_DNS) - np.min(U_DNS)) - 1.0
        V_DNS_org = 2.0*(V_DNS - np.min(V_DNS))/(np.max(V_DNS) - np.min(V_DNS)) - 1.0
        P_DNS_org = 2.0*(P_DNS - np.min(P_DNS))/(np.max(P_DNS) - np.min(P_DNS)) - 1.0


        # print plots
        if (tv==-1):
            filename = "results_reconstruction/plots_org/Plots_DNS_org_" + str(k).zfill(4) +".png"
            print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N_DNS, filename, \
                Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)
                # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)



        #-------------- save centerline for DNS values
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


        #-------------- preprare targets
        U_DNS = U_DNS_org[np.newaxis,np.newaxis,:,:]
        V_DNS = V_DNS_org[np.newaxis,np.newaxis,:,:]
        P_DNS = P_DNS_org[np.newaxis,np.newaxis,:,:]

        U_DNS = tf.convert_to_tensor(U_DNS)
        V_DNS = tf.convert_to_tensor(V_DNS)
        P_DNS = tf.convert_to_tensor(P_DNS)

        # concatenate
        imgA  = tf.concat([U_DNS, V_DNS, P_DNS], 1)

        # filter
        fU_DNS = filter(U_DNS)
        fV_DNS = filter(V_DNS)
        fP_DNS = filter(P_DNS)                

        fimgA  = tf.concat([fU_DNS, fV_DNS, fP_DNS], 1)

        if (tv==0 and k==0):
            print("LES resolution is " + str(fimgA.shape[2]) + "x" + str(fimgA.shape[3]))

        if (k==0):
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

        # save old w
        wto = tf.identity(w0)

        # start research on the latent space
        it = 0
        resREC = large
        opt_LES.initial_learning_rate = lr_LES      # reload initial learning rate
        while (resREC>tollLES and it<lr_LES_maxIt):                

            lr = lr_schedule_LES(it)
<<<<<<< HEAD
            resREC, resLES, resDNS, UVP_DNS, loss_fil = step_find_latents_LES(wl_synthesis, filter, opt_LES, w0, w1, fimgA, ltv_LES)
=======
            resREC, resLES, resDNS, UVP_DNS, loss_fil = step_find_latents_LES(w0, w1, fimgA, ltv_LES)
>>>>>>> main
            
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

<<<<<<< HEAD
=======
               
>>>>>>> main
           
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


        # save old w
        mLES = layer_LES.trainable_variables[0]
        wa = mLES*w0[:,0:M_LAYERS,:] + (1.0-mLES)*w1[:,0:M_LAYERS,:]
        wb = wa[:,M_LAYERS-1:M_LAYERS,:]
        wb = tf.tile(wb, [1,G_LAYERS-M_LAYERS,1])
        wa = wa[:,0:M_LAYERS,:]
        wto = tf.concat([wa,wb], axis=1)


        # separate DNS fields from GAN
        U_DNS = UVP_DNS[0, 0, :, :].numpy()
        V_DNS = UVP_DNS[0, 1, :, :].numpy()
        P_DNS = UVP_DNS[0, 2, :, :].numpy()


        # save fields
        if ((k==0) or k==(NL-1)):
            filename = "results_reconstruction/fields/fields_tv" + str(tv) + "_k" + str(k) + ".npz"
            save_fields(0, U_DNS, V_DNS, P_DNS, filename)


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
            predictions = wl_synthesis([w0, w1], training=False)
            UVP_LES = predictions[RES_LOG2-FIL-2]

            U_LES = UVP_LES[0, 0, :, :].numpy()
            V_LES = UVP_LES[0, 1, :, :].numpy()
            P_LES = UVP_LES[0, 2, :, :].numpy()

            # mvars = layer_LES.trainable_variables[0].numpy()
            # print("Final min and max k: ", np.min(mvars), np.max(mvars))

            # filename = "results_reconstruction/plots/Plots_LES_diffs_" + tail + "_" + str(tv) + ".png"
            # print_fields_2(P_LES, P_DNS, N_DNS, filename, Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0)

            filename = "results_reconstruction/plots/Plots_DNS_diffs_" + str(k).zfill(5) + "_tv" + str(tv) + ".png"
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

    ax1.plot(totTime[:], velx_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'$\epsilon_{REC}$=' + stollLES + ',  $L_2$ = {:.2f}'.format(velx_norm))
    ax2.plot(totTime[:], vely_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'$\epsilon_{REC}$=' + stollLES + ',  $L_2$ = {:.2f}'.format(vely_norm))
    ax3.plot(totTime[:], vort_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'$\epsilon_{REC}$=' + stollLES + ',  $L_2$ = {:.2f}'.format(vort_norm))

    # save centerline values on file
    np.savez("results_reconstruction/uvw_vs_time.npz", totTime=totTime, \
        U_DNS=velx_DNS, V_DNS=vely_DNS, W_DNS=vort_DNS, \
        U_LES=velx_LES, V_LES=vely_LES, W_LES=vort_LES)

    ax1.legend(frameon=False, prop={'size': 14})
    ax2.legend(frameon=False, prop={'size': 14})
    ax3.legend(frameon=False, prop={'size': 14})

    ax1.set_xlabel(r'steps', fontsize=20)
    ax2.set_xlabel(r'steps', fontsize=20)
    ax3.set_xlabel(r'steps', fontsize=20)
    
    if (TESTCASE=='HIT_2D'):
        ax1.set_ylabel(r'$u$',      fontsize=20)
        ax2.set_ylabel(r'$v$',      fontsize=20)
        ax3.set_ylabel(r'$\omega$', fontsize=20)
    else:
        ax1.set_ylabel(r'$n$',      fontsize=20)
        ax2.set_ylabel(r'$\phi$',   fontsize=20)
        ax3.set_ylabel(r'$\omega$', fontsize=20)

    ax1.xaxis.set_major_formatter(formatter_kilos)
    ax2.xaxis.set_major_formatter(formatter_kilos)
    ax3.xaxis.set_major_formatter(formatter_kilos)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(16)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(16)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(16)
    
    fig.tight_layout(pad=5.0)
                                   
    plt.savefig("results_reconstruction/uvw_vs_time.png", bbox_inches='tight', pad_inches=0.05)

plt.close()


#------------------- plot centerline profiles
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

cax1.set_xlabel("x", fontsize=20)
cax2.set_xlabel("x", fontsize=20)
cax3.set_xlabel("x", fontsize=20)

cax4.set_xlabel("x", fontsize=20)
cax5.set_xlabel("x", fontsize=20)
cax6.set_xlabel("x", fontsize=20)

for label in (cax1.get_xticklabels() + cax1.get_yticklabels()):
    label.set_fontsize(16)
for label in (cax2.get_xticklabels() + cax2.get_yticklabels()):
    label.set_fontsize(16)
for label in (cax3.get_xticklabels() + cax3.get_yticklabels()):
    label.set_fontsize(16)

for label in (cax4.get_xticklabels() + cax4.get_yticklabels()):
    label.set_fontsize(16)
for label in (cax5.get_xticklabels() + cax5.get_yticklabels()):
    label.set_fontsize(16)
for label in (cax6.get_xticklabels() + cax6.get_yticklabels()):
    label.set_fontsize(16)
        
cax1.legend(frameon=False, prop={'size': 14})
cax2.legend(frameon=False, prop={'size': 14})
cax3.legend(frameon=False, prop={'size': 14})

cax4.legend(frameon=False, prop={'size': 14})
cax5.legend(frameon=False, prop={'size': 14})
cax6.legend(frameon=False, prop={'size': 14})

plt.savefig("results_reconstruction/uvw_vs_x.png", bbox_inches='tight', pad_inches=0.05)
plt.close()

np.savez("results_reconstruction/uvw_vs_x.npz", totTime=totTime, U=c_velx, V=c_vely, W=c_vort)




#------------------- plot energy spectra
plt.plot(spectra[0,0,0,:], spectra[0,0,1,:], color='k',                     linewidth=0.5, label=r'DNS at ' + t0_label)
plt.plot(spectra[0,1,0,:], spectra[0,1,1,:], color='r',                     linewidth=0.5, label=r'DNS at ' + tf_label)
plt.plot(spectra[1,0,0,:], spectra[1,0,1,:], color='k', linestyle='dotted', linewidth=0.5, label=r'StylES at ' + t0_label)
plt.plot(spectra[1,1,0,:], spectra[1,1,1,:], color='r', linestyle='dotted', linewidth=0.5, label=r'StylES at ' + tf_label)

plt.legend(frameon=False, prop={'size': 14})

plt.xscale("log")
plt.yscale("log")
plt.xlim(xLogLim)
plt.ylim(yLogLim)
plt.xlabel("k")
plt.ylabel("E")

plt.savefig("results_reconstruction/Energy_spectrum.png", pad_inches=0.5)

np.savez("results_reconstruction/uvw_vs_time.npz", spectra=spectra)