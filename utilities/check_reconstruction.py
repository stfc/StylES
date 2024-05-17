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
import glob
import imageio

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




#------------------------------------------------------ set local parameters
TUNE_NOISE    = False
NITEZ         = 0   # number of attempts to find a closer z. When restart from a GAN field, use NITEZ=0
RELOAD_FREQ   = 100000
tollLESValues = [10.0]
maxitLES      = 1000000
USE_DIFF_LES  = False
PATH_ANIMAT   = "./results_reconstruction/plots/"

if (TESTCASE=='HIT_2D'):
    FILE_REAL_PATH = "../LES_Solvers/fields/"
    NL             = 101     # number of different latent vectors randomly selected
    t0_label       = 'step 0'
    tf_label       = 'step 10k'
    Z0_DIR_WL      = "../LES_Solvers/restart_fromGAN/"
    CHKP_DIR_WL    = "../LES_Solvers/bout_interfaces/restart_fromGAN/checkpoints_wl/"
    from HIT_2D import L
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    FILE_REAL_PATH  = "../bout_interfaces/results_DNS/fields/"
    NL              = 101     # number of different latent vectors randomly selected
    t0_label        = r'0 $\omega^{-1}_{ci}$'
    tf_label        = r'100 $\omega^{-1}_{ci}$'
    L               = 50.176
    Z0_DIR_WL      = "../bout_interfaces/restart_fromGAN/"
    CHKP_DIR_WL    = "../bout_interfaces/restart_fromGAN/checkpoints_wl/"

DELX = L/N_DNS
DELY = L/N_DNS


#------------------------------------------------------ initialization
# set folders
os.system("rm -rf results_reconstruction/plots")
os.system("rm -rf results_reconstruction/fields")
os.system("rm -rf results_reconstruction/energy")
os.system("rm -rf results_reconstruction/plots_org")
os.system("rm -rf results_reconstruction/fields_org")
os.system("rm -rf results_reconstruction/energy_org")
os.system("rm -rf results_reconstruction/logs")

os.system("mkdir -p results_reconstruction/plots")
os.system("mkdir -p results_reconstruction/fields")
os.system("mkdir -p results_reconstruction/energy")
os.system("mkdir -p results_reconstruction/plots_org/")
os.system("mkdir -p results_reconstruction/fields_org")
os.system("mkdir -p results_reconstruction/energy_org")

dir_log = './results_reconstruction/logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(SEED_RESTART)


# define optimizer for z and w search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_DNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_DNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_kDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_DNS)


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)


# create fixed synthesis model
flayer_kDNS   = layer_zlatent_kDNS()

fz_in         = tf.keras.Input(shape=([1+2*(G_LAYERS-M_LAYERS), LATENT_SIZE]),  dtype=DTYPE)
fw            = flayer_kDNS(mapping, fz_in)
fpre_w        = pre_synthesis(fw)
foutputs      = synthesis([fw, fpre_w], training=False)
fwl_synthesis = tf.keras.Model(inputs=fz_in, outputs=[foutputs, fw])


# create variable synthesis model
layer_kDNS   = layer_zlatent_kDNS()

z_in   = tf.keras.Input(shape=([1+2*(G_LAYERS-M_LAYERS), LATENT_SIZE]),  dtype=DTYPE)
img_in = []
for res in range(2,RES_LOG2-FIL+1):
    img_in.append(tf.keras.Input(shape=([NUM_CHANNELS, 2**res, 2**res]), dtype=DTYPE))
w            = layer_kDNS(mapping, z_in)
outputs      = synthesis([w, img_in], training=False)
wl_synthesis = tf.keras.Model(inputs=[z_in, img_in], outputs=[outputs, w])


# create filter model
if (GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = gaussian_filter(x_in[0,0,:,:], rs=RS2, rsca=RS)
    gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    x_in        = tf.keras.Input(shape=([1, RS+1, RS+1]), dtype=DTYPE)
    out         = gaussian_filter(x_in[0,0,:,:], rs=RS2, rsca=RS, subsection=True)
    gfilter_sub = tf.keras.Model(inputs=x_in, outputs=out)
else:
    gfilter = filters[IFIL]


# define checkpoints wl_synthesis and filter
checkpoint_wl        = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    
for variable in layer_kDNS.trainable_variables:
    ltv_DNS.append(variable)

print("\n kDNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)


# set initial scaling coefficients
UVP_max = [INIT_SCA, INIT_SCA, INIT_SCA]

time.sleep(3)




#------------------------------------------------------ define functions
def kilos(x, pos):
    'The two args are the value and tick position'
    return '%1dk' % (x*1e-3)




#------------------------------------------------------ loop over all tollerances

# preprare
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

totResREC = np.zeros((ltoll,NL), dtype=DTYPE)

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

# set z
z0i = tf.random.uniform(shape=[BATCH_SIZE,                 1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
z0p = tf.random.uniform(shape=[BATCH_SIZE, G_LAYERS-M_LAYERS, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
z0m = -z0p
z0  = tf.concat([z0i, z0p, z0m], axis=1)




#--------------------------- start main loop
itot   = 0
tstart = time.time()

for tv, tollLES in enumerate(tollLESValues):

    for k in range(NL):
        
        #--------------------------- load initial flow
        if (TESTCASE=='HIT_2D'):
            tail = str(int(k*100))
            totTime[k] = k*100
            FILE_REAL = FILE_REAL_PATH + "fields_run0_it" + tail + ".npz"

        if (TESTCASE=='HW' or TESTCASE=='mHW'):
            tail = str(int(k)).zfill(5)
            totTime[k] = k
            FILE_REAL = FILE_REAL_PATH + "fields_time" + tail + ".npz"

        if (FILE_REAL.endswith('.npz')):

            # load numpy array
            U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_REAL)
            U_DNS = np.cast[DTYPE](U_DNS)
            V_DNS = np.cast[DTYPE](V_DNS)
            P_DNS = np.cast[DTYPE](P_DNS)
        
        # find amax oridingal data
        maxU = np.max(U_DNS)
        minU = np.min(U_DNS)
        amaxU_org = max(abs(maxU), abs(minU))
        
        maxV = np.max(V_DNS)
        minV = np.min(V_DNS)
        amaxV_org = max(abs(maxV), abs(minV))

        maxP = np.max(P_DNS)
        minP = np.min(P_DNS)
        amaxP_org = max(abs(maxP), abs(minP))

        # save original DNS fields
        U_DNS_org = np.copy(U_DNS)
        V_DNS_org = np.copy(V_DNS)
        P_DNS_org = np.copy(P_DNS)

        # print plots
        if (tv==-1):
            filename = "results_reconstruction/plots_org/Plots_DNS_org_" + str(k).zfill(4) +".png"
            print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N=N_DNS, filename=filename, \
                Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)


        #--------------------------- save centerline for DNS values
        velx_DNS[tv,0,k] = U_DNS_org[N_DNS2, N_DNS2]
        vely_DNS[tv,0,k] = V_DNS_org[N_DNS2, N_DNS2]
        vort_DNS[tv,0,k] = P_DNS_org[N_DNS2, N_DNS2]

        # save centerline for initial and final values
        if ((tv==len(tollLESValues)-1) and (k==0)):
            c_velx[0,0,:] = U_DNS_org[N_DNS2, :]
            c_vely[0,0,:] = V_DNS_org[N_DNS2, :]
            c_vort[0,0,:] = P_DNS_org[N_DNS2, :]

            # find spectrum
            if (TESTCASE=='HIT_2D'):
                gradV_DNS_org = V_DNS_org
            elif (TESTCASE=='HW' or TESTCASE=='mHW'):
                gradV_DNS_org = np.sqrt(((cr(V_DNS_org, 1, 0) - cr(V_DNS_org, -1, 0))/(2.0*DELX))**2 \
                                      + ((cr(V_DNS_org, 0, 1) - cr(V_DNS_org, 0, -1))/(2.0*DELY))**2)
            spectra[0,0,:,:] = plot_spectrum_noPlots(U_DNS_org, gradV_DNS_org, L)

        if ((tv==len(tollLESValues)-1) and (k==(NL-1))):
            c_velx[0,1,:] = U_DNS_org[N_DNS2, :]
            c_vely[0,1,:] = V_DNS_org[N_DNS2, :]
            c_vort[0,1,:] = P_DNS_org[N_DNS2, :]

            # find spectrum
            if (TESTCASE=='HIT_2D'):
                gradV_DNS_org = V_DNS_org
            elif (TESTCASE=='HW' or TESTCASE=='mHW'):
                gradV_DNS_org = np.sqrt(((cr(V_DNS_org, 1, 0) - cr(V_DNS_org, -1, 0))/(2.0*DELX))**2 \
                                      + ((cr(V_DNS_org, 0, 1) - cr(V_DNS_org, 0, -1))/(2.0*DELY))**2)
            spectra[0,1,:,:] = plot_spectrum_noPlots(U_DNS_org, gradV_DNS_org, L)


        #---------------------------  preprare targets
        U_DNS = tf.convert_to_tensor(U_DNS, dtype=DTYPE)
        V_DNS = tf.convert_to_tensor(V_DNS, dtype=DTYPE)
        P_DNS = tf.convert_to_tensor(P_DNS, dtype=DTYPE)

        U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
        V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
        P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]
        
        fU_DNS = gfilter(U_DNS)
        fV_DNS = gfilter(V_DNS)
        fP_DNS = gfilter(P_DNS)
        
        # normalize filtered field
        U_min = tf.abs(tf.reduce_min(fU_DNS))
        U_max = tf.abs(tf.reduce_max(fU_DNS))
        V_min = tf.abs(tf.reduce_min(fV_DNS))
        V_max = tf.abs(tf.reduce_max(fV_DNS))
        P_min = tf.abs(tf.reduce_min(fP_DNS))
        P_max = tf.abs(tf.reduce_max(fP_DNS))

        nU_amax = tf.maximum(U_min, U_max)
        nV_amax = tf.maximum(V_min, V_max)
        nP_amax = tf.maximum(P_min, P_max)

        nfU = fU_DNS/nU_amax
        nfV = fV_DNS/nV_amax
        nfP = fP_DNS/nV_amax # we scale for same factor of phi to maintain the equation D2 phi = vort

        # set targets
        imgA   = tf.concat([ U_DNS,  V_DNS,  P_DNS], axis=1)
        fimgA  = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)
        nfimgA = tf.concat([nfU, nfV, nfP], axis=1)



        #--------------------------- load coefficients
        if (k%RELOAD_FREQ==0):

            # loading wl_synthesis checkpoint
            if managerCheckpoint_wl.latest_checkpoint:
                print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
            else:
                print("Initializing wl_synthesis from scratch.")

            filename = Z0_DIR_WL + "z0.npz"
                        
            data = np.load(filename)

            z0      = data["z0"]
            kDNS    = data["kDNS"]
            LES_in0 = data["LES_in0"]

            print("z0",      z0.shape,      np.min(z0),      np.max(z0))
            print("kDNS",    kDNS.shape,    np.min(kDNS),    np.max(kDNS))
            print("LES_in0", LES_in0.shape, np.min(LES_in0), np.max(LES_in0))

            # assign variables
            z0 = tf.convert_to_tensor(z0, dtype=DTYPE)

            for nvars in range(len(kDNS)):
                tkDNS = tf.convert_to_tensor(kDNS[nvars], dtype=DTYPE)
                layer_kDNS.trainable_variables[nvars].assign(tkDNS)

            LES_all0 = []
            for res in range(2,RES_LOG2-FIL):
                rs = 2**(RES_LOG2-FIL-res)
                LES_all0.append(tf.convert_to_tensor(LES_in0[:,:,::rs,::rs], dtype=DTYPE))

            # assign variable noise
            if (TUNE_NOISE):
                noise_DNS = data["noise_DNS"]
                print("noise_DNS", noise_DNS.shape, np.min(noise_DNS), np.max(noise_DNS))
                noise_DNS  = tf.convert_to_tensor(noise_DNS,  dtype=DTYPE)
                it=0
                for layer in synthesis.layers:
                    if "layer_noise_constants" in layer.name:
                        layer.trainable_variables[0].assign(noise_DNS[it])
                        it=it+1



        #--------------------------- find LES_all and multiplier for DNS field
        # UVP_max = [INIT_SCA, INIT_SCA, INIT_SCA]

        if (k==0):

            # find new scaling
            fnUVPo, nfUVPo, fUVP_amaxo, nUVP_amaxo  = find_scaling(imgA, gfilter_sub)
            
            UVP_max = tf.identity(nUVP_amaxo)

            print("Step " + str(k) + " new/old/diff U_amax/V_amax/P_amax values: ", UVP_max[0], amaxU_org, UVP_max[0]-amaxU_org, \
                UVP_max[1], amaxV_org, UVP_max[1]-amaxV_org, \
                UVP_max[2], amaxP_org, UVP_max[2]-amaxP_org)

        else:

            fnUVP, nfUVP, fUVP_amax, nUVP_amax  = find_scaling(imgA, gfilter_sub)

            # find scaling coefficients
            kUmax   = (fnUVPo[0]*nfUVP[0])/(fnUVP[0]*nfUVPo[0])*nUVP_amaxo[0]*fUVP_amax[0]/fUVP_amaxo[0]
            kVmax   = (fnUVPo[1]*nfUVP[1])/(fnUVP[1]*nfUVPo[1])*nUVP_amaxo[1]*fUVP_amax[1]/fUVP_amaxo[1]
            kPmax   = (fnUVPo[2]*nfUVP[2])/(fnUVP[2]*nfUVPo[2])*nUVP_amaxo[2]*fUVP_amax[2]/fUVP_amaxo[2]
            UVP_max = [kUmax, kVmax, kPmax]

            # save old values
            fnUVPo     = tf.identity(fnUVP)
            nfUVPo     = tf.identity(nfUVP)
            fUVP_amaxo = tf.identity(fUVP_amax)
            nUVP_amaxo = tf.identity(nUVP_amax)

            print("Step " + str(k) + " new/old/diff U_amax/V_amax/P_amax values: ", UVP_max[0].numpy(), amaxU_org, UVP_max[0].numpy()-amaxU_org, \
                UVP_max[1].numpy(), amaxV_org, UVP_max[1].numpy()-amaxV_org, \
                UVP_max[2].numpy(), amaxP_org, UVP_max[2].numpy()-amaxP_org)



        #--------------------------- set new LES_all
        if (USE_DIFF_LES):
            if (k==0):
                LES_ino = tf.identity(LES_in0)
                nfimgAo = tf.identity(nfimgA)
            LES_diff = nfimgA-nfimgAo
            LES_in   = LES_ino + LES_diff
            LES_all  = [LES_all0, LES_in]
            LES_ino  = tf.identity(LES_in)
            nfimgAo  = tf.identity(nfimgA)
        else:    
            if (k==0):
                LES_in = tf.identity(LES_in0)
            else:
                LES_in = tf.identity(nfimgA)
            LES_all = [LES_all0, LES_in]
            


        #--------------------------- start research on the latent space
        UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=0)

        if (k==-1):
            it = 0
            while (resREC>tollLES and it<lr_DNS_maxIt):

                lr = lr_schedule_DNS(it)
                UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, _, _ = \
                    step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, [z0, LES_all], imgA, fimgA, ltv_DNS, UVP_max, typeRes=0)

                # adjust variables
                kDNS  = layer_kDNS.trainable_variables[0]
                kDNSn = tf.clip_by_value(kDNS, 0.0, 1.0)
                if (tf.reduce_any((kDNS-kDNSn)>0)):
                    layer_kDNS.trainable_variables[0].assign(kDNSn)
                    UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)
                    resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=0)


                # print residuals and fields
                #if ((it%100==0 and it!=0) or (it%100==0 and k==0)):
                if (it%100==0):

                    # print residuals
                    tend = time.time()
                    lr = lr_schedule_DNS(it)
                    print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fil {6:3e} lr {7:3e}" \
                        .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

                    # write losses to tensorboard
                    with train_summary_writer.as_default():
                        itot = itot + it
                        tf.summary.scalar('resREC',   resREC,   step=itot)
                        tf.summary.scalar('resDNS',   resDNS,   step=itot)
                        tf.summary.scalar('resLES',   resLES,   step=itot)
                        tf.summary.scalar('loss_fil', loss_fil, step=itot)

                    if (it%1000==0):

                        # separate fields from GAN
                        U_DNS = UVP_DNS[0, 0, :, :].numpy()
                        V_DNS = UVP_DNS[0, 1, :, :].numpy()
                        P_DNS = UVP_DNS[0, 2, :, :].numpy()

                        U_LES = UVP_LES[0, 0, :, :].numpy()
                        V_LES = UVP_LES[0, 1, :, :].numpy()
                        P_LES = UVP_LES[0, 2, :, :].numpy()

                        fU_DNS = fUVP_DNS[0, 0, :, :].numpy()
                        fV_DNS = fUVP_DNS[0, 1, :, :].numpy()
                        fP_DNS = fUVP_DNS[0, 2, :, :].numpy()

                        # filename = "results_reconstruction/plots/Plots_UDNS_fromGAN.png"
                        filename = "results_reconstruction/plots/Plots_UDNS_fromGAN_" + str(it).zfill(5) + ".png"
                        print_fields_3(U_DNS, U_DNS_org, U_DNS-U_DNS_org, N=N_DNS, filename=filename, diff=True, \
                            Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

                        # filename = "results_reconstruction/plots/Plots_VDNS_fromGAN.png"
                        filename = "results_reconstruction/plots/Plots_VDNS_fromGAN_" + str(it).zfill(5) + ".png"
                        print_fields_3(V_DNS, V_DNS_org, V_DNS-V_DNS_org, N=N_DNS, filename=filename, diff=True, \
                            Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

                        # filename = "results_reconstruction/plots/Plots_PDNS_fromGAN.png"
                        filename = "results_reconstruction/plots/Plots_PDNS_fromGAN_" + str(it).zfill(5) + ".png"
                        print_fields_3(P_DNS, P_DNS_org, P_DNS-P_DNS_org, N=N_DNS, filename=filename, diff=True, \
                            Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

                        # filename = "results_reconstruction/plots/Plots_DNS_fromGAN.png"
                        filename = "results_reconstruction/plots/Plots_DNS_fromGAN_" + str(it).zfill(5) + ".png"
                        print_fields_3(U_DNS-U_DNS_org, V_DNS-V_DNS_org, P_DNS-P_DNS_org, N=N_DNS, filename=filename, diff=True, \
                            Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

                it = it+1


            # print final residuals
            tend = time.time()
            lr = lr_schedule_DNS(it)
            print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fil {6:3e}  lr {7:3e}" \
                .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

        print("Reconstruction on fields_time" + tail + ".npz: residuals {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e}" \
            .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))
            

        # accumulate resREC
        totResREC[tv,k] = resREC.numpy()


        # separate DNS fields from GAN
        U_DNS = UVP_DNS[0, 0, :, :].numpy()
        V_DNS = UVP_DNS[0, 1, :, :].numpy()
        P_DNS = UVP_DNS[0, 2, :, :].numpy()

        # save fields
        if ((k==0) or k==(NL-1)):
            filename = "results_reconstruction/fields/fields_tv" + str(tv) + "_k" + str(k) + ".npz"
            save_fields(0, U_DNS, V_DNS, P_DNS, filename)


        #--------------------------- save centerline DNS from GAN values
        velx_DNS[tv,1,k] = U_DNS[N_DNS2, N_DNS2]
        vely_DNS[tv,1,k] = V_DNS[N_DNS2, N_DNS2]
        vort_DNS[tv,1,k] = P_DNS[N_DNS2, N_DNS2]

        if ((tv==len(tollLESValues)-1) and (k==0)):
            c_velx[1,0,:] = U_DNS[N_DNS2, :]
            c_vely[1,0,:] = V_DNS[N_DNS2, :]
            c_vort[1,0,:] = P_DNS[N_DNS2, :]

            # find spectrum
            if (TESTCASE=='HIT_2D'):
                gradV_DNS = V_DNS
            elif (TESTCASE=='HW' or TESTCASE=='mHW'):
                gradV_DNS = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*DELX))**2 \
                                  + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*DELY))**2)
            spectra[1,0,:,:] = plot_spectrum_noPlots(U_DNS, gradV_DNS, L)

        if ((tv==len(tollLESValues)-1) and (k==(NL-1))):
            c_velx[1,1,:] = U_DNS[N_DNS2, :]
            c_vely[1,1,:] = V_DNS[N_DNS2, :]
            c_vort[1,1,:] = P_DNS[N_DNS2, :]

            # find spectrum
            if (TESTCASE=='HIT_2D'):
                gradV_DNS = V_DNS
            elif (TESTCASE=='HW' or TESTCASE=='mHW'):
                gradV_DNS = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*DELX))**2 \
                                  + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*DELY))**2)
            spectra[1,1,:,:] = plot_spectrum_noPlots(U_DNS, gradV_DNS, L)

        if (tv==len(tollLESValues)-1):

            # DNS diff fields
            filename = "results_reconstruction/plots/Plots_DNS_diffs_" + tail + "_" + str(tv) + ".png"
            print_fields_3(P_DNS_org, P_DNS, tf.math.abs(P_DNS_org-P_DNS), N=N_DNS, filename=filename, diff=True, \
                Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

            filename = "results_reconstruction/fields/fields_lat"  + str(k).zfill(5) + "_tv" + str(tv) + ".npz"
            save_fields(0, U_DNS, V_DNS, P_DNS, filename=filename)

            # LES diff fields
            U_LES = UVP_LES[0, 0, :, :].numpy()
            V_LES = UVP_LES[0, 1, :, :].numpy()
            P_LES = UVP_LES[0, 2, :, :].numpy()

            fU_DNS = fUVP_DNS[0, 0, :, :].numpy()
            fV_DNS = fUVP_DNS[0, 1, :, :].numpy()
            fP_DNS = fUVP_DNS[0, 2, :, :].numpy()

            filename = "results_reconstruction/plots/Plots_LES_diffs_" + tail + "_" + str(tv) + ".png"
            print_fields_3(P_LES, fP_DNS, tf.math.abs(P_LES - fP_DNS), N=N_DNS, filename=filename, diff=True, \
                Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)


    #--------------------------- plot values
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


    #--------------------------- save centerline values on file
    np.savez("results_reconstruction/uvw_vs_time.npz", totTime=totTime, \
        U_DNS=velx_DNS, V_DNS=vely_DNS, W_DNS=vort_DNS, \
        U_LES=velx_LES, V_LES=vely_LES, W_LES=vort_LES)

    ax1.legend(frameon=False, prop={'size': 14})
    ax2.legend(frameon=False, prop={'size': 14})
    ax3.legend(frameon=False, prop={'size': 14})

    ax1.set_xlabel(r'time [$\omega_i^{-1}$]', fontsize=20)
    ax2.set_xlabel(r'time [$\omega_i^{-1}$]', fontsize=20)
    ax3.set_xlabel(r'time [$\omega_i^{-1}$]', fontsize=20)
    
    if (TESTCASE=='HIT_2D'):
        ax1.set_ylabel(r'$u$',      fontsize=20)
        ax2.set_ylabel(r'$v$',      fontsize=20)
        ax3.set_ylabel(r'$\omega$', fontsize=20)
    else:
        ax1.set_ylabel(r'$n$',      fontsize=20)
        ax2.set_ylabel(r'$\phi$',   fontsize=20)
        ax3.set_ylabel(r'$\omega$', fontsize=20)

    # ax1.xaxis.set_major_formatter(formatter_kilos)
    # ax2.xaxis.set_major_formatter(formatter_kilos)
    # ax3.xaxis.set_major_formatter(formatter_kilos)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(16)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(16)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(16)
    
    fig.tight_layout(pad=5.0)
                                   
    plt.savefig("results_reconstruction/uvw_vs_time.png", bbox_inches='tight', pad_inches=0.05)
    
plt.close()




#------------------------------------------------------ plot other quantities

#--------------------------- plot residuals
for tv, tollLES in enumerate(tollLESValues):
    stollLES = r"$10^{{{0:d}}}$".format(exponent, 1)    
    plt.plot(totResREC[tv,:], label=r'$\epsilon_{REC}$=' + stollLES )
    maxResREC = max(totResREC[tv,:])
    plt.ylim(0,2*maxResREC)
    plt.savefig("results_reconstruction/plot_totResREC.png")
plt.close()


#--------------------------- plot centerline profiles
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


#--------------------------- plot energy spectra
plt.plot(spectra[0,0,0,:], spectra[0,0,1,:], color='k',                     linewidth=0.5, label=r'DNS at ' + t0_label)
plt.plot(spectra[0,1,0,:], spectra[0,1,1,:], color='r',                     linewidth=0.5, label=r'DNS at ' + tf_label)
plt.plot(spectra[1,0,0,:], spectra[1,0,1,:], color='k', linestyle='dotted', linewidth=0.5, label=r'StylES at ' + t0_label)
plt.plot(spectra[1,1,0,:], spectra[1,1,1,:], color='r', linestyle='dotted', linewidth=0.5, label=r'StylES at ' + tf_label)

plt.legend(frameon=False, prop={'size': 14})

plt.xscale("log")
plt.yscale("log")
# plt.xlim(xLogLim)
# plt.ylim(yLogLim)
plt.xlabel("k")
plt.ylabel("E")

plt.savefig("results_reconstruction/Energy_spectrum.png", pad_inches=0.5)

np.savez("results_reconstruction/uvw_vs_time.npz", spectra=spectra)


#--------------------------- make animation
anim_file = './results_reconstruction/animation.gif'
filenames = glob.glob(PATH_ANIMAT + "*.png")
filenames = sorted(filenames)

with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)


print ("All tasks successfully completed!!")    