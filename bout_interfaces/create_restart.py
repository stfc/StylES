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
import glob
import imageio

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *
from MSG_StyleGAN_tf2 import *

tf.random.set_seed(seed=SEED_RESTART)


#------------------------------------------------------ parameters
TUNE        = False 
TUNE_NOISE  = False 
tollDNS     = 1e-3
RESTART_WL  = False
if (N_DNS==256):
    FILE_DNS = FILE_DNS_N256
elif (N_DNS==512):
    FILE_DNS = FILE_DNS_N512
elif (N_DNS==1024):
    FILE_DNS = FILE_DNS_N1024
    
# check that randomization is off
noise_DNS=[]
for layer in synthesis.layers:
    if "layer_noise_constants" in layer.name:
        if len(layer.trainable_variables[:]) == 0:
            print("Carefull! Noise randomization is on!! Swith it to off in ../parameters.py")
            exit()

# set folders and paths
if (TESTCASE=='HIT_2D'):
    from HIT_2D import L
    os.system("mkdir -p ../LES_Solvers/restart_fromGAN/")
    Z0_DIR_WL = "../LES_Solvers/restart_fromGAN/"
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = 50.176
    os.system("mkdir -p ../bout_interfaces/restart_fromGAN/logs/")
    os.system("rm ../bout_interfaces/restart_fromGAN/*.png")
    os.system("rm ../bout_interfaces/restart_fromGAN/*.txt")
    Z0_DIR_WL      = "../bout_interfaces/restart_fromGAN/"
    Z0_DIR_WL_LOGS = "../bout_interfaces/restart_fromGAN/logs/"

if (not RESTART_WL):
    os.system("rm -rf " + Z0_DIR_WL)
    os.system("mkdir -p " + Z0_DIR_WL)
CHKP_DIR_WL = Z0_DIR_WL + "checkpoints_wl/"



#------------------------------------------------------ define optimizer for z and w search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_DNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_DNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_kDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_DNS)

train_summary_writer = tf.summary.create_file_writer(Z0_DIR_WL_LOGS + "search/")


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)



# create variable synthesis model
lcnoise    = layer_create_noise([1, LATENT_SIZE], 0, randomize_noise=False, nc_noise=NC_NOISE_IN, name="input_layer_noise")
x_in       = tf.constant(0, shape=[1, LATENT_SIZE], dtype=DTYPE)

z_in  = tf.keras.Input(shape=([1]),  dtype=DTYPE)
z_new = lcnoise(x_in, z_in, scalingNoise=1.0)
img_in = []
for res in range(2,RES_LOG2-FIL+1):
    img_in.append(tf.keras.Input(shape=([NUM_CHANNELS, 2**res, 2**res]), dtype=DTYPE))
w            = mapping(z_new, training=False)
outputs      = synthesis([w, img_in], training=False)
wl_synthesis = tf.keras.Model(inputs=[z_in, img_in], outputs=[outputs, w])


# create filter model
if (GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = define_filter(x_in[0,0,:,:], size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian')
    gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    x_in              = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out               = define_filter(x_in[0,0,:,:], size=4*RS, rsca=1, mean=0.0, delta=RS, type='Gaussian')
    gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)
else:
    gfilter = filters[IFIL]


# define checkpoints wl_synthesis and filter
checkpoint_wl        = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    
for variable in lcnoise.trainable_variables:
    ltv_DNS.append(variable)

print("\n kDNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)


time.sleep(3)


#------------------------------------------------------ load values
# load reference DNS

# load numpy array
U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_DNS)
U_DNS = np.cast[DTYPE](U_DNS)
V_DNS = np.cast[DTYPE](V_DNS)
P_DNS = np.cast[DTYPE](P_DNS)

tU_DNS = U_DNS[np.newaxis,np.newaxis,:,:]
tV_DNS = V_DNS[np.newaxis,np.newaxis,:,:]
tP_DNS = P_DNS[np.newaxis,np.newaxis,:,:]

UVP_DNS = np.concatenate([tU_DNS, tV_DNS, tP_DNS], axis=1)
UVP_DNS, nUVP_amaxo = np_normalize_max(UVP_DNS)

if (TESTCASE=='mHW'):
    fU_DNS = sc.ndimage.gaussian_filter(U_DNS, RS, mode=['constant','wrap'])
    fV_DNS = sc.ndimage.gaussian_filter(V_DNS, RS, mode=['constant','wrap'])
    fP_DNS = sc.ndimage.gaussian_filter(P_DNS, RS, mode=['constant','wrap'])
else:
    fU_DNS = sc.ndimage.gaussian_filter(U_DNS, RS, mode='wrap')
    fV_DNS = sc.ndimage.gaussian_filter(V_DNS, RS, mode='wrap')
    fP_DNS = sc.ndimage.gaussian_filter(P_DNS, RS, mode='wrap')

fU_DNS = fU_DNS[np.newaxis,np.newaxis,:,:]
fV_DNS = fV_DNS[np.newaxis,np.newaxis,:,:]
fP_DNS = fP_DNS[np.newaxis,np.newaxis,:,:]

UVP_LES = np.concatenate([fU_DNS, fV_DNS, fP_DNS], axis=1)
UVP_LES, fUVP_amaxo = np_normalize_max(UVP_LES)

UVP_max = nUVP_amaxo + fUVP_amaxo


filename = Z0_DIR_WL + "plots_DNS_org.png"
print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE, \
            Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

closePlot=False
filename = Z0_DIR_WL + "energy_spectrum_DNS.png"
dVdx = (-cr(V_DNS, 2, 0) + 8*cr(V_DNS, 1, 0) - 8*cr(V_DNS, -1,  0) + cr(V_DNS, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(V_DNS, 0, 2) + 8*cr(V_DNS, 0, 1) - 8*cr(V_DNS,  0, -1) + cr(V_DNS,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(U_DNS, dVdx, dVdy, L, filename, label="DNS", close=False)



# filter image
rs = 2
for reslog in range(RES_LOG2, RES_LOG2-FIL-1, -1):
    res = 2**reslog
    if (reslog==RES_LOG2):
        fU_DNS = U_DNS
        fV_DNS = V_DNS
        fP_DNS = P_DNS
    else:
        if (TESTCASE=='mHW'):
            fU_DNS = sc.ndimage.gaussian_filter(fU_DNS, rs, mode=['constant','wrap'])
            fV_DNS = sc.ndimage.gaussian_filter(fV_DNS, rs, mode=['constant','wrap'])
            fP_DNS = sc.ndimage.gaussian_filter(fP_DNS, rs, mode=['constant','wrap'])
        else:
            fU_DNS = sc.ndimage.gaussian_filter(fU_DNS, rs, mode='wrap')
            fV_DNS = sc.ndimage.gaussian_filter(fV_DNS, rs, mode='wrap')
            fP_DNS = sc.ndimage.gaussian_filter(fP_DNS, rs, mode='wrap')

        fU_DNS = fU_DNS[::rs,::rs]
        fV_DNS = fV_DNS[::rs,::rs]
        fP_DNS = fP_DNS[::rs,::rs]

    # normalize the data
    minU = np.min(fU_DNS)
    maxU = np.max(fU_DNS)
    amaxU = max(abs(minU), abs(maxU))
    fU_DNS = fU_DNS / amaxU
    
    minV = np.min(fV_DNS)
    maxV = np.max(fV_DNS)
    amaxV = max(abs(minV), abs(maxV))
    fV_DNS = fV_DNS / amaxV

    minP = np.min(fP_DNS)
    maxP = np.max(fP_DNS)
    amaxP = max(abs(minP), abs(maxP))
    fP_DNS = fP_DNS / amaxP

        

# save LES_in0
fU_DNS = tf.convert_to_tensor(fU_DNS[np.newaxis,np.newaxis,:,:])
fV_DNS = tf.convert_to_tensor(fV_DNS[np.newaxis,np.newaxis,:,:])
fP_DNS = tf.convert_to_tensor(fP_DNS[np.newaxis,np.newaxis,:,:])

LES_in0 = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)


# save original DNS field
U_DNS = tf.convert_to_tensor(U_DNS[np.newaxis,np.newaxis,:,:])
V_DNS = tf.convert_to_tensor(V_DNS[np.newaxis,np.newaxis,:,:])
P_DNS = tf.convert_to_tensor(P_DNS[np.newaxis,np.newaxis,:,:])
UVP_DNS_org = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)


# save original LES field
U_LES = gfilter(U_DNS)
V_LES = gfilter(V_DNS)
P_LES = gfilter(P_DNS)
UVP_LES_org = tf.concat([U_LES, V_LES, P_LES], axis=1)


# set LES_all0
LES_all0 = []
if (RESTART_WL):

    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    filename = Z0_DIR_WL + "z0.npz"
                
    data = np.load(filename)

    z0         = data["z0"]
    LES_in0    = data["LES_in0"]
    nUVP_amaxo = data["nUVP_amaxo"]
    fUVP_amaxo = data["fUVP_amaxo"]
    noise_in   = data["noise_in"]
    
    print("z0",                 z0.shape, np.min(z0),         np.max(z0))
    print("LES_in0",       LES_in0.shape, np.min(LES_in0),    np.max(LES_in0))
    print("nUVP_amaxo", nUVP_amaxo.shape, np.min(nUVP_amaxo), np.max(nUVP_amaxo))
    print("fUVP_amaxo", fUVP_amaxo.shape, np.min(fUVP_amaxo), np.max(fUVP_amaxo))        
    print("noise_in",     noise_in.shape, np.min(noise_in),   np.max(noise_in))

    # assign variables
    z0         = tf.convert_to_tensor(z0, dtype=DTYPE)
    LES_in0    = tf.convert_to_tensor(LES_in0, dtype=DTYPE)

    UVP_max = np.concatenate([nUVP_amaxo, fUVP_amaxo], 0)

    for nvars in range(len(noise_in)):
        tnoise_in = tf.convert_to_tensor(noise_in[nvars], dtype=DTYPE)
        lcnoise.trainable_variables[nvars].assign(tnoise_in)

    # assign variable noise
    if (TUNE_NOISE):
        noise_DNS = data["noise_DNS"]
        print("noise_DNS", noise_DNS.shape, np.min(noise_DNS), np.max(noise_DNS))
        noise_DNS = tf.convert_to_tensor(noise_DNS,  dtype=DTYPE)
        it=0
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                layer.trainable_variables[0].assign(noise_DNS[it])
                it=it+1

    # set LES_in fields
    for res in range(2,RES_LOG2-FIL):
        rs = 2**(RES_LOG2-FIL-res)
        LES_all0.append(LES_in0[:,:,::rs,::rs])

else:

    for res in range(2,RES_LOG2-FIL+1):
        rs = 2**(RES_LOG2-FIL-res)
        if (res != RES_LOG2-FIL):
            U_LES = U_DNS[:,:,::rs,::rs]
            V_LES = V_DNS[:,:,::rs,::rs]
            P_LES = P_DNS[:,:,::rs,::rs]
            img_LES = tf.concat([U_LES,V_LES,P_LES], axis=1)
            LES_all0.append(img_LES)

    # set z
    z0 = tf.random.uniform(shape=[NC2_NOISE_IN, 1], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)

LES_all = [LES_all0, LES_in0]



print ("============================Completed setup!\n\n")



#------------------------------------------------------ find initial residuals
# find inference...
UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)
#... and correct it with new LES_in0
LES_in0 = normalize_max(fUVP_DNS)
LES_all = [LES_all0, LES_in0]
UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)

# find residuals
resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS_org, UVP_LES, typeRes=1)
print("\nInitial residuals ------------------------:     resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
        .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))



#------------------------------------------------------ tune to given tollerance
if (TUNE):
    it  = 0
    tstart = time.time()
    while (resREC>tollDNS and it<lr_DNS_maxIt):

        lr = lr_schedule_DNS(it)
        UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, _, predictions = \
            step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, [z0, LES_all], UVP_DNS_org, UVP_LES, ltv_DNS, UVP_max, typeRes=1)

        # write losses to tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('resREC',   resREC,   step=it)
            tf.summary.scalar('resLES',   resLES,   step=it)
            tf.summary.scalar('resDNS',   resDNS,   step=it)
            tf.summary.scalar('loss_fil', loss_fil, step=it)
            tf.summary.scalar('lr',       lr,       step=it)

        # print fields
        if (it%100==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   it {1:6d}  resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} lr {6:3e}" \
                .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy(), lr))

            if (it%1000==0):
                U_DNS = UVP_DNS[0, 0, :, :].numpy()
                V_DNS = UVP_DNS[0, 1, :, :].numpy()
                P_DNS = UVP_DNS[0, 2, :, :].numpy()

                filename = Z0_DIR_WL + "plots_StylES_it" + str(it).zfill(5) + ".png"
                print_fields_3(UVP_DNS[0,0,:,:], UVP_DNS[0,1,:,:], UVP_DNS[0,2,:,:],
                    filename=filename, testcase=TESTCASE, labels=[r"n", r"$\phi$", r"$\zeta$"], \
                    Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

                minf, maxf = find_minmax2(UVP_LES[0,0,:,:], fUVP_DNS[0,0,:,:])
                filename = Z0_DIR_WL + "plots_diffLESfDNS_n_it" + str(it).zfill(5) + ".png"
                print_fields_3(UVP_LES[0,0,:,:], fUVP_DNS[0,0,:,:], UVP_LES[0,0,:,:]-fUVP_DNS[0,0,:,:],
                    filename=filename, testcase=TESTCASE, plot='diff', labels=["LES", "fDNS", "diff"], \
                    Umin=minf, Umax=maxf, Vmin=minf, Vmax=maxf, Pmin=minf, Pmax=maxf)

                minf, maxf = find_minmax2(UVP_LES[0,1,:,:], fUVP_DNS[0,1,:,:])
                filename = Z0_DIR_WL + "plots_diffLESfDNS_phi_it" + str(it).zfill(5) + ".png"
                print_fields_3(UVP_LES[0,1,:,:], fUVP_DNS[0,1,:,:], UVP_LES[0,1,:,:]-fUVP_DNS[0,1,:,:],
                    filename=filename, testcase=TESTCASE, plot='diff', labels=["LES", "fDNS", "diff"], \
                    Umin=minf, Umax=maxf, Vmin=minf, Vmax=maxf, Pmin=minf, Pmax=maxf)

                minf, maxf = find_minmax2(UVP_LES[0,2,:,:], fUVP_DNS[0,2,:,:])    
                filename = Z0_DIR_WL + "plots_diffLESfDNS_vort_it" + str(it).zfill(5) + ".png"
                print_fields_3(UVP_LES[0,2,:,:], fUVP_DNS[0,2,:,:], UVP_LES[0,2,:,:]-fUVP_DNS[0,2,:,:],
                    filename=filename, testcase=TESTCASE, plot='diff', labels=["LES", "fDNS", "diff"], \
                    Umin=minf, Umax=maxf, Vmin=minf, Vmax=maxf, Pmin=minf, Pmax=maxf)


        # # normalize the data
        # LES_in0 = 1.0*normalize_max(fUVP_DNS) + 0.0*LES_in0
        # LES_all = [LES_all0, LES_in0]

        # move next iteration
        it = it+1

    # print final iteration
    tend = time.time()
    print("LES iterations:  time {0:3e}   it {1:6d}  resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} " \
        .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))
            


#------------------------------------------------------ save NN configuration
if (not RESTART_WL):
    managerCheckpoint_wl.save()

    # save z
    z0 = z0.numpy()

    noise_in = []
    for nvars in range(len(lcnoise.trainable_variables[:])):
        noise_in.append(lcnoise.trainable_variables[nvars].numpy())

    # load noise
    if (TUNE_NOISE):
        it=0
        noise_DNS=[]
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                noise_DNS.append(layer.trainable_variables[0].numpy())

        filename =  Z0_DIR_WL + "z0.npz"
        np.savez(filename,
                z0         = z0, \
                noise_in   = noise_in, \
                LES_in0    = LES_all[-1], \
                nUVP_amaxo = nUVP_amaxo, \
                fUVP_amaxo = fUVP_amaxo, \
                noise_DNS = noise_DNS)
    else:
        filename =  Z0_DIR_WL + "z0.npz"
        np.savez(filename,
                z0       = z0, \
                noise_in = noise_in, \
                LES_in0  = LES_all[-1], \
                nUVP_amaxo = nUVP_amaxo, \
                fUVP_amaxo = fUVP_amaxo)


#------------------------------------------------------ check, find and print fields
if (TESTCASE=='HW' or TESTCASE=='mHW'):
    print("Mean U ", tf.reduce_mean(UVP_DNS[0, 0, :, :]))
    print("Mean V ", tf.reduce_mean(UVP_DNS[0, 1, :, :]))
    print("Mean P ", tf.reduce_mean(UVP_DNS[0, 2, :, :]))


#--------------------------- find DNS, LES and filtered fields
# DNS
U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
P_DNS = UVP_DNS[0, 2, :, :].numpy()

print("Min/max values in each field :")
print(np.min(U_DNS), np.max(U_DNS), np.min(V_DNS), np.max(V_DNS), np.min(P_DNS), np.max(P_DNS))

# DNS
U_LES = UVP_LES[0, 0, :, :].numpy()
V_LES = UVP_LES[0, 1, :, :].numpy()
P_LES = UVP_LES[0, 2, :, :].numpy()

# filtered
fU_DNS = fUVP_DNS[0, 0, :, :].numpy()
fV_DNS = fUVP_DNS[0, 1, :, :].numpy()
fP_DNS = fUVP_DNS[0, 2, :, :].numpy()


#--------------------------- print final fields, differences and spectra
if (TESTCASE=='HIT_2D'):

    filename = Z0_DIR_WL + "plots.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, N=OUTPUT_DIM, filename=filename, testcase=TESTCASE)

    filename = Z0_DIR_WL + "restart"
    save_fields(0.6, U_DNS, V_DNS, P_DNS, filename=filename)  # Note: t=0.6 is the corrisponding time to t=545 tau_e

    filename = Z0_DIR_WL + "energy_spectrum.png"
    closePlot=True
    plot_spectrum_2d_3v(U_DNS, V_DNS, L, filename, close=closePlot)

elif(TESTCASE=='HW' or TESTCASE=='mHW'):

    # fields
    filename = Z0_DIR_WL + "plots_DNS.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE, \
                Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

    filename = Z0_DIR_WL + "plots_LES.png"
    print_fields_3(U_LES, V_LES, P_LES, filename=filename, testcase=TESTCASE) #, \
                #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

    filename = Z0_DIR_WL + "plots_fDNS.png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, filename=filename, testcase=TESTCASE) #, \
                #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)


    # differences
    minf, maxf = find_minmax2(U_LES, fU_DNS)
    filename = Z0_DIR_WL + "plots_diff_LES_fDNS_U.png"
    print_fields_3(U_LES, fU_DNS, fU_DNS-U_LES, filename=filename, testcase=TESTCASE, plot='diff', \
                labels=["LES", "fDNS", "diff"], \
                Umin=minf, Umax=maxf, Vmin=minf, Vmax=maxf, Pmin=minf, Pmax=maxf)

    minf, maxf = find_minmax2(V_LES, fV_DNS)
    filename = Z0_DIR_WL + "plots_diff_LES_fDNS_V.png"
    print_fields_3(V_LES, fV_DNS, fV_DNS-V_LES, filename=filename, testcase=TESTCASE, plot='diff', \
                labels=["LES", "fDNS", "diff"], \
                Umin=minf, Umax=maxf, Vmin=minf, Vmax=maxf, Pmin=minf, Pmax=maxf)

    minf, maxf = find_minmax2(P_LES, fP_DNS)
    filename = Z0_DIR_WL + "plots_diff_LES_fDNS_P.png"
    print_fields_3(P_LES, fP_DNS, fP_DNS-P_LES, filename=filename, testcase=TESTCASE, plot='diff', \
                Umin=minf, Umax=maxf, Vmin=minf, Vmax=maxf, Pmin=minf, Pmax=maxf)

    cP_DNS = (-tr(V_DNS, 2, 0) + 16*tr(V_DNS, 1, 0) - 30*V_DNS + 16*tr(V_DNS,-1, 0) - tr(V_DNS,-2, 0))/(12*DELX**2) \
           + (-tr(V_DNS, 0, 2) + 16*tr(V_DNS, 0, 1) - 30*V_DNS + 16*tr(V_DNS, 0,-1) - tr(V_DNS, 0,-2))/(12*DELY**2)
    filename = Z0_DIR_WL + "plots_diff_Phi.png"
    print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, filename=filename, testcase=TESTCASE, plot='diff')


    # spectrum
    closePlot=False
    filename = Z0_DIR_WL + "energy_spectrum_StylES.png"
    dVdx = (-cr(V_DNS, 2, 0) + 8*cr(V_DNS, 1, 0) - 8*cr(V_DNS, -1,  0) + cr(V_DNS, -2,  0))/(12.0*DELX_LES)
    dVdy = (-cr(V_DNS, 0, 2) + 8*cr(V_DNS, 0, 1) - 8*cr(V_DNS,  0, -1) + cr(V_DNS,  0, -2))/(12.0*DELY_LES)
    plot_spectrum_2d_3v(U_DNS, dVdx, dVdy, L, filename, label="StylES", close=closePlot)


    filename = Z0_DIR_WL + "energy_spectrum_LES.png"
    dVdx = (-cr(V_LES, 2, 0) + 8*cr(V_LES, 1, 0) - 8*cr(V_LES, -1,  0) + cr(V_LES, -2,  0))/(12.0*DELX_LES)
    dVdy = (-cr(V_LES, 0, 2) + 8*cr(V_LES, 0, 1) - 8*cr(V_LES,  0, -1) + cr(V_LES,  0, -2))/(12.0*DELY_LES)
    plot_spectrum_2d_3v(U_LES, dVdx, dVdy, L, filename, label="LES", close=closePlot)

    closePlot=True
    filename = Z0_DIR_WL + "energy_spectrum_fDNS.png"
    dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
    dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
    plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename, label="fDNS", close=closePlot)


print ("============================Completed tuning!\n\n")



#--------------------------- verify filter properties

# find
cf_DNS = 10.0*U_DNS  # conservation
lf_DNS = U_DNS + V_DNS  # linearity
df_DNS = ((tr(P_DNS, 1, 0) - tr(P_DNS,-1, 0))/(2*DELX)) + ((tr(P_DNS, 0, 1) - tr(P_DNS, 0,-1))/(2*DELY))  # commutative
        
cf_DNS = (gfilter(cf_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
lf_DNS = (gfilter(lf_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
df_DNS = (gfilter(df_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]

c_LES = 10.0*fU_DNS
l_LES = fU_DNS + fV_DNS
if (GAUSSIAN_FILTER):
    fP_DNS_noSca = (gfilter_noScaling(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]   # the downscaling must happens after the filtering!!
    d_LES = ((tr(fP_DNS_noSca, 1, 0) - tr(fP_DNS_noSca,-1, 0))/(2*DELX)) + ((tr(fP_DNS_noSca, 0, 1) - tr(fP_DNS_noSca, 0,-1))/(2*DELY))
    d_LES = d_LES[::RS,::RS]
else:
    d_LES = ((tr(fP_DNS, 1, 0) - tr(fP_DNS,-1, 0))/(2*DELX_LES)) + ((tr(fP_DNS, 0, 1) - tr(fP_DNS, 0,-1))/(2*DELY_LES))

# plot
filename = Z0_DIR_WL + "plots_filterProperty_conservation.png"
print("Mean error on conservation: ", tf.reduce_mean(cf_DNS-c_LES).numpy())
print_fields_3(cf_DNS, c_LES, cf_DNS-c_LES, N=res, filename=filename, testcase=TESTCASE, plot='diff')

filename = Z0_DIR_WL + "plots_filterProperty_linearity.png"
print("Mean error on linearity: ", tf.reduce_mean(lf_DNS-l_LES).numpy())
print_fields_3(lf_DNS, l_LES, lf_DNS-l_LES, N=res, filename=filename, testcase=TESTCASE, plot='diff')

filename = Z0_DIR_WL + "plots_filterProperty_derivative.png"
print("Mean error on derivative: ", tf.reduce_mean(df_DNS-d_LES).numpy())
print_fields_3(df_DNS, d_LES, df_DNS-d_LES, N=res, filename=filename, testcase=TESTCASE, plot='diff')

print ("============================Completed filter properties check!\n\n")



print ("Completed all tasks successfully!!")




# ------------- extra pieces

# ltv_gauss = []
# for variable in lgauss.trainable_variables:
#     ltv_gauss.append(variable)

# print("\n filter variables:")
# for variable in ltv_gauss:
#     print(variable.name, variable.shape)




    # #------ Sharp (spectral)
    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # out     = sharp_filter(x_in[0,0,:,:], delta=L/N_LES, size=4, rsca=RS)  # delta = pi/Kc, where Kc is the LES wave number (N_LES/2).
    # gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # out     = sharp_filter(x_in[0,0,:,:], delta=L/N_LES, size=4, rsca=1)  # delta = pi/Kc, where Kc is the LES wave number (N_LES/2).
    # gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)


    # #------ Downscaling
    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # out     = x_in[:,:,::RS,::RS]
    # gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # out     = x_in[:,:,::1,::1]
    # gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)

    # lgauss =  layer_gaussian(rs=RS, rsca=RS)

    # #------ Gaussian normalized
    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # x_max   = tf.abs(tf.reduce_max(x_in))
    # x_min   = tf.abs(tf.reduce_min(x_in))
    # xamax   = tf.maximum(x_max, x_min)
    # x       = x_in/xamax
    # x       = gaussian_filter(x[0,0,:,:], rs=RS, rsca=RS)
    # x_max   = tf.abs(tf.reduce_max(x))
    # x_min   = tf.abs(tf.reduce_min(x))
    # xamaxn  = tf.maximum(x_max, x_min)
    # out     = x/xamaxn * xamax
    # gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # x_max   = tf.abs(tf.reduce_max(x_in))
    # x_min   = tf.abs(tf.reduce_min(x_in))
    # xamax   = tf.maximum(x_max, x_min)
    # x       = x_in/xamax
    # x       = gaussian_filter(x[0,0,:,:], rs=RS, rsca=1)
    # x_max   = tf.abs(tf.reduce_max(x))
    # x_min   = tf.abs(tf.reduce_min(x))
    # xamaxn  = tf.maximum(x_max, x_min)
    # out     = x/xamaxn * xamax
    # gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)



    # loss_fill = step_find_gaussianfilter(gfilter, opt_kDNS, UVP_DNS, UVP_LES, ltv_gauss)
    # print(loss_fill)

    #kDNSo = layer_kDNS.trainable_variables[0]


    # # adjust variables
    # kDNS  = layer_kDNS.trainable_variables[0]
    # kDNSn = tf.clip_by_value(kDNS, 0.0, 1.0)
    # if (tf.reduce_any((kDNS-kDNSn)>0)):
    #     layer_kDNS.trainable_variables[0].assign(kDNSn)
    #     UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)
    #     resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, tDNS, tLES, typeRes=1)


    # # adjust variables
    # valid_z0 = True
    # kDNS     = layer_kDNS.trainable_variables[0]
    # kDNSt = tf.clip_by_value(kDNS, 0.0, 1.0)
    # if (tf.reduce_any((kDNS-kDNSt)>0) or (it2%1000==0)):  # find new left and right z0
    #     if (valid_z0):
    #         print("reset z at iteration", it)
    #         valid_z0 = False
    #         it2 = 0
    #     z0o = kDNSo*z0p + (1.0-kDNSo)*z0m
    #     z0n = tf.random.uniform(shape=[BATCH_SIZE, G_LAYERS-M_LAYERS, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
    #     z0p = (z0o+z0n)/2.0
    #     z0m = (z0o-z0n)/2.0
    #     z0  = tf.concat([z0i, z0p, z0m], axis=1)
    #     kDNSn = 0.5*tf.ones_like(kDNS)
    #     layer_kDNS.trainable_variables[0].assign(kDNSn)
    #     UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)
    #     resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, tDNS, tLES, typeRes=0)

    # kDNSo = layer_kDNS.trainable_variables[0]
