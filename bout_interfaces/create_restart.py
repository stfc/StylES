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
FILE_DNS    = FILE_DNS_N1024
TUNE        = False 
TUNE_NOISE  = False 
tollDNS     = 1e-3
RESTART_WL  = False

 
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

filename_spectra = Z0_DIR_WL + "energy_spectrum.png"

if (not RESTART_WL):
    os.system("rm -rf " + Z0_DIR_WL)
    os.system("mkdir -p " + Z0_DIR_WL)
CHKP_DIR_WL = Z0_DIR_WL + "checkpoints_wl/"



#------------------------------------------------------ define optimizer for z and wl_dlatents search
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



# create filter model
if (GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([NUM_CHANNELS, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = apply_filter_NCH(x_in, size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian')
    gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    x_in        = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out         = apply_filter(x_in[0,0,:,:], size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian')
    gfilter_1ch = tf.keras.Model(inputs=x_in, outputs=out)

    x_in              = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out               = apply_filter(x_in[0,0,:,:], size=4*RS, rsca=1, mean=0.0, delta=RS, type='Gaussian')
    gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)
else:
    gfilter = filters[IFIL]



# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    
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

# convert to tf
U_DNS = tf.convert_to_tensor(U_DNS, dtype=DTYPE)
V_DNS = tf.convert_to_tensor(V_DNS, dtype=DTYPE)
P_DNS = tf.convert_to_tensor(P_DNS, dtype=DTYPE)

if (DIMS_3D):
    U_DNS = tf.transpose(U_DNS, [1,0,2])
    V_DNS = tf.transpose(V_DNS, [1,0,2])
    P_DNS = tf.transpose(P_DNS, [1,0,2])
    U_DNS = U_DNS[:,tf.newaxis,:,:]
    V_DNS = V_DNS[:,tf.newaxis,:,:]
    P_DNS = P_DNS[:,tf.newaxis,:,:]
else:
    U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
    V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
    P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]
    
UVP_DNS_org = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)

# filter and dowscale if needed
NX_DNS = len(U_DNS[0,0,0,:])
rsin = int(NX_DNS/OUTPUT_DIM)
if (rsin>1):
    UVP_DNS     = apply_filter_NCH(UVP_DNS_org, size=4*rsin, rsca=rsin, mean=0.0, delta=rsin, type='Gaussian', NCH=3)
    U_DNS       = UVP_DNS[:,0:1,:,:]
    V_DNS       = UVP_DNS[:,1:2,:,:]
    if (USE_VORTICITY):
        P_DNS = find_vorticity_HW(V_DNS, DELX*rsin, DELY*rsin)
    else:
        P_DNS = UVP_DNS[:,2:3,:,:]
    UVP_DNS_org = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)

UVP_LES_org = apply_filter_NCH(UVP_DNS_org, size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian', NCH=3)


# print values
U_DNS = UVP_DNS_org[0,0,:,:].numpy()
V_DNS = UVP_DNS_org[0,1,:,:].numpy()
P_DNS = UVP_DNS_org[0,2,:,:].numpy()

filename = Z0_DIR_WL + "plots_DNS_org.png"
print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE), \
            #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

U_LES = UVP_LES_org[0,0,:,:].numpy()
V_LES = UVP_LES_org[0,1,:,:].numpy()
P_LES = UVP_LES_org[0,2,:,:].numpy()

filename = Z0_DIR_WL + "plots_LES_org.png"
print_fields_3(U_LES, V_LES, P_LES, filename=filename, testcase=TESTCASE) #, \
            #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)


dVdx = (-cr(V_DNS, 2, 0) + 8*cr(V_DNS, 1, 0) - 8*cr(V_DNS, -1,  0) + cr(V_DNS, -2,  0))/(12.0)
dVdy = (-cr(V_DNS, 0, 2) + 8*cr(V_DNS, 0, 1) - 8*cr(V_DNS,  0, -1) + cr(V_DNS,  0, -2))/(12.0)
plot_spectrum_2d_3v(U_DNS, dVdx, dVdy, L, filename_spectra, label="DNS", close=False)

dVdx = (-cr(V_LES, 2, 0) + 8*cr(V_LES, 1, 0) - 8*cr(V_LES, -1,  0) + cr(V_LES, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(V_LES, 0, 2) + 8*cr(V_LES, 0, 1) - 8*cr(V_LES,  0, -1) + cr(V_LES,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(U_LES, dVdx, dVdy, L, filename_spectra, label="LES_org", close=False)


# Normalize values
nUVP_DNS, nUVP_amaxo = normalize_max(UVP_DNS_org)
nUVP_LES, fUVP_amaxo = normalize_max(UVP_LES_org)
UVP_max = [nUVP_amaxo] + [fUVP_amaxo]



# filter image
if (USE_IMGSLES):
    rs = 2
    for reslog in range(RES_LOG2, RES_LOG2-FIL-1, -1):
        res = 2**reslog
        if (reslog==RES_LOG2):
            fUVP_DNS = nUVP_DNS
        else:
            fUVP_DNS = apply_filter_NCH(fUVP_DNS, size=4, rsca=rs, mean=0.0, delta=1.0, type='Gaussian', NCH=3)
            U_DNS    = fUVP_DNS[:,0:1,:,:]
            V_DNS    = fUVP_DNS[:,1:2,:,:]
            if (USE_VORTICITY):        
                P_DNS    = find_vorticity_HW(V_DNS, DELX*OUTPUT_DIM/res, DELY*OUTPUT_DIM/res)
                fUVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)
            else:
                P_DNS    = fUVP_DNS[:,2:3,:,:]
            fUVP_DNS = find_centred_fields(fUVP_DNS)
            fUVP_DNS, _ = normalize_max(fUVP_DNS)

        # normalize the data
        fUVP_DNS, _ = normalize_max(fUVP_DNS)
        
    # save LES_in0
    LES_in0 = tf.identity(fUVP_DNS)


# use single channel
LES_in0 = LES_in0[:,1:2,:,:]
print("aaa")
exit()


# set LES_all0
LES_all0 = []
if (RESTART_WL):

    filename = Z0_DIR_WL + "z0.npz"
                
    data = np.load(filename)

    z0         = data["z0"]
    dlatents   = data["dlatents"]
    LES_in0    = data["LES_in0"]
    nUVP_amaxo = data["nUVP_amaxo"]
    fUVP_amaxo = data["fUVP_amaxo"]
    
    print("z0",                 z0.shape, np.min(z0),         np.max(z0))
    print("dlatents",     dlatents.shape, np.min(dlatents),   np.max(dlatents))
    print("LES_in0",       LES_in0.shape, np.min(LES_in0),    np.max(LES_in0))
    print("nUVP_amaxo", nUVP_amaxo.shape, np.min(nUVP_amaxo), np.max(nUVP_amaxo))
    print("fUVP_amaxo", fUVP_amaxo.shape, np.min(fUVP_amaxo), np.max(fUVP_amaxo))        

    # assign variables
    z0         = tf.convert_to_tensor(z0, dtype=DTYPE)
    LES_in0    = tf.convert_to_tensor(LES_in0, dtype=DTYPE)

    UVP_max = np.concatenate([nUVP_amaxo, fUVP_amaxo], 0)

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

    # set z
    if (DIMS_3D):
        z0a = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
        z0b = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
        z0 = z0a
        for i in range(1,BATCH_SIZE):
            wi = np.cos(i/float(BATCH_SIZE-1)*2.0*np.pi)
            zi = z0a*wi + z0b*(1.0-wi)
            z0 = tf.concat([z0, zi], axis=0)
    else:
        z0 = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
        
    dlatents = mapping(z0, training=False)

    if (not USE_IMGSLES):
        pre_img  = pre_synthesis(dlatents, training = False)
        LES_in0  = pre_img[-1] 

    for res in range(2,RES_LOG2-FIL+1):
        rs = 2**(RES_LOG2-FIL-res)
        if (res != RES_LOG2-FIL):
            LES_all0.append(LES_in0[:,:,::rs,::rs])

LES_all = [LES_all0, LES_in0]


print ("============================Completed setup!\n\n")



#------------------------------------------------------ find initial residuals
# find inference...
UVP_DNS, UVP_LES, fUVP_DNS, _ = find_predictions(synthesis, gfilter, [dlatents, LES_all0, LES_in0], UVP_max)

# #... and correct it with new LES_in0
# LES_in0, _ = normalize_max(fUVP_DNS)
# LES_all = [LES_all0, LES_in0]
# UVP_DNS, UVP_LES, fUVP_DNS, _ = find_predictions(synthesis, gfilter, [z0, LES_all], UVP_max)


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
        # LES_in0 = 1.0*normalize_max(fUVP_DNS)[0] + 0.0*LES_in0
        # LES_all = [LES_all0, LES_in0]

        # move next iteration
        it = it+1

    # print final iteration
    tend = time.time()
    print("LES iterations:  time {0:3e}   it {1:6d}  resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} " \
        .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))
            


#------------------------------------------------------ save NN configuration
if (not RESTART_WL):

    # save z
    z0 = z0.numpy()

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
                dlatents   = dlatents, \
                LES_in0    = LES_all[-1], \
                nUVP_amaxo = nUVP_amaxo, \
                fUVP_amaxo = fUVP_amaxo, \
                noise_DNS = noise_DNS)
    else:
        filename =  Z0_DIR_WL + "z0.npz"
        np.savez(filename,
                z0       = z0, \
                dlatents   = dlatents, \
                LES_in0  = LES_all[-1], \
                nUVP_amaxo = nUVP_amaxo, \
                fUVP_amaxo = fUVP_amaxo)


#------------------------------------------------------ check, find and print fields
if (TESTCASE=='HW' or TESTCASE=='mHW'):
    print("Mean U ", tf.reduce_mean(UVP_DNS[:, 0, :, :]))
    print("Mean V ", tf.reduce_mean(UVP_DNS[:, 1, :, :]))
    print("Mean P ", tf.reduce_mean(UVP_DNS[:, 2, :, :]))


#--------------------------- find mean and min/max values
print("Min/max values in each field :")
print(np.min(UVP_DNS[0, 0, :, :].numpy()), np.min(UVP_DNS[0, 1, :, :].numpy()), np.min(UVP_DNS[0, 2, :, :].numpy()), \
      np.max(UVP_DNS[0, 0, :, :].numpy()), np.max(UVP_DNS[0, 1, :, :].numpy()), np.max(UVP_DNS[0, 2, :, :].numpy()))



#--------------------------- plot final fields, differences and spectra
# DNS
U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
P_DNS = UVP_DNS[0, 2, :, :].numpy()

# DNS
U_LES = UVP_LES[0, 0, :, :].numpy()
V_LES = UVP_LES[0, 1, :, :].numpy()
P_LES = UVP_LES[0, 2, :, :].numpy()

# filtered
fU_DNS = fUVP_DNS[0, 0, :, :].numpy()
fV_DNS = fUVP_DNS[0, 1, :, :].numpy()
fP_DNS = fUVP_DNS[0, 2, :, :].numpy()

if (TESTCASE=='HIT_2D'):

    filename = Z0_DIR_WL + "plots.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, N=OUTPUT_DIM, filename=filename, testcase=TESTCASE)

    filename = Z0_DIR_WL + "restart"
    save_fields(0.6, U_DNS, V_DNS, P_DNS, filename=filename)  # Note: t=0.6 is the corrisponding time to t=545 tau_e

    filename = Z0_DIR_WL + "energy_spectrum.png"
    plot_spectrum_2d_3v(U_DNS, V_DNS, L, filename, close=True)

elif(TESTCASE=='HW' or TESTCASE=='mHW'):

    # fields
    filename = Z0_DIR_WL + "plots_DNS.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE) #, \
                #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

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

    cP_DNS = (-cr(V_DNS, 2, 0) + 16*cr(V_DNS, 1, 0) - 30*V_DNS + 16*cr(V_DNS,-1, 0) - cr(V_DNS,-2, 0))/(12*DELX**2) \
           + (-cr(V_DNS, 0, 2) + 16*cr(V_DNS, 0, 1) - 30*V_DNS + 16*cr(V_DNS, 0,-1) - cr(V_DNS, 0,-2))/(12*DELY**2)
    filename = Z0_DIR_WL + "plots_diff_Phi.png"
    print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, filename=filename, testcase=TESTCASE, plot='diff')


    # spectra
    filename = Z0_DIR_WL + "energy_spectrum_StylES.png"
    dVdx = (-cr(V_DNS, 2, 0) + 8*cr(V_DNS, 1, 0) - 8*cr(V_DNS, -1,  0) + cr(V_DNS, -2,  0))/(12.0*DELX_LES)
    dVdy = (-cr(V_DNS, 0, 2) + 8*cr(V_DNS, 0, 1) - 8*cr(V_DNS,  0, -1) + cr(V_DNS,  0, -2))/(12.0*DELY_LES)
    plot_spectrum_2d_3v(U_DNS, dVdx, dVdy, L, filename, label="StylES", close=False)

    filename = Z0_DIR_WL + "energy_spectrum_LES.png"
    dVdx = (-cr(V_LES, 2, 0) + 8*cr(V_LES, 1, 0) - 8*cr(V_LES, -1,  0) + cr(V_LES, -2,  0))/(12.0*DELX_LES)
    dVdy = (-cr(V_LES, 0, 2) + 8*cr(V_LES, 0, 1) - 8*cr(V_LES,  0, -1) + cr(V_LES,  0, -2))/(12.0*DELY_LES)
    plot_spectrum_2d_3v(U_LES, dVdx, dVdy, L, filename, label="LES", close=False)

    filename = Z0_DIR_WL + "energy_spectrum_fDNS.png"
    dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
    dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
    plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename, label="fDNS", close=True)


print ("============================Completed tuning!\n\n")



#--------------------------- verify filter properties

# find
cf_DNS = 10.0*U_DNS  # conservation
lf_DNS = U_DNS + V_DNS  # linearity
df_DNS = ((cr(P_DNS, 1, 0) - cr(P_DNS,-1, 0))/(2*DELX)) + ((cr(P_DNS, 0, 1) - cr(P_DNS, 0,-1))/(2*DELY))  # commutative
        
cf_DNS = (gfilter_1ch(cf_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
lf_DNS = (gfilter_1ch(lf_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
df_DNS = (gfilter_1ch(df_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]

c_LES = 10.0*fU_DNS
l_LES = fU_DNS + fV_DNS
if (GAUSSIAN_FILTER):
    fP_DNS_noSca = (gfilter_noScaling(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]   # the downscaling must happens after the filtering!!
    d_LES = ((cr(fP_DNS_noSca, 1, 0) - cr(fP_DNS_noSca,-1, 0))/(2*DELX)) + ((cr(fP_DNS_noSca, 0, 1) - cr(fP_DNS_noSca, 0,-1))/(2*DELY))
    d_LES = d_LES[::RS,::RS]
else:
    d_LES = ((cr(fP_DNS, 1, 0) - cr(fP_DNS,-1, 0))/(2*DELX_LES)) + ((cr(fP_DNS, 0, 1) - cr(fP_DNS, 0,-1))/(2*DELY_LES))

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
