#----------------------------------------------------------------------------------------------
#
#    Copyright (C): 2022 UKRI-STFC (Hartree Centre)
#
#    Author: Jony Castagna, Francesca Schiavello, Josh Williams
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
if (NDIMS==2):
    FILE_DNS = FILE_DNS_N256
elif (NDIMS==3):
    FILE_DNS = FILE_DNS_N1024_3D
TUNE        = True
TUNE_NOISE  = False 
tollDNS     = 1.0e-3
RESTART_WL  = False
CHKP_DIR_WL = "./checkpoints_wl"
 
# check that randomization is off
if (RANDOMIZE_NOISE):
    print("Carefull! Noise randomization is on!! Swith it to off in ../parameters.py")
    exit()

# set folders and paths
if (TESTCASE=='HIT_2D'):
    from HIT_2D import L
    os.system("mkdir -p ../LES_Solvers/restart_fromGAN/")
    Z0_DIR_WL = "../LES_Solvers/restart_fromGAN/"
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = LEN_DOMAIN
    os.system("mkdir -p ../bout_interfaces/restart_fromGAN/logs/")
    os.system("rm ../bout_interfaces/restart_fromGAN/*.png")
    os.system("rm ../bout_interfaces/restart_fromGAN/*.txt")
    Z0_DIR_WL      = "../bout_interfaces/restart_fromGAN/"
    Z0_DIR_WL_LOGS = "../bout_interfaces/restart_fromGAN/logs/"

filename_spectra = Z0_DIR_WL + "energy_spectra.png"

if (not RESTART_WL):
    os.system("rm -rf " + Z0_DIR_WL)
    os.system("mkdir -p " + Z0_DIR_WL)


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint_StylES, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)

train_summary_writer = tf.summary.create_file_writer(Z0_DIR_WL_LOGS + "search/")


# create filter model
if (GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([NUM_CHANNELS, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = apply_filter_NCH(x_in, size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian', NCH=NUM_CHANNELS)
    gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    x_1ch       = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out_1ch     = apply_filter_NCH(x_1ch, size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian', NCH=1)
    gfilter_1ch = tf.keras.Model(inputs=x_1ch, outputs=out_1ch)

    x_sub       = tf.keras.Input(shape=([1, RS+1, RS+1]), dtype=DTYPE)
    out_sub     = apply_filter_NCH(x_sub, size=4*RS, rsca=1, mean=0.0, delta=RS, subsection=True, type='Gaussian', NCH=1)
    gfilter_sub = tf.keras.Model(inputs=x_sub, outputs=out_sub)
    
    x_noScaling       = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out_noScaling     = apply_filter_NCH(x_noScaling, size=4*RS, rsca=1, mean=0.0, delta=RS, type='Gaussian', NCH=1)
    gfilter_noScaling = tf.keras.Model(inputs=x_noScaling, outputs=out_noScaling)
else:
    gfilter = filters[IFIL]
    

#------------------------------------------------------ define variables for latents space interpolation
# create variable synthesis model
wl_w0 = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
wl_w1 = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
wl_img = []
for res in range(2,RES_LOG2-FIL+1):
    wl_img.append(tf.keras.Input(shape=([NUM_CHANNELS, 2**res, 2**res]), dtype=DTYPE))
if (NUM_CHANNELS==1):
    wl_LES = tf.keras.Input(shape=([1, 2**res, 2**res]), dtype=DTYPE)
else:
    wl_LES = tf.keras.Input(shape=([NUM_CHANNELS, 2**res, 2**res]), dtype=DTYPE)
layer_LES    = layer_wlatent_mLES()
wi_in        = layer_LES(wl_w0, wl_w1)
outputs      = synthesis([wi_in, wl_img, wl_LES], training=False)
wl_synthesis = tf.keras.Model(inputs=[wl_w0, wl_w1, wl_img, wl_LES], outputs=outputs)


# optimizer
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_DNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_DNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_kDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_DNS)


# define checkpoints wl_synthesis and filter
checkpoint_wl        = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []

for variable in layer_LES.trainable_variables:
    ltv_DNS.append(variable)
    
print("\n kDNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)


time.sleep(3)


print("============================Finished initialization")




#------------------------------------------------------ set reference DNS and LES
# set z
z0 = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
dlatents = mapping(z0, training=False)

z1 = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
dlatents_z1 = mapping(z1, training=False)

if (LOAD_DNS):
    
    # load numpy array
    U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_DNS)
    U_DNS = np.cast[DTYPE](U_DNS)
    V_DNS = np.cast[DTYPE](V_DNS)
    P_DNS = np.cast[DTYPE](P_DNS)

    NX_DNS = U_DNS.shape[0]
    NY_DNS = U_DNS.shape[1]

    if (NDIMS==3):
        NZ_DNS = U_DNS.shape[2]
        print("Dimensions of original file are: ", NX_DNS, NY_DNS, NZ_DNS)
    else:
        print("Dimensions of original file are: ", NX_DNS, 1, NY_DNS)

    # reduce dimension in y direction if needed
    RSY = int(U_DNS.shape[1]/BATCH_SIZE)
    if (RSY>1):
        if (NDIMS==3):
            U_DNS = U_DNS[:,::RSY,:]
            V_DNS = V_DNS[:,::RSY,:]
            P_DNS = P_DNS[:,::RSY,:]
            NX_DNS = U_DNS.shape[0]
            NY_DNS = U_DNS.shape[1]
            NZ_DNS = U_DNS.shape[2]
            print("Dimensions of current DNS are:  ", NX_DNS, NY_DNS, NZ_DNS)
        else:
            print("Dimensions of current DNS are:  ", NX_DNS, 1, NY_DNS)
            
    
    # convert to tf
    U_DNS = tf.convert_to_tensor(U_DNS, dtype=DTYPE)
    V_DNS = tf.convert_to_tensor(V_DNS, dtype=DTYPE)
    P_DNS = tf.convert_to_tensor(P_DNS, dtype=DTYPE)

    if (NDIMS==3):
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
    rsin = int(NX_DNS/OUTPUT_DIM)
    if (rsin>1):
        UVP_DNS     = apply_filter_NCH(UVP_DNS_org, size=4*rsin, rsca=rsin, mean=0.0, delta=rsin, type='Gaussian', NCH=3)
        U_DNS       = UVP_DNS[:,0:1,:,:]
        V_DNS       = UVP_DNS[:,1:2,:,:]
        if (CALC_VORTICITY):
            P_DNS  = apply_filter_NCH(V_DNS, size=2, rsca=1, mean=0.0, delta=DELX*rsin, type='Vorticity', NCH=1)
        else:
            P_DNS = UVP_DNS[:,2:3,:,:]
        UVP_DNS_org = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)

    UVP_LES_org = gfilter(UVP_DNS_org)
    if (CALC_VORTICITY):
        P_DNS = find_vorticity_HW(UVP_LES_org[:,1:2,:,:], DELX_LES, DELY_LES)
        UVP_LES_org = tf.concat([UVP_LES_org[:,0:2,:,:], P_DNS], axis=1)  
        
    # save LES_in0
    LES_in0, _ = normalize_max(UVP_LES_org)

    # find new scaling
    fnUVPo, nfUVPo, fUVP_amaxo, nUVP_amaxo = find_scaling(UVP_DNS_org, gfilter_1ch, subSection=False, findNewValues=False)
    UVP_max = [nUVP_amaxo] + [fUVP_amaxo]

else:

    # set UVP max
    UVP_max = tf.constant([INIT_SCA], dtype=DTYPE)
    UVP_max = tf.tile(UVP_max, [3])
    UVP_max = UVP_max[tf.newaxis, :, tf.newaxis, tf.newaxis]
    UVP_max = [UVP_max] + [UVP_max]

    # inference
    if (NUM_CHANNELS==1):
        pre_img, V_LES  = pre_synthesis(dlatents, training = False)
        U_LES = V_LES
        P_LES = apply_filter_NCH(V_LES, size=2, rsca=1, mean=0.0, delta=DELX*RS, type='Vorticity', NCH=1)
        P_LES = find_centred_fields(P_LES)
        P_LES, _ = normalize_max(P_LES)
        LES_in0 = tf.concat([U_LES, V_LES, P_LES], axis=1)
        zAll = [dlatents, dlatents_z1, pre_img, LES_in0]
    else:
        pre_img = pre_synthesis(dlatents, training = False)
        LES_in0 = pre_img[-1]
        zAll    = [dlatents, dlatents_z1, pre_img, LES_in0]

    if (NDIMS==3):
        LES_in0_init = LES_in0[0:1,:,:,:]
        LES_in0      = LES_in0_init
        for j in range(1, BATCH_SIZE):
            sinLES = int(np.sin(2*np.pi*j/float(BATCH_SIZE-1))*N_LES2)
            sinDNS = sinLES*RS
            LES_in0 = tf.concat([LES_in0, tr(LES_in0_init, 0, sinLES)], axis=0)

    UVP_DNS, _, _ = wl_find_predictions(wl_synthesis, gfilter, zAll, UVP_max)
    
    fnUVPo, nfUVPo, fUVP_amaxo, nUVP_amaxo = find_scaling(UVP_DNS, gfilter_1ch, subSection=False, findNewValues=False)
    UVP_max = [nUVP_amaxo] + [fUVP_amaxo]
        
    UVP_DNS_org, UVP_LES_org, fUVP_DNS = wl_find_predictions(wl_synthesis, gfilter, zAll, UVP_max)


# find min/max values
Umax = abs(tf.reduce_max(nUVP_amaxo[:,0,:,:]).numpy())
Vmax = abs(tf.reduce_max(nUVP_amaxo[:,1,:,:]).numpy())
Pmax = abs(tf.reduce_max(nUVP_amaxo[:,2,:,:]).numpy())
Umin = -Umax
Vmin = -Vmax
Pmin = -Pmax


# print original plots DNS and LES
U_DNS = UVP_DNS_org[0,0,:,:].numpy()
V_DNS = UVP_DNS_org[0,1,:,:].numpy()
P_DNS = UVP_DNS_org[0,2,:,:].numpy()

filename = Z0_DIR_WL + "plots_DNS_org.png"
print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE, \
            Umin=Umin, Umax=Umax, Vmin=Vmin, Vmax=Vmax, Pmin=Pmin, Pmax=Pmax)

U_LES = UVP_LES_org[0,0,:,:].numpy()
V_LES = UVP_LES_org[0,1,:,:].numpy()
P_LES = UVP_LES_org[0,2,:,:].numpy()

filename = Z0_DIR_WL + "plots_LES_org.png"
print_fields_3(U_LES, V_LES, P_LES, filename=filename, testcase=TESTCASE, \
            Umin=Umin, Umax=Umax, Vmin=Vmin, Vmax=Vmax, Pmin=Pmin, Pmax=Pmax)


dVdx = (-cr(V_DNS, 2, 0) + 8*cr(V_DNS, 1, 0) - 8*cr(V_DNS, -1,  0) + cr(V_DNS, -2,  0))/(12.0*DELX)
dVdy = (-cr(V_DNS, 0, 2) + 8*cr(V_DNS, 0, 1) - 8*cr(V_DNS,  0, -1) + cr(V_DNS,  0, -2))/(12.0*DELY)
plot_spectrum_2d_3v(U_DNS, dVdx, dVdy, L, filename_spectra, label="DNS(org)", close=False)

dVdx = (-cr(V_LES, 2, 0) + 8*cr(V_LES, 1, 0) - 8*cr(V_LES, -1,  0) + cr(V_LES, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(V_LES, 0, 2) + 8*cr(V_LES, 0, 1) - 8*cr(V_LES,  0, -1) + cr(V_LES,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(U_LES, dVdx, dVdy, L, filename_spectra, label="LES(org)", close=False)


print("============================Set reference DNS and LES")





# set LES_all0
LES_all  = []
LES_all0 = []
if (RESTART_WL):

    filename = Z0_DIR_WL + "z0.npz"
                
    data = np.load(filename)

    z0          = data["z0"]
    dlatents    = data["dlatents"]
    dlatents_z1 = data["dlatents_z1"]
    LES_in0     = data["LES_in0"]
    nUVP_amaxo  = data["nUVP_amaxo"]
    fUVP_amaxo  = data["fUVP_amaxo"]
    mLES        = data["mLES"]
    
    UVP_max = [nUVP_amaxo] + [fUVP_amaxo]
    
    print("z0",                 z0.shape, np.min(z0),          np.max(z0))
    print("dlatents",     dlatents.shape, np.min(dlatents),    np.max(dlatents))
    print("dlatents_z1",  dlatents.shape, np.min(dlatents_z1), np.max(dlatents_z1))
    print("LES_in0",       LES_in0.shape, np.min(LES_in0),     np.max(LES_in0))
    print("nUVP_amaxo", nUVP_amaxo.shape, np.min(nUVP_amaxo),  np.max(nUVP_amaxo))
    print("fUVP_amaxo", fUVP_amaxo.shape, np.min(fUVP_amaxo),  np.max(fUVP_amaxo))        
    print("mLES",             mLES.shape, np.min(mLES),        np.max(mLES)) 

    # assign variables
    z0          = tf.convert_to_tensor(z0, dtype=DTYPE)
    dlatents    = tf.convert_to_tensor(dlatents, dtype=DTYPE)
    dlatents_z1 = tf.convert_to_tensor(dlatents_z1, dtype=DTYPE)
    LES_in0     = tf.convert_to_tensor(LES_in0, dtype=DTYPE)
    nUVP_amaxo  = tf.convert_to_tensor(nUVP_amaxo, dtype=DTYPE)
    fUVP_amaxo  = tf.convert_to_tensor(fUVP_amaxo, dtype=DTYPE)
    mLES        = tf.convert_to_tensor(mLES, dtype=DTYPE)

    # assign kDNS
    layer_LES.trainable_variables[0].assign(mLES)

    # set LES_in fields
    for res in range(2,RES_LOG2-FIL+1):
        rs = 2**(RES_LOG2-FIL-res)
        if (res!=RES_LOG2-FIL):
            LES_all0.append(LES_in0[:,:,::rs,::rs])
        LES_all.append(LES_in0[:,:,::rs,::rs])

else:

    for res in range(2,RES_LOG2-FIL+1):
        rs = 2**(RES_LOG2-FIL-res)
        if (res != RES_LOG2-FIL):
            LES_all0.append(LES_in0[:,:,::rs,::rs])
        LES_all.append(LES_in0[:,:,::rs,::rs])


print("Shape of dlatents, LES_all, LES_in0: ", dlatents.shape, len(LES_all), LES_in0.shape)
print ("============================Completed setup!\n\n")


    
    
#------------------------------------------------------ find initial residuals
if (LOAD_DNS):
    # find inference...
    zAll = [dlatents, dlatents_z1, LES_all, LES_in0]
    UVP_DNS, UVP_LES, fUVP_DNS = wl_find_predictions(wl_synthesis, gfilter, zAll, UVP_max)
else:
    # UVP_DNS  = UVP_DNS_org
    # UVP_LES  = UVP_LES_org
    # fUVP_DNS = UVP_LES_org
    #... and correct it with new LES_in0
    LES_in0, _ = normalize_max(fUVP_DNS)
    zAll = [dlatents, dlatents_z1, LES_all, LES_in0]
    UVP_DNS, UVP_LES, fUVP_DNS = wl_find_predictions(wl_synthesis, gfilter, zAll, UVP_max)


# find residuals
resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS_org, UVP_LES_org, typeRes=0)
print("\nInitial residuals ------------------------:     resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
        .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))



#------------------------------------------------------ tune to given tollerance
if (TUNE):
    it  = 0
    tstart = time.time()
    while (resREC>tollDNS and it<lr_DNS_maxIt):

        # find gradients
        lr = lr_schedule_DNS(it)
        UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil = \
            step_find_dlatents_mDNS(wl_synthesis, gfilter, opt_kDNS, zAll, UVP_DNS, UVP_LES, ltv_DNS, UVP_max, typeRes=0)

        # check if new alpha is out of bounds
        ioutbounds = False
        mLES = layer_LES.trainable_variables[0]

        if (it==0):
            mLESo = mLES

        if (tf.reduce_any(mLES>1) or tf.reduce_any(mLES<0)):
            ioutbounds = True
        
        if (ioutbounds):
            
            # print("Find new z are values are out of bounds!")
            z2 = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
            dlatents_z2 = mapping(z2, training=False)
            if (tf.reduce_any(mLES[0,:]>1) or tf.reduce_any(mLES[0,:]<0)):
                dlatents_new    = mLES[0,:]*dlatents[:,M_LAYERS:M_LAYERS+1,:] \
                    + (1.0-mLES[0,:])*dlatents_z1[:,M_LAYERS:M_LAYERS+1,:] # Note: we should use mLESo here to avoid extrapolation
                dlatents_z1_new = dlatents_z2[:,M_LAYERS:M_LAYERS+1,:]     # but this leads quickly to a minimal value as it takes
                mLES_new = tf.ones_like(mLESo[0:1,:], dtype=DTYPE)         # longer and longer to find a random W via random z!
            else:                                                          # Instead, the small extrapolation improves converges.
                dlatents_new    = dlatents[:,M_LAYERS:M_LAYERS+1,:]
                dlatents_z1_new = dlatents_z1[:,M_LAYERS:M_LAYERS+1,:]
                mLES_new        = mLES[0:1,:]

            for ilay in range(1,G_LAYERS-M_LAYERS):
                ii = M_LAYERS+ilay
                if (tf.reduce_any(mLES[ilay,:]>1) or tf.reduce_any(mLES[ilay,:]<0)):
                    wa    = mLES[ilay,:]*dlatents[:,ii:ii+1,:] + (1.0-mLES[ilay,:])*dlatents_z1[:,ii:ii+1,:]
                    wb    = dlatents_z2[:,ii:ii+1,:]
                    mLESn = tf.ones_like(mLESo[ilay:ilay+1,:], dtype=DTYPE)
                else:
                    wa    = dlatents[:,ii:ii+1,:]
                    wb    = dlatents_z1[:,ii:ii+1,:]
                    mLESn = mLES[ilay:ilay+1,:]
                dlatents_new    = tf.concat([dlatents_new,    wa], axis=1)
                dlatents_z1_new = tf.concat([dlatents_z1_new, wb], axis=1)
                mLES_new        = tf.concat([mLES_new,     mLESn], axis=0)
        
            dlatents    = tf.concat([   dlatents[:,0:M_LAYERS,:],dlatents_new],    axis=1)
            dlatents_z1 = tf.concat([dlatents_z1[:,0:M_LAYERS,:],dlatents_z1_new], axis=1)

            zAll  = [dlatents, dlatents_z1, LES_all, LES_in0]
            layer_LES.trainable_variables[0].assign(mLES_new)
            mLESo = mLES_new

        elif (it%100==0 and it>0):

            print("Find new z!")
            dlatents_new    = mLESo*dlatents[:,M_LAYERS:G_LAYERS,:] + (1.0-mLESo)*dlatents_z1[:,M_LAYERS:G_LAYERS,:]
            z2              = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
            dlatents_z1_new = mapping(z2, training=False)
            mLES_new        = 0.5*tf.ones_like(mLESo, dtype=DTYPE)
            
            dlatents_new2    = dlatents_new + dlatents_z1_new[:,M_LAYERS:G_LAYERS,:]
            dlatents_z1_new2 = dlatents_new - dlatents_z1_new[:,M_LAYERS:G_LAYERS,:]

            dlatents    = tf.concat([   dlatents[:,0:M_LAYERS,:],dlatents_new2],    axis=1)
            dlatents_z1 = tf.concat([dlatents_z1[:,0:M_LAYERS,:],dlatents_z1_new2], axis=1)

            zAll  = [dlatents, dlatents_z1, LES_all, LES_in0]
            layer_LES.trainable_variables[0].assign(mLES_new)
            mLESo = mLES_new
        
        else:

            mLESo = mLES

        # find new residuals
        UVP_DNS, UVP_LES, fUVP_DNS = wl_find_predictions(wl_synthesis, gfilter, zAll, UVP_max)
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS_org, UVP_LES_org, typeRes=0)


        # write losses to tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('resREC',   resREC,   step=it)
            tf.summary.scalar('resLES',   resLES,   step=it)
            tf.summary.scalar('resDNS',   resDNS,   step=it)
            tf.summary.scalar('loss_fil', loss_fil, step=it)
            tf.summary.scalar('lr',       lr,       step=it)

        # print fields
        if (it%1==0):
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
                    Umin=Umin, Umax=Umax, Vmin=Vmin, Vmax=Vmax, Pmin=Pmin, Pmax=Pmax)

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
    LES_in0     = LES_all[-1].numpy()
    mLES        = layer_LES.trainable_variables[0].numpy()

    # load data
    filename =  Z0_DIR_WL + "z0.npz"
    np.savez(filename,
            z0          = z0, \
            dlatents    = dlatents, \
            dlatents_z1 = dlatents_z1, \
            LES_in0     = LES_in0, \
            nUVP_amaxo  = nUVP_amaxo, \
            fUVP_amaxo  = fUVP_amaxo, \
            mLES        = mLES)


#------------------------------------------------------ check, find and print fields
if (TESTCASE=='HW' or TESTCASE=='mHW'):
    print("Mean U ", tf.reduce_mean(UVP_DNS[:, 0, :, :]))
    print("Mean V ", tf.reduce_mean(UVP_DNS[:, 1, :, :]))
    print("Mean P ", tf.reduce_mean(UVP_DNS[:, 2, :, :]))


#--------------------------- find mean and min/max values
print("Min/max values in each field :")
print("Umin Umax ", np.min(UVP_DNS[:, 0, :, :].numpy()), np.max(UVP_DNS[:, 0, :, :].numpy()))
print("Vmin Vmax ", np.min(UVP_DNS[:, 1, :, :].numpy()), np.max(UVP_DNS[:, 1, :, :].numpy()))
print("Pmin Pmax ", np.min(UVP_DNS[:, 2, :, :].numpy()), np.max(UVP_DNS[:, 2, :, :].numpy()))



#--------------------------- plot final fields, differences and spectra
# DNS
U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
P_DNS = UVP_DNS[0, 2, :, :].numpy()

# LES
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
    print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE, \
                Umin=Umin, Umax=Umax, Vmin=Vmin, Vmax=Vmax, Pmin=Pmin, Pmax=Pmax)

    filename = Z0_DIR_WL + "plots_LES.png"
    print_fields_3(U_LES, V_LES, P_LES, filename=filename, testcase=TESTCASE, \
                Umin=Umin, Umax=Umax, Vmin=Vmin, Vmax=Vmax, Pmin=Pmin, Pmax=Pmax)

    filename = Z0_DIR_WL + "plots_fDNS.png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, filename=filename, testcase=TESTCASE, \
                Umin=Umin, Umax=Umax, Vmin=Vmin, Vmax=Vmax, Pmin=Pmin, Pmax=Pmax)


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
                labels=["LES", "fDNS", "diff"], \
                Umin=minf, Umax=maxf, Vmin=minf, Vmax=maxf, Pmin=minf, Pmax=maxf)

    cP_DNS = (-cr(V_DNS, 2, 0) + 16*cr(V_DNS, 1, 0) - 30*V_DNS + 16*cr(V_DNS,-1, 0) - cr(V_DNS,-2, 0))/(12*DELX**2) \
           + (-cr(V_DNS, 0, 2) + 16*cr(V_DNS, 0, 1) - 30*V_DNS + 16*cr(V_DNS, 0,-1) - cr(V_DNS, 0,-2))/(12*DELY**2)
    filename = Z0_DIR_WL + "plots_diff_Phi.png"
    print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, filename=filename, testcase=TESTCASE, plot='diff')


    # spectra
    dVdx = (-cr(V_DNS, 2, 0) + 8*cr(V_DNS, 1, 0) - 8*cr(V_DNS, -1,  0) + cr(V_DNS, -2,  0))/(12.0*DELX)
    dVdy = (-cr(V_DNS, 0, 2) + 8*cr(V_DNS, 0, 1) - 8*cr(V_DNS,  0, -1) + cr(V_DNS,  0, -2))/(12.0*DELY)
    plot_spectrum_2d_3v(U_DNS, dVdx, dVdy, L, filename_spectra, label="StylES", close=False)

    dVdx = (-cr(V_LES, 2, 0) + 8*cr(V_LES, 1, 0) - 8*cr(V_LES, -1,  0) + cr(V_LES, -2,  0))/(12.0*DELX_LES)
    dVdy = (-cr(V_LES, 0, 2) + 8*cr(V_LES, 0, 1) - 8*cr(V_LES,  0, -1) + cr(V_LES,  0, -2))/(12.0*DELY_LES)
    plot_spectrum_2d_3v(U_LES, dVdx, dVdy, L, filename_spectra, label="LES", close=False)

    dVdx = (-cr(fV_DNS, 2, 0) + 8*cr(fV_DNS, 1, 0) - 8*cr(fV_DNS, -1,  0) + cr(fV_DNS, -2,  0))/(12.0*DELX_LES)
    dVdy = (-cr(fV_DNS, 0, 2) + 8*cr(fV_DNS, 0, 1) - 8*cr(fV_DNS,  0, -1) + cr(fV_DNS,  0, -2))/(12.0*DELY_LES)
    plot_spectrum_2d_3v(fU_DNS, dVdx, dVdy, L, filename_spectra, label="fDNS", close=True)


print ("============================Completed tuning!\n\n")


plt.close()
DNS_alongY = UVP_DNS[:, :, N_DNS2, N_DNS2].numpy()
plt.plot(DNS_alongY[:,0])
plt.plot(DNS_alongY[:,1])
plt.plot(DNS_alongY[:,2])
plt.savefig(Z0_DIR_WL + "plots_DNS_alongY.png")
plt.close()

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
# from tensorflow.python.compiler.tensorrt import trt_convert as trt

# checkpoint_StylES = tf.train.Checkpoint(synthesis=synthesis)
# managerCheckpoint_StylES = tf.train.CheckpointManager(checkpoint_StylES, '../' + CHKP_DIR_MIN, max_to_keep=1)
# managerCheckpoint_StylES.save()

# # Conversion Parameters 
# conversion_params = trt.TrtConversionParams()

# converter = trt.TrtGraphConverterV2(
#     input_saved_model_dir= '../' + CHKP_DIR_MIN,
#     conversion_params=conversion_params)

# # Converter method used to partition and optimize TensorRT compatible segments
# converter.convert()

# # Optionally, build TensorRT engines before deployment to save time at runtime
# # Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime
# converter.build(input_fn=my_input_fn)

# # Save the model to the disk 
# converter.save( '../' + CHKP_DIR_FAST)
