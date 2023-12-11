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


#------------------------------------------------------ initialize StylES procedure

print("\n\n------------------------------------- Start StylES initialization -------------")

import os
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
import sys

RUN_TEST = False
if (RUN_TEST):
    PATH_StylES = "../"
else:
    PATH_StylES = "../../../../StylES/"

sys.path.insert(0, PATH_StylES + 'LES_Solvers/')
sys.path.insert(0, PATH_StylES + '../TurboGenPY/')


from LES_constants import *
from LES_parameters import *
from LES_plot import *

from parameters       import *
from MSG_StyleGAN_tf2 import *
from functions        import *




#--------------------------- define local parameters
TUNE        = True
TUNE_NOISE  = False
RELOAD_FREQ = 10000
N_DNS       = 2**RES_LOG2
N_LES       = 2**(RES_LOG2-FIL)
N2L         = int(N_LES/2)
RS          = int(2**FIL)
delx        = 1.0
dely        = 1.0
delx_LES    = 1.0
dely_LES    = 1.0
tollLES     = 2.5e-1
CHKP_DIR    = PATH_StylES + "checkpoints/"
CHKP_DIR_WL = PATH_StylES + "bout_interfaces/restart_fromGAN/checkpoints_wl/"
LES_pass    = lr_DNS_maxIt
pPrintFreq  = 0.01
INIT_SCAL   = 10.0
RUN_DNS     = False
RESTART_WL  = True




#--------------------------- define optimizer, wl_synthesis and variables

# clean up and prepare folders
os.system("rm -rf results_StylES")
os.system("mkdir -p results_StylES/fields")
os.system("mkdir -p " + CHKP_DIR_WL)

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(SEED_RESTART)

BOUT_U_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
BOUT_V_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
BOUT_P_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
BOUT_F_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
BOUT_G_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)


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
managerCheckpoint = tf.train.CheckpointManager(checkpoint, CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")


# create variable synthesis model
layer_kDNS = layer_zlatent_kDNS()
z_in         = tf.keras.Input(shape=([2*M_LAYERS+1, LATENT_SIZE]), dtype=DTYPE)
w            = layer_kDNS(mapping, z_in)
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=z_in, outputs=[outputs, w])


# create filter model
if (GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = gaussian_filter(x_in[0,0,:,:], rs=RS, rsca=RS)
    gfilter = tf.keras.Model(inputs=x_in, outputs=out)
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




#--------------------------- restart from defined values
if (RESTART_WL):

    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    filename = PATH_StylES + "/bout_interfaces/restart_fromGAN/z0.npz"

    data = np.load(filename)

    z0        = data["z0"]
    kDNS      = data["kDNS"]
    noise_DNS = data["noise_DNS"]

    print("z0",        z0.shape,        np.min(z0),        np.max(z0))
    print("kDNS",      kDNS.shape,      np.min(kDNS),      np.max(kDNS))

    z0 = tf.convert_to_tensor(z0, dtype=DTYPE)
    for nvars in range(len(kDNS)):
        tkDNS = tf.convert_to_tensor(kDNS[nvars], dtype=DTYPE)
        layer_kDNS.trainable_variables[nvars].assign(tkDNS)

    # assign variable noise
    if (TUNE_NOISE):
        print("noise_DNS", noise_DNS.shape, np.min(noise_DNS), np.max(noise_DNS))
        noise_DNS = tf.convert_to_tensor(noise_DNS, dtype=DTYPE)
        it=0
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                print(layer.trainable_variables)
                layer.trainable_variables[:].assign(noise_DNS[it])
                it=it+1

else:

    # set z
    z0 = tf.random.uniform(shape=[BATCH_SIZE, 2*(G_LAYERS-M_LAYERS)+1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)




#---------------------------  load DNS field
# load numpy array
UVP_max = [INIT_SCA, INIT_SCA, INIT_SCA]
UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, z0, UVP_max)

U_DNS = UVP_DNS[0,0,:,:]
V_DNS = UVP_DNS[0,1,:,:]
P_DNS = UVP_DNS[0,2,:,:]

# save original DNS
U_DNS_org = tf.identity(U_DNS)
V_DNS_org = tf.identity(V_DNS)
P_DNS_org = tf.identity(P_DNS)

U_DNS_org = U_DNS_org[tf.newaxis,tf.newaxis,:,:]
V_DNS_org = V_DNS_org[tf.newaxis,tf.newaxis,:,:]
P_DNS_org = P_DNS_org[tf.newaxis,tf.newaxis,:,:]

# set target DNS image for first iteration
imgA = tf.concat([U_DNS_org, V_DNS_org, P_DNS_org], axis=1)


# find filtered field
fU = gfilter(U_DNS_org)[0,0,:,:]
fV = gfilter(V_DNS_org)[0,0,:,:]
fP = gfilter(P_DNS_org)[0,0,:,:]

fimgA = tf.concat([fU[tf.newaxis,tf.newaxis,:,:], fV[tf.newaxis,tf.newaxis,:,:], fP[tf.newaxis,tf.newaxis,:,:]], axis=1)


# normalize
U_min = np.min(fU)
U_max = np.max(fU)
V_min = np.min(fV)
V_max = np.max(fV)
P_min = np.min(fP)
P_max = np.max(fP)

fU_amax = max(np.absolute(U_min), np.absolute(U_max))
fV_amax = max(np.absolute(V_min), np.absolute(V_max))
fP_amax = max(np.absolute(P_min), np.absolute(P_max))

print("Filtered DNS min/max ", U_min, U_max, V_min, V_max, P_min, P_max)
print("Normalization values of filtered DNS", fU_amax, fV_amax, fP_amax)

nfU = fU/fU_amax
nfV = fV/fV_amax
nfP = fP/fP_amax

nfimgA = tf.concat([nfU[tf.newaxis,tf.newaxis,:,:], nfV[tf.newaxis,tf.newaxis,:,:], nfP[tf.newaxis,tf.newaxis,:,:]], axis=1)


#---------------------------  set scaling coefficients



# find multiplier for DNS field
U = tf.identity(U_DNS_org)
V = tf.identity(V_DNS_org)
P = tf.identity(P_DNS_org)

U_min = np.min(U)
U_max = np.max(U)
V_min = np.min(V)
V_max = np.max(V)
P_min = np.min(P)
P_max = np.max(P)

nU_amax = max(np.absolute(U_min), np.absolute(U_max))
nV_amax = max(np.absolute(V_min), np.absolute(V_max))
nP_amax = max(np.absolute(P_min), np.absolute(P_max))

print("DNS fields min/max", U_min, U_max, V_min, V_max, P_min, P_max)
print("Normalization values of DNS", nU_amax, nV_amax, nP_amax)

nU = U/nU_amax
nV = V/nV_amax
nP = P/nP_amax


# find LES
fnU = gfilter(nU)[0,0,:,:]
fnV = gfilter(nV)[0,0,:,:]
fnP = gfilter(nP)[0,0,:,:]

kUmax = ( fnV[N2L,N2L]*nfU[N2L,N2L])/(fnU[N2L,N2L]*nfV[N2L,N2L])*fU_amax*nV_amax/fV_amax
kVmax = ( fnU[N2L,N2L]*nfV[N2L,N2L])/(fnV[N2L,N2L]*nfU[N2L,N2L])*fV_amax*nU_amax/fU_amax
kPmax = ( fnU[N2L,N2L]*nfP[N2L,N2L])/(fnP[N2L,N2L]*nfU[N2L,N2L])*fP_amax*nU_amax/fU_amax

#verify multipliers
if (abs((kUmax-nU_amax) + (kVmax-nV_amax) + (kPmax-nP_amax))>1.e-4):
    print("Diff on kUmax =", kUmax - nP_amax)
    print("Diff on kUmax =", kVmax - nV_amax)
    print("Diff on kUmax =", kPmax - nP_amax)

    print("Mismatch in the filter properties!!!")
    # exit(0)
else:
    print("Diff on kUmax =", kUmax - nU_amax)
    print("Diff on kUmax =", kVmax - nV_amax)
    print("Diff on kUmax =", kPmax - nP_amax)

    UVP_max = [kUmax, kVmax, kPmax]

    print("old scaling values ", UVP_max[0], UVP_max[1], UVP_max[2])





#------------------------------------------------------ initialize the flow
def initFlow(npv):

    
    # pass delx and dely
    delx_LES = npv[0]
    dely_LES = npv[1]

    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
    
    print("delx, delx_LES, N_DNS, N_LES ", delx, delx_LES, N_DNS, N_LES)

    if (RUN_DNS):

        U_LES = (imgA[0, 0, :, :]).numpy()
        V_LES = (imgA[0, 1, :, :]).numpy()
        P_LES = (imgA[0, 2, :, :]).numpy()

    else:

        UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, z0, UVP_max)
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=1)
        print("Starting residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
            .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

        # find fields
        U_LES = fimgA[0, 0, :, :].numpy()
        V_LES = fimgA[0, 1, :, :].numpy()
        P_LES = fimgA[0, 2, :, :].numpy()


    # pass values back
    U_LES = tf.reshape(U_LES, [-1])
    V_LES = tf.reshape(V_LES, [-1])
    P_LES = tf.reshape(P_LES, [-1])
    
    U_LES = tf.cast(U_LES, dtype="float64")
    V_LES = tf.cast(V_LES, dtype="float64")
    P_LES = tf.cast(P_LES, dtype="float64")

    U_LES = U_LES.numpy()
    V_LES = V_LES.numpy()
    P_LES = P_LES.numpy()
    
    rnpv = np.concatenate((U_LES, V_LES, P_LES), axis=0)

    print("------------------------------------- Done StylES initialization -------------\n\n")


    return rnpv









def findLESTerms(pLES):

    tstart2 = time.time()

    global pPrint, rLES, z0, simtimeo, pStepo
    global UVP_max, tollDNS
    global nfUo, nfVo, nfPo
    global fnUo, fnVo, fnPo
    global nU_amaxo, nV_amaxo, nP_amaxo
    global fU_amaxo, fV_amaxo, fP_amaxo



    #------------------------------------- pass values from BOUT++
    pLES = pLES.astype(DTYPE)
    
    pStep      = int(pLES[0])
    pStepStart = int(pLES[1])
    
    delx_LES = pLES[2]
    dely_LES = delx_LES
    simtime  = pLES[3]

    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
    
    BOUT_fU = pLES[4+0*N_LES*N_LES:4+1*N_LES*N_LES]
    BOUT_fV = pLES[4+1*N_LES*N_LES:4+2*N_LES*N_LES]
    BOUT_fP = pLES[4+2*N_LES*N_LES:4+3*N_LES*N_LES]
    BOUT_F_LES = pLES[4+3*N_LES*N_LES:4+4*N_LES*N_LES]
    BOUT_G_LES = pLES[4+4*N_LES*N_LES:4+5*N_LES*N_LES]

    fU = np.reshape(BOUT_fU, (N_LES, N_LES))
    fV = np.reshape(BOUT_fV, (N_LES, N_LES))
    fP = np.reshape(BOUT_fP, (N_LES, N_LES))
    F_LES = np.reshape(BOUT_F_LES, (N_LES, N_LES))
    G_LES = np.reshape(BOUT_G_LES, (N_LES, N_LES))        



    #------------------------------------- normalize filtered field
    U_min = np.min(fU)
    U_max = np.max(fU)
    V_min = np.min(fV)
    V_max = np.max(fV)
    P_min = np.min(fP)
    P_max = np.max(fP)

    fU_amax = max(np.absolute(U_min), np.absolute(U_max))
    fV_amax = max(np.absolute(V_min), np.absolute(V_max))
    fP_amax = max(np.absolute(P_min), np.absolute(P_max))
    
    nfU = fU/fU_amax
    nfV = fV/fV_amax
    nfP = fP/fP_amax

    nfU = tf.convert_to_tensor(nfU, dtype=DTYPE)
    nfV = tf.convert_to_tensor(nfV, dtype=DTYPE)
    nfP = tf.convert_to_tensor(nfP, dtype=DTYPE)

    fimgA  = tf.concat([ fU[tf.newaxis,tf.newaxis,:,:],    fV[tf.newaxis,tf.newaxis,:,:],  fP[tf.newaxis,tf.newaxis,:,:]], axis=1)
    nfimgA = tf.concat([nfU[tf.newaxis,tf.newaxis, :, :], nfV[tf.newaxis,tf.newaxis,:,:], nfP[tf.newaxis,tf.newaxis,:,:]], axis=1)



    #------------------------------------- find reconstructed field
    UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, z0, UVP_max)
    resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=1)
    # print("Starting residuals: step {0:6d} simtime {1:3e} resREC {2:3e} resLES {3:3e} resDNS {4:3e} loss_fil {5:3e}" \
    #     .format(pStep, simtime, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

    if (pStep==pStepStart):
        tollDNS  = tollLES #resREC.numpy()
        pPrint   = simtime
        maxit    = lr_DNS_maxIt
        simtimeo = simtime
        delt     = 0.0
        pStepo   = pStep
    else:
        #tollDNS  = tollLES
        maxit    = LES_pass
        delt     = (simtime - simtimeo)/(pStep - pStepo)
        simtimeo = simtime
        pStepo   = pStep



    # # find new scaling coefficients
    # fnUVP, nfUVP, fUVP_amax, nUVP_amax  = find_scaling(UVP_DNS[0,0,:,:], UVP_DNS[0,1,:,:], UVP_DNS[0,2,:,:], gfilter)

    # fnU = fnUVP[0]
    # fnV = fnUVP[1]
    # fnP = fnUVP[2]

    # nfU = nfUVP[0]
    # nfV = nfUVP[1]
    # nfP = nfUVP[2]

    # fU_amax = fUVP_amax[0]
    # fV_amax = fUVP_amax[1]
    # fP_amax = fUVP_amax[2]

    # kUmax = (fnUo[N2L,N2L]*nfU[N2L,N2L])/(fnU[N2L,N2L]*nfUo[N2L,N2L])*fU_amax*nU_amaxo/fU_amaxo
    # kVmax = (fnVo[N2L,N2L]*nfV[N2L,N2L])/(fnV[N2L,N2L]*nfVo[N2L,N2L])*fV_amax*nV_amaxo/fV_amaxo
    # kPmax = (fnPo[N2L,N2L]*nfP[N2L,N2L])/(fnP[N2L,N2L]*nfPo[N2L,N2L])*fP_amax*nP_amaxo/fP_amaxo

    # print("new scaling values ", kUmax.numpy(), kVmax.numpy(), kPmax.numpy())

    # # rescale DNS fields
    # U_DNS = UVP_DNS[0,0,:,:]/UVP_max[0]*kUmax
    # V_DNS = UVP_DNS[0,1,:,:]/UVP_max[1]*kVmax
    # P_DNS = UVP_DNS[0,2,:,:]/UVP_max[2]*kPmax
    
    # U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
    # V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
    # P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]
    
    # UVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)

    # # rescale fDNS fields
    # fU_DNS = fUVP_DNS[0,0,:,:]/UVP_max[0]*kUmax
    # fV_DNS = fUVP_DNS[0,1,:,:]/UVP_max[1]*kVmax
    # fP_DNS = fUVP_DNS[0,2,:,:]/UVP_max[2]*kPmax
    
    # fU_DNS = fU_DNS[tf.newaxis,tf.newaxis,:,:]
    # fV_DNS = fV_DNS[tf.newaxis,tf.newaxis,:,:]
    # fP_DNS = fP_DNS[tf.newaxis,tf.newaxis,:,:]
    
    # fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)

    # # overwrite new values
    # UVP_max = [kUmax, kVmax, kPmax]
    

    # iterate to given tollerance
    if (pStep>=pStepStart and TUNE):

        # iterate
        it = 0
        lr = lr_schedule_DNS(it)
        tstart = time.time()
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=1)
        # print("Starting residuals: step {0:6d} simtime {1:3e} resREC {2:3e} resLES {3:3e} resDNS {4:3e} loss_fil {5:3e} lr {6:3e}" \
        #     .format(pStep, simtime, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))
        while (resREC>tollDNS and it<maxit):

            UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, _, preds = \
                step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, z0, imgA, fimgA, ltv_DNS, UVP_max, typeRes=1)
            
            
            # # adjust interpolation factors
            # kDNS = layer_kDNS.trainable_variables[1]
            # kDNS = tf.clip_by_value(kDNS, 0.0, 1.0)
            # layer_kDNS.trainable_variables[1].assign(kDNS)
            
            # valid_zn = True
            # kDNS  = layer_kDNS.trainable_variables[0]
            # kDNSc = tf.clip_by_value(kDNS, 0.0, 1.0)
            # if (tf.reduce_any((kDNS-kDNSc)>0) or (it%1000==0 and it!=0)):
            #     print("reset values")
            #     z0p = tf.random.uniform(shape=[BATCH_SIZE, (G_LAYERS-M_LAYERS), LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
            #     z0n = z0[:,0:1,:]
            #     for i in range(G_LAYERS-M_LAYERS):
            #         zs = kDNSo[i,:]*z0[:,2*i+1,:] + (1.0-kDNSo[i,:])*z0[:,2*i+2,:]
            #         z1s = zs[:,tf.newaxis,:] + z0p[:,i:i+1,:]
            #         z2s = zs[:,tf.newaxis,:] - z0p[:,i:i+1,:]
            #         z0n = tf.concat([z0n,z1s,z2s], axis=1)
            #     kDNSn = 0.5*tf.ones_like(kDNS)
            #     valid_zn = False

            # if (not valid_zn):
            #     z0 = tf.identity(z0n)
            #     layer_kDNS.trainable_variables[0].assign(kDNSn)
            #     UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, z0, UVP_max)
            #     resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=1)        

            # kDNSo = layer_kDNS.trainable_variables[0]


            # # find new scaling coefficients
            # fnUVP, nfUVP, fUVP_amax, nUVP_amax  = find_scaling(UVP_DNS[0,0,:,:], UVP_DNS[0,1,:,:], UVP_DNS[0,2,:,:], gfilter)

            # fnU = fnUVP[0]
            # fnV = fnUVP[1]
            # fnP = fnUVP[2]

            # nfU = nfUVP[0]
            # nfV = nfUVP[1]
            # nfP = nfUVP[2]
            
            # fU_amax = fUVP_amax[0]
            # fV_amax = fUVP_amax[1]
            # fP_amax = fUVP_amax[2]
            
            # kUmax = (fnUo[N2L,N2L]*nfU[N2L,N2L])/(fnU[N2L,N2L]*nfUo[N2L,N2L])*fU_amax*nU_amaxo/fU_amaxo
            # kVmax = (fnVo[N2L,N2L]*nfV[N2L,N2L])/(fnV[N2L,N2L]*nfVo[N2L,N2L])*fV_amax*nV_amaxo/fV_amaxo
            # kPmax = (fnPo[N2L,N2L]*nfP[N2L,N2L])/(fnP[N2L,N2L]*nfPo[N2L,N2L])*fP_amax*nP_amaxo/fP_amaxo

            # # print("new scaling values ", kUmax, kVmax, kPmax)


            # # adjust DNS fields according to new scaling
            # U_DNS = UVP_DNS[0,0,:,:]/UVP_max[0]*kUmax
            # V_DNS = UVP_DNS[0,1,:,:]/UVP_max[1]*kVmax
            # P_DNS = UVP_DNS[0,2,:,:]/UVP_max[2]*kPmax
            
            # U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
            # V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
            # P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]
            
            # UVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)

            # fU_DNS = fUVP_DNS[0,0,:,:]/UVP_max[0]*kUmax
            # fV_DNS = fUVP_DNS[0,1,:,:]/UVP_max[1]*kVmax
            # fP_DNS = fUVP_DNS[0,2,:,:]/UVP_max[2]*kPmax
            
            # fU_DNS = fU_DNS[tf.newaxis,tf.newaxis,:,:]
            # fV_DNS = fV_DNS[tf.newaxis,tf.newaxis,:,:]
            # fP_DNS = fP_DNS[tf.newaxis,tf.newaxis,:,:]
            
            # fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)

            # UVP_max = [kUmax, kVmax, kPmax]


            # print residuals            
            # if (it%1==0):
            if (it!=0 and it%100==0):
                tend = time.time()
                lr = lr_schedule_DNS(it)
                print("LES iterations:  time {0:3e}   step {1:6d}   it {2:6d}  residuals {3:3e} resLES {4:3e} resDNS {5:3e} loss_fil {6:3e} lr {7:3e}" \
                    .format(tend-tstart, pStep, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

            it = it+1

        # print final residuals
        tend = time.time()
        if (it>0):
            print("Finishing residuals: step {0:6d} it {1:4d} simtime {2:3e} delt {3:3e} resREC {4:3e} resLES {5:3e} resDNS {6:3e} loss_fil {7:3e} lr {8:3e}" \
                .format(pStep, it, simtime, delt, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

        

    #------------------------------------- find poisson terms
    spacingFactor = tf.constant(1.0/(12.0*delx*dely), dtype=DTYPE)        
    F = UVP_DNS[0, 1, :, :]
    G = UVP_DNS[0, 2, :, :]
    fpPhiVort_DNS = find_bracket(F, G, gfilter, spacingFactor)
    G = UVP_DNS[0, 0, :, :]
    fpPhiN_DNS = find_bracket(F, G, gfilter, spacingFactor)

    # print("derivatives ", time.time() - tstart2)

    # maxPhiVort = tf.reduce_max(fpPhiVort_DNS)
    # maxPhiN    = tf.reduce_max(fpPhiN_DNS)
    # print(maxPhiVort, maxPhiN)


    
    #------------------------------------- pass back diffusion terms
    fpPhiVort_DNS = tf.reshape(fpPhiVort_DNS, [-1])
    fpPhiN_DNS    = tf.reshape(fpPhiN_DNS,    [-1])
    
    fpPhiVort_DNS = tf.cast(fpPhiVort_DNS, dtype="float64")
    fpPhiN_DNS    = tf.cast(fpPhiN_DNS, dtype="float64")

    fpPhiVort_DNS = fpPhiVort_DNS.numpy()
    fpPhiN_DNS    = fpPhiN_DNS.numpy()

    LES_it = np.asarray([0], dtype="float64")
    rLES = np.concatenate((LES_it, fpPhiVort_DNS, fpPhiN_DNS), axis=0)    
        
    # print("concatenate ", time.time() - tstart2)



    #------------------------------------- save field values
    if (simtime>=pPrint):
        pPrint = pPrint + pPrintFreq

        # find DNS fields
        U_DNS = UVP_DNS[0,0,:,:].numpy()
        V_DNS = UVP_DNS[0,1,:,:].numpy()
        P_DNS = UVP_DNS[0,2,:,:].numpy()

        # rescale
        U_DNS = U_DNS
        V_DNS = V_DNS
        P_DNS = P_DNS

        # save
        filename = "./results_StylES/fields/fields_DNS_" + str(pStep).zfill(7)
        np.savez(filename, pStep=pStep, simtime=simtime, U=U_DNS, V=V_DNS, P=P_DNS)


    return rLES




def writePoissonDNS(pLES):

    global pPrint, rLES, z0

    #------------------------------------- pass values from BOUT++
    pLES = pLES.astype(DTYPE)
    
    pStep      = int(pLES[0])
    pStepStart = int(pLES[1])

    delx_LES = pLES[2]
    dely_LES = delx_LES
    simtime  = pLES[3]
    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
 
    # print("L, delx and delx_LES are: ", L, delx, delx_LES, N_DNS, N_LES)

    BOUT_U_LES = pLES[4+0*N_DNS*N_DNS:4+1*N_DNS*N_DNS]
    BOUT_V_LES = pLES[4+1*N_DNS*N_DNS:4+2*N_DNS*N_DNS]
    BOUT_P_LES = pLES[4+2*N_DNS*N_DNS:4+3*N_DNS*N_DNS]
    BOUT_N_LES = pLES[4+3*N_DNS*N_DNS:4+4*N_DNS*N_DNS]
    BOUT_F_LES = pLES[4+4*N_DNS*N_DNS:4+5*N_DNS*N_DNS]        

    U_LES = np.reshape(BOUT_U_LES, (N_DNS, N_DNS))
    V_LES = np.reshape(BOUT_V_LES, (N_DNS, N_DNS))
    P_LES = np.reshape(BOUT_P_LES, (N_DNS, N_DNS))
    fpPhiVort_DNS = np.reshape(BOUT_N_LES, (N_DNS, N_DNS))
    fpPhiN_DNS    = np.reshape(BOUT_F_LES, (N_DNS, N_DNS))

    filename = "./results_StylES/fields/fields_PoissonDNS_" + str(pStep).zfill(7)
    np.savez(filename, pStep=pStep, simtime=simtime, U=fpPhiVort_DNS, V=fpPhiN_DNS, P=fpPhiN_DNS)

    return fpPhiVort_DNS



# test
if (RUN_TEST):
    dummy=np.array([0.2, 0.2])
    initFlow(dummy)





# #------------------------------------------- Extra pieces....


    # fpPhiVort_DNS = tf.zeros([N_LES,N_LES], dtype=DTYPE)
    # fpPhiN_DNS    = tf.zeros([N_LES,N_LES], dtype=DTYPE)

    # fpPhiVort_DNS = tf.reshape(fpPhiVort_DNS, [-1])
    # fpPhiN_DNS    = tf.reshape(fpPhiN_DNS,    [-1])
    
    # fpPhiVort_DNS = tf.cast(fpPhiVort_DNS, dtype="float64")
    # fpPhiN_DNS    = tf.cast(fpPhiN_DNS, dtype="float64")

    # fpPhiVort_DNS = fpPhiVort_DNS.numpy()
    # fpPhiN_DNS    = fpPhiN_DNS.numpy()

    # LES_it = np.asarray([0], dtype="float64")
    # rLES = np.concatenate((LES_it, fpPhiVort_DNS, fpPhiN_DNS), axis=0)     


    # return rLES
