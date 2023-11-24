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
TUNE_NOISE  = False
RELOAD_FREQ = 10000
N_DNS       = 2**RES_LOG2
N_LES       = 2**(RES_LOG2-FIL)
RS          = int(2**FIL)
delx        = 1.0
dely        = 1.0
delx_LES    = 1.0
dely_LES    = 1.0
tollDNS     = 1.e-2
CHKP_DIR    = PATH_StylES + "checkpoints/"
CHKP_DIR_WL = PATH_StylES + "bout_interfaces/restart_fromGAN/checkpoints_wl/"
LES_pass    = lr_LES_maxIt
pPrintFreq  = 1.0
INIT_SCAL   = 10.0
RUN_DNS     = True



#--------------------------- define optimizer, synthesis and variables

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


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")

# create variable synthesis model
layer_kDNS = layer_zlatent_kDNS()
z_in         = tf.keras.Input(shape=([2*(G_LAYERS-M_LAYERS)+1, LATENT_SIZE]), dtype=DTYPE)
img_in       = tf.keras.Input(shape=([NUM_CHANNELS, 2**(RES_LOG2-FIL), 2**(RES_LOG2-FIL)]), dtype=DTYPE)
w            = layer_kDNS(mapping, z_in)
outputs      = synthesis([w,img_in], training=False)
wl_synthesis = tf.keras.Model(inputs=[z_in, img_in], outputs=[outputs, w])


# create filter model
x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
out     = gaussian_filter(x_in[0,0,:,:], rs=RS, rsca=RS)
gfilter = tf.keras.Model(inputs=x_in, outputs=out)



# define optimizer for DNS search
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
checkpoint_wl = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []

for variable in layer_kDNS.trainable_variables:
    ltv_DNS.append(variable)

print("\n DNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)

time.sleep(3)



#---------------------------  loading wl_synthesis checkpoint and zlatents
if managerCheckpoint_wl.latest_checkpoint:
    print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing wl_synthesis from scratch.")

filename = PATH_StylES + "/bout_interfaces/restart_fromGAN/z0.npz"

data = np.load(filename)

z0        = data["z0"]
kDNS      = data["kDNS"]
noise_DNS = data["noise_DNS"]

# convert to TensorFlow tensors            
z0 = tf.convert_to_tensor(z0, dtype=DTYPE)
for nvars in range(len(kDNS)):
    tkDNS = tf.convert_to_tensor(kDNS[nvars], dtype=DTYPE)
    layer_kDNS.trainable_variables[nvars].assign(tkDNS)

# assign variable noise
if (TUNE_NOISE):
    noise_DNS = tf.convert_to_tensor(noise_DNS, dtype=DTYPE)
    it=0
    for layer in synthesis.layers:
        if "layer_noise_constants" in layer.name:
            layer.trainable_variables[0].assign(noise_DNS[it])
            it=it+1


# load DNS fields
data = np.load(FILE_DNS)
U_DNS = data['U']
V_DNS = data['V']
P_DNS = data['P']

# convert to tensor
U_DNS = tf.convert_to_tensor(U_DNS, dtype=DTYPE)
V_DNS = tf.convert_to_tensor(V_DNS, dtype=DTYPE)
P_DNS = tf.convert_to_tensor(P_DNS, dtype=DTYPE)

# normalize
U_min = np.min(U_DNS)
U_max = np.max(U_DNS)
V_min = np.min(V_DNS)
V_max = np.max(V_DNS)
P_min = np.min(P_DNS)
P_max = np.max(P_DNS)

U_norm = max(np.absolute(U_min), np.absolute(U_max))
V_norm = max(np.absolute(V_min), np.absolute(V_max))
P_norm = max(np.absolute(P_min), np.absolute(P_max))

# print("DNS fields min/max", U_min, U_max, V_min, V_max, P_min, P_max)
print("Normalization values", U_norm, V_norm, P_norm)

U_DNS = U_DNS/U_norm
V_DNS = V_DNS/V_norm
P_DNS = P_DNS/P_norm

U_DNS = U_DNS[tf.newaxis, tf.newaxis, :, :]
V_DNS = V_DNS[tf.newaxis, tf.newaxis, :, :]
P_DNS = P_DNS[tf.newaxis, tf.newaxis, :, :]

imgA = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)

# find LES
fU_DNS = gfilter(U_DNS)
fV_DNS = gfilter(V_DNS)
fP_DNS = gfilter(P_DNS)

fimgA = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)


print("---------------- Done Python initialization -------------")







#--------------------------- initialize the flow taking the LES field from a GAN inference
def initFlow(npv):

    
    # pass delx and dely
    delx_LES = npv[0]
    dely_LES = npv[1]

    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
    
    print("delx, delx_LES, N_DNS, N_LES ", delx, delx_LES, N_DNS, N_LES)

    if (RUN_DNS):

        U_LES = (imgA[0, 0, :, :]*U_norm).numpy()
        V_LES = (imgA[0, 1, :, :]*V_norm).numpy()
        P_LES = (imgA[0, 2, :, :]*P_norm).numpy()

    else:

        UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, fimgA])
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=3)
        print("Starting residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
            .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

        # find fields
        U_DNS = (imgA[0, 0, :, :]*U_norm).numpy()
        V_DNS = (imgA[0, 1, :, :]*V_norm).numpy()
        P_DNS = (imgA[0, 2, :, :]*P_norm).numpy()

        U_DNS = U_DNS[tf.newaxis, tf.newaxis, :, :]
        V_DNS = V_DNS[tf.newaxis, tf.newaxis, :, :]
        P_DNS = P_DNS[tf.newaxis, tf.newaxis, :, :]

        fU_DNS = gfilter(U_DNS, training = False)
        fV_DNS = gfilter(V_DNS, training = False)
        fP_DNS = gfilter(P_DNS, training = False)

        fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)

        U_LES = fUVP_DNS[0, 0, :, :].numpy()
        V_LES = fUVP_DNS[0, 1, :, :].numpy()
        P_LES = fUVP_DNS[0, 2, :, :].numpy()


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

    print("---------------- Done init flow from GAN -------------")


    return rnpv







#---------------------------------------------------------------------- define functions
@tf.function
def find_bracket(F, G, filter, spacingFactor):

    # find pPhiVort_DNS
    Jpp = (tr(F, 0, 1) - tr(F, 0,-1)) * (tr(G, 1, 0) - tr(G,-1, 0)) \
        - (tr(F, 1, 0) - tr(F,-1, 0)) * (tr(G, 0, 1) - tr(G, 0,-1))
    Jpx = (tr(G, 1, 0) * (tr(F, 1, 1) - tr(F, 1,-1)) - tr(G,-1, 0) * (tr(F,-1, 1) - tr(F,-1,-1)) \
         - tr(G, 0, 1) * (tr(F, 1, 1) - tr(F,-1, 1)) + tr(G, 0,-1) * (tr(F, 1,-1) - tr(F,-1,-1)))
    Jxp = (tr(G, 1, 1) * (tr(F, 0, 1) - tr(F, 1, 0)) - tr(G,-1,-1) * (tr(F,-1, 0) - tr(F, 0,-1)) \
         - tr(G,-1, 1) * (tr(F, 0, 1) - tr(F,-1, 0)) + tr(G, 1,-1) * (tr(F, 1, 0) - tr(F, 0,-1)))

    pPhi_DNS = (Jpp + Jpx + Jxp)

    # filter
    fpPhi_DNS = filter(pPhi_DNS[tf.newaxis,tf.newaxis,:,:], training=False)
    fpPhi_DNS = fpPhi_DNS[0,0,:,:]*spacingFactor

    return fpPhi_DNS










#---------------------------------------------------------------------- find missing LES sub-grid scale terms 
def findLESTerms(pLES):

    tstart2 = time.time()

    global pPrint, rLES, z0, simtimeo, pStepo

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

    BOUT_U_LES = pLES[4+0*N_LES*N_LES:4+1*N_LES*N_LES]
    BOUT_V_LES = pLES[4+1*N_LES*N_LES:4+2*N_LES*N_LES]
    BOUT_P_LES = pLES[4+2*N_LES*N_LES:4+3*N_LES*N_LES]
    BOUT_F_LES = pLES[4+3*N_LES*N_LES:4+4*N_LES*N_LES]
    BOUT_G_LES = pLES[4+4*N_LES*N_LES:4+5*N_LES*N_LES]

    U_LES = np.reshape(BOUT_U_LES, (N_LES, N_LES))
    V_LES = np.reshape(BOUT_V_LES, (N_LES, N_LES))
    P_LES = np.reshape(BOUT_P_LES, (N_LES, N_LES))
    F_LES = np.reshape(BOUT_F_LES, (N_LES, N_LES))
    G_LES = np.reshape(BOUT_G_LES, (N_LES, N_LES))        

    # F_min = np.min(F_LES)
    # F_max = np.max(F_LES)
    # G_min = np.min(G_LES)
    # G_max = np.max(G_LES)

    # print("Poisson min/max", F_min, F_max, G_min, G_max)

    # normalize
    U_min = np.min(U_LES)
    U_max = np.max(U_LES)
    V_min = np.min(V_LES)
    V_max = np.max(V_LES)
    P_min = np.min(P_LES)
    P_max = np.max(P_LES)

    U_norm = max(np.absolute(U_min), np.absolute(U_max))
    V_norm = max(np.absolute(V_min), np.absolute(V_max))
    P_norm = max(np.absolute(P_min), np.absolute(P_max))
    
    # print("LES fields min/max", U_min, U_max, V_min, V_max, P_min, P_max)
    # print("Normalization values", U_norm, V_norm, P_norm)


    U_LES = U_LES/U_norm
    V_LES = V_LES/V_norm
    P_LES = P_LES/P_norm

    U_LES = U_LES[tf.newaxis, tf.newaxis, :, :]
    V_LES = V_LES[tf.newaxis, tf.newaxis, :, :]
    P_LES = P_LES[tf.newaxis, tf.newaxis, :, :]

    fimgA = tf.concat([U_LES, V_LES, P_LES], axis=1)

    if (pStep==pStepStart):
        pPrint   = simtime
        maxit    = lr_LES_maxIt
        simtimeo = simtime
        delt     = 0.0
        pStepo   = pStep
    else:
        maxit    = LES_pass
        delt     = (simtime - simtimeo)/(pStep - pStepo)
        simtimeo = simtime
        pStepo   = pStep



    #------------------------------------- find reconstructed field
    it = 0
    itDNS = lr_DNS_maxIt
    lr = lr_schedule_DNS(it)
    tstart = time.time()

    UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, fimgA])
    if (pStep==pStepStart):
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=3)
        print("Starting residuals: step {0:6d} simtime {1:3e} resREC {2:3e} resLES {3:3e} resDNS {4:3e} loss_fil {5:3e} lr {6:3e}" \
            .format(pStep, simtime, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

    # to make sure we use the same restarting point as comparison between different tollerances...
    if (pStep==pStepStart):

        while (resREC>tollDNS and it<maxit):

            UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, _ = \
                step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, [z0, fimgA], imgA, fimgA, ltv_DNS, typeRes=3)
            
            valid_zn = True
            kDNS  = layer_kDNS.trainable_variables[0]
            kDNSc = tf.clip_by_value(kDNS, 0.0, 1.0)
            if (tf.reduce_any((kDNS-kDNSc)>0) or (it%1000==0 and it!=0)):
                print("reset values")
                z0p = tf.random.uniform(shape=[BATCH_SIZE, (G_LAYERS-M_LAYERS), LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
                z0n = z0[:,0:1,:]
                for i in range(G_LAYERS-M_LAYERS):
                    zs = kDNSo[i,:]*z0[:,2*i+1,:] + (1.0-kDNSo[i,:])*z0[:,2*i+2,:]
                    z1s = zs[:,tf.newaxis,:] + z0p[:,i:i+1,:]
                    z2s = zs[:,tf.newaxis,:] - z0p[:,i:i+1,:]
                    z0n = tf.concat([z0n,z1s,z2s], axis=1)
                kDNSn = 0.5*tf.ones_like(kDNS)
                valid_zn = False

            if (not valid_zn):
                z0 = tf.identity(z0n)
                layer_kDNS.trainable_variables[0].assign(kDNSn)
                UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, fimgA])
                resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=3)        

            kDNSo = layer_kDNS.trainable_variables[0]

            # if (it%1==0):
            if (it!=0 and it%1000==0):
                tend = time.time()
                lr = lr_schedule_DNS(it)
                print("LES iterations:  time {0:3e}   step {1:6d}   it {2:6d}  residuals {3:3e} resLES {4:3e} resDNS {5:3e} loss_fil {6:3e} lr {7:3e}" \
                    .format(tend-tstart, pStep, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

            it = it+1

        # print final residuals
        tend = time.time()
        print("Finishing residuals: step {0:6d} it {1:4d} simtime {2:3e} delt {3:3e} resREC {4:3e} resLES {5:3e} resDNS {6:3e} loss_fil {7:3e} lr {8:3e}" \
            .format(pStep, it, simtime, delt, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))
        

    #------------------------------------- find new terms if adjustment occurred

    # find Poisson terms
    spacingFactor = tf.constant(1.0/(12.0*delx*dely), dtype=DTYPE)        
    F = UVP_DNS[0, 1, :, :]
    G = UVP_DNS[0, 2, :, :]
    fpPhiVort_DNS = find_bracket(F, G, gfilter, spacingFactor)
    G = UVP_DNS[0, 0, :, :]
    fpPhiN_DNS = find_bracket(F, G, gfilter, spacingFactor)

    # rescale
    fpPhiVort_DNS = fpPhiVort_DNS*V_norm*P_norm
    fpPhiN_DNS    = fpPhiN_DNS   *V_norm*U_norm

    # F_min = np.min(fpPhiVort_DNS)
    # F_max = np.max(fpPhiVort_DNS)
    # G_min = np.min(fpPhiN_DNS)
    # G_max = np.max(fpPhiN_DNS)
    # print("Poisson fp calculated min/max", F_min, F_max, G_min, G_max, V_norm*P_norm)
    

    # filename = "./results_StylES/fields/fields_p_DNS_" + str(pStep).zfill(7)
    # np.savez(filename, pStep=pStep, simtime=simtime, U=pPhiVort_DNS, V=pPhiN_DNS, P=pPhiN_DNS)

    # filename = "./results_StylES/fields/fields_fp_DNS_" + str(pStep).zfill(7)
    # np.savez(filename, pStep=pStep, simtime=simtime, U=fpPhiVort_DNS, V=fpPhiN_DNS, P=fpPhiN_DNS)

    # F_min = np.min(fpPhiVort_DNS)
    # F_max = np.max(fpPhiVort_DNS)
    # G_min = np.min(fpPhiN_DNS)
    # G_max = np.max(fpPhiN_DNS)
    # print("Poisson fp calculated min/max", F_min, F_max, G_min, G_max)

    # F_min = np.min(pPhiVort_DNS)
    # F_max = np.max(pPhiVort_DNS)
    # G_min = np.min(pPhiN_DNS)
    # G_max = np.max(pPhiN_DNS)
    # print("Poisson p calculated min/max", F_min, F_max, G_min, G_max)
    
            
    # print("derivatives ", time.time() - tstart2)
    
    # pass back diffusion terms
    fpPhiVort_DNS = tf.reshape(fpPhiVort_DNS, [-1])
    fpPhiN_DNS    = tf.reshape(fpPhiN_DNS,    [-1])
    
    fpPhiVort_DNS = tf.cast(fpPhiVort_DNS, dtype="float64")
    fpPhiN_DNS    = tf.cast(fpPhiN_DNS, dtype="float64")

    fpPhiVort_DNS = fpPhiVort_DNS.numpy()
    fpPhiN_DNS    = fpPhiN_DNS.numpy()


    if (it==LES_pass):
        # print("Reset LES fields")

        # find LES fields to pass back
        U_LES = fUVP_DNS[0,0,:,:]
        V_LES = fUVP_DNS[0,1,:,:]
        P_LES = fUVP_DNS[0,2,:,:]

        # rescale
        U_LES = U_LES*U_norm
        V_LES = V_LES*V_norm
        P_LES = P_LES*P_norm

        # reshape, cast and pass to CPU (numpy arrays)
        U_LES = tf.reshape(U_LES, [-1])
        V_LES = tf.reshape(V_LES, [-1])
        P_LES = tf.reshape(P_LES, [-1])
        
        U_LES = tf.cast(U_LES, dtype="float64")
        V_LES = tf.cast(V_LES, dtype="float64")
        P_LES = tf.cast(P_LES, dtype="float64")
        
        U_LES = U_LES.numpy()
        V_LES = V_LES.numpy()
        P_LES = P_LES.numpy()

        # concatenate to pass it back
        LES_it = np.asarray([-1], dtype="float64")
        rLES = np.concatenate((LES_it, fpPhiVort_DNS, fpPhiN_DNS, U_LES, V_LES, P_LES), axis=0)
    else:
        LES_it = np.asarray([it], dtype="float64")
        rLES = np.concatenate((LES_it, fpPhiVort_DNS, fpPhiN_DNS), axis=0)    
        
    # print("concatenate ", time.time() - tstart2)


    #------------------------------------- print values
    if (simtime>=pPrint):
        pPrint = pPrint + pPrintFreq

        # find DNS fields
        U_DNS = UVP_DNS[0,0,:,:].numpy()
        V_DNS = UVP_DNS[0,1,:,:].numpy()
        P_DNS = UVP_DNS[0,2,:,:].numpy()

        # rescale
        U_DNS = U_DNS*U_norm
        V_DNS = V_DNS*V_norm
        P_DNS = P_DNS*P_norm

        # save
        filename = "./results_StylES/fields/fields_DNS_" + str(pStep).zfill(7)
        np.savez(filename, pStep=pStep, simtime=simtime, U=U_DNS, V=V_DNS, P=P_DNS)


    return rLES



    



#---------------------------------------------------------------------- find missing LES sub-grid scale terms 
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



