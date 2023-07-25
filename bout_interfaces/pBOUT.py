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

from LES_constants import *
from LES_parameters import *
from LES_plot import *

from parameters       import *
from MSG_StyleGAN_tf2 import *
from functions        import *


# local parameters
LOAD_DATA   = False
TUNE_NOISE  = False
NITEZ       = 0   # number of attempts to find a closer z. When restart from a GAN field, use NITEZ=0
RELOAD_FREQ = 10000
N_DNS       = 2**RES_LOG2
N_LES       = 2**(RES_LOG2-FIL)
RUN_TEST    = False
delx        = 1.0
dely        = 1.0
delx_LES    = 1.0
dely_LES    = 1.0
tollLES     = 1.0e-2
tollDNS     = 1.0e-4 
FILTER_SIG  = 2
step        = 0
lr_kMat_It  = 100
lr_kDNS_It  = 100
lr          = lr_DNS
INIT_SCAL   = 10.0
PATH_StylES = "../../../../StylES/"
CHKP_DIR_WL = PATH_StylES + "bout_interfaces/restart_fromGAN/checkpoints_wl/"
CHKP_DIR_KL = PATH_StylES + "bout_interfaces/restart_fromGAN/checkpoints_kl/"
LES_pass    = 1001
maxitLES    = 100
maxitDNS    = 100
pPrintFreq  = 1000


# clean up and prepare folders
os.system("rm -rf results_StylES")
os.system("mkdir -p results_StylES/fields")
os.system("mkdir -p " + CHKP_DIR_WL)
os.system("mkdir -p " + CHKP_DIR_KL)

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(SEED_RESTART)

U_min = -INIT_SCAL     # to do: values taken from DNS
U_max =  INIT_SCAL
V_min = -INIT_SCAL
V_max =  INIT_SCAL
P_min = -INIT_SCAL
P_max =  INIT_SCAL

UVP_minmax = np.asarray([U_min, U_max, V_min, V_max, P_min, P_max])
UVP_minmax = tf.convert_to_tensor(UVP_minmax, dtype=DTYPE)
        
BOUT_U_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
BOUT_V_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)
BOUT_P_LES  = np.zeros((N_LES,N_LES), dtype=DTYPE)



# loading StyleGAN checkpoint
managerCheckpoint = tf.train.CheckpointManager(checkpoint,
    '/home/jcastagna/projects/Turbulence_with_Style/PhaseII_FARSCAPE2/codes/StylES/checkpoints/',
    max_to_keep=1)
checkpoint.restore(managerCheckpoint.latest_checkpoint)

if managerCheckpoint.latest_checkpoint:
    print("StyleGAN restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing StyleGAN from scratch.")



# create base synthesis model
layer_kDNS       = layer_wlatent_kDNS()
layer_DNS_coarse = layer_wlatent_LES_coarse()
layer_DNS_medium = layer_wlatent_LES_medium()
layer_DNS_finest = layer_wlatent_LES_finest()

z0           = tf.keras.Input(shape=([NWTOT, LATENT_SIZE]), dtype=DTYPE)
w0           = layer_kDNS(z0, mapping)
wc           = layer_DNS_coarse(w0)
wm           = layer_DNS_medium(w0)
wf           = layer_DNS_finest(w0)
w            = tf.concat([wc, wm, wf], axis=1)
outputs      = synthesis(w, training=False)
kl_synthesis = tf.keras.Model(inputs=z0, outputs=[outputs, w])



# create variable synthesis model
layer_LES_coarse = layer_wlatent_LES_coarse()
layer_LES_medium = layer_wlatent_LES_medium()
layer_LES_finest = layer_wlatent_LES_finest()

w0           = tf.keras.Input(shape=([NWTOT, G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
wc           = layer_LES_coarse(w0)
wm           = layer_LES_medium(w0)
wf           = layer_LES_finest(w0)
w            = tf.concat([wc, wm, wf], axis=1)
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=w0, outputs=outputs)



# define optimizer for z and w search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_kDNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_kDNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_kDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_kDNS)

if (lr_LES_POLICY=="EXPONENTIAL"):
    lr_schedule_LES_coarse  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_LES_coarse,
        decay_steps=lr_LES_STEP,
        decay_rate=lr_LES_RATE,
        staircase=lr_LES_EXP_ST)
elif (lr_LES_POLICY=="PIECEWISE"):
    lr_schedule_LES_coarse = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_LES_BOUNDS, lr_LES_VALUES)
opt_LES_coarse = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_LES_coarse)

if (lr_LES_POLICY=="EXPONENTIAL"):
    lr_schedule_LES_medium  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_LES_medium,
        decay_steps=lr_LES_STEP,
        decay_rate=lr_LES_RATE,
        staircase=lr_LES_EXP_ST)
elif (lr_LES_POLICY=="PIECEWISE"):
    lr_schedule_LES_medium = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_LES_BOUNDS, lr_LES_VALUES)
opt_LES_medium = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_LES_medium)

if (lr_LES_POLICY=="EXPONENTIAL"):
    lr_schedule_LES_finest  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_LES_finest,
        decay_steps=lr_LES_STEP,
        decay_rate=lr_LES_RATE,
        staircase=lr_LES_EXP_ST)
elif (lr_LES_POLICY=="PIECEWISE"):
    lr_schedule_LES_finest = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_LES_BOUNDS, lr_LES_VALUES)
opt_LES_finest = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_LES_finest)


# define checkpoints wl_synthesis and filter
checkpoint_wl        = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
checkpoint_kl        = tf.train.Checkpoint(kl_synthesis=kl_synthesis)

managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)
managerCheckpoint_kl = tf.train.CheckpointManager(checkpoint_kl, CHKP_DIR_KL, max_to_keep=1)


# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_LES = []
    ltv_DNS = []
    
for variable in layer_kDNS.trainable_variables:
    ltv_DNS.append(variable)
    
ltv_LES_coarse = []
for variable in layer_LES_coarse.trainable_variables:
    ltv_LES_coarse.append(variable)

ltv_LES_medium = []
for variable in layer_LES_medium.trainable_variables:
    ltv_LES_medium.append(variable)

ltv_LES_finest = []
for variable in layer_LES_finest.trainable_variables:
    ltv_LES_finest.append(variable)

print("\n variables:")
for variable in ltv_DNS:
    print(variable.name)

for variable in ltv_LES_coarse:
    print(variable.name)

for variable in ltv_LES_medium:
    print(variable.name)

for variable in ltv_LES_finest:
    print(variable.name)

time.sleep(3)


# loading wl_synthesis checkpoint and zlatents
if managerCheckpoint_wl.latest_checkpoint:
    print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing wl_synthesis from scratch.")

if managerCheckpoint_kl.latest_checkpoint:
    print("kl_synthesis restored from {}".format(managerCheckpoint_kl.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing kl_synthesis from scratch.")
        
data = np.load(PATH_StylES + "/bout_interfaces/restart_fromGAN/z0.npz")

z0         = data["z0"]
w0         = data["w0"]
LES_coarse = data["LES_coarse"]
LES_medium = data["LES_medium"]
LES_finest = data["LES_finest"]
noise_DNS  = data["noise_DNS"]

print("z0",         z0.shape,         np.min(z0),         np.max(z0))
print("w0",         w0.shape,         np.min(w0),         np.max(w0))
print("LES_coarse", LES_coarse.shape, np.min(LES_coarse), np.max(LES_coarse))
print("LES_medium", LES_medium.shape, np.min(LES_medium), np.max(LES_medium))
print("LES_finest", LES_finest.shape, np.min(LES_finest), np.max(LES_finest))
if (noise_DNS!=[]):
    print("noise_DNS",  noise_DNS.shape,  np.min(noise_DNS),  np.max(noise_DNS))

# convert to TensorFlow tensors            
z0         = tf.convert_to_tensor(z0,         dtype=DTYPE)
w0         = tf.convert_to_tensor(w0,         dtype=DTYPE)
LES_coarse = tf.convert_to_tensor(LES_coarse, dtype=DTYPE)
LES_medium = tf.convert_to_tensor(LES_medium, dtype=DTYPE)
LES_finest = tf.convert_to_tensor(LES_finest, dtype=DTYPE)
noise_DNS  = tf.convert_to_tensor(noise_DNS,  dtype=DTYPE)

# assign variables
layer_LES_coarse.trainable_variables[0].assign(LES_coarse)
layer_LES_medium.trainable_variables[0].assign(LES_medium)
layer_LES_finest.trainable_variables[0].assign(LES_finest)

it=0
for layer in wl_synthesis.layers:
    if "layer_noise_constants" in layer.name:
        print(layer.trainable_variables)
        layer.trainable_variables[:].assign(noise_DNS[it])
        it=it+1

# assign variables base DNS
layer_DNS_coarse.trainable_variables[0].assign(LES_coarse)
layer_DNS_medium.trainable_variables[0].assign(LES_medium)
layer_DNS_finest.trainable_variables[0].assign(LES_finest)

it=0
for layer in kl_synthesis.layers:
    if "layer_noise_constants" in layer.name:
        print(layer.trainable_variables)
        layer.trainable_variables[:].assign(noise_DNS[it])
        it=it+1

# set average values
U_LESm = 0.0
V_LESm = 0.0
P_LESm = 0.0

print("---------------- Done Python initialization -------------")



#---------------------------------------------------------------------- initialize the flow taking the LES field from a GAN inference
def initFlow(npv):

    global U_LESm, V_LESm, P_LESm
    
    # pass delx and dely
    delx_LES = npv[0]
    dely_LES = npv[1]

    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
    
    print("delx, delx_LES, N_LES, N_DNS ", delx, delx_LES, N_LES, N_DNS)
    

    # load fields from file or from inference
    if (LOAD_DATA):

        # data = np.load('../../../../StylES/bout_interfaces/restart_fromGAN/restart_UVPLES.npz')
        data = np.load('../../../../StylES/bout_interfaces/restart_fromGAN/fields_fDNS_0000300.npz')
        # data = np.load('./results_StylES_10tu_tollm3/fields/fields_DNS_0000300.npz')
        U_LES = data['U']
        V_LES = data['V']
        P_LES = data['P']

        # convert to TensorFlow
        U_LES = tf.convert_to_tensor(U_LES, dtype=DTYPE)
        V_LES = tf.convert_to_tensor(V_LES, dtype=DTYPE)
        P_LES = tf.convert_to_tensor(P_LES, dtype=DTYPE)
        
    else:

        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_latents_LES_restart_A(wl_synthesis, filter, w0, INIT_SCAL)
        print("Starting residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
            .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

        # find fields
        U_DNS = UVP_DNS[0, 0, :, :]
        V_DNS = UVP_DNS[0, 1, :, :]
        P_DNS = UVP_DNS[0, 2, :, :]

        U_DNSm = tf.reduce_mean(U_DNS)
        V_DNSm = tf.reduce_mean(V_DNS)
        P_DNSm = tf.reduce_mean(P_DNS)

        U_DNS = U_DNS - U_DNSm
        V_DNS = V_DNS - V_DNSm
        P_DNS = P_DNS - P_DNSm

        U_DNS = U_DNS[tf.newaxis, tf.newaxis, :, :]
        V_DNS = V_DNS[tf.newaxis, tf.newaxis, :, :]
        P_DNS = P_DNS[tf.newaxis, tf.newaxis, :, :]

        fU_DNS = filter(U_DNS, training = False)
        fV_DNS = filter(V_DNS, training = False)
        fP_DNS = filter(P_DNS, training = False)

        UVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], 1)
        fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)

        U_DNS = UVP_DNS[0, 0, :, :].numpy()
        V_DNS = UVP_DNS[0, 1, :, :].numpy()
        P_DNS = UVP_DNS[0, 2, :, :].numpy()

        U_LES = fUVP_DNS[0, 0, :, :].numpy()
        V_LES = fUVP_DNS[0, 1, :, :].numpy()
        P_LES = fUVP_DNS[0, 2, :, :].numpy()

        fU_DNS = fUVP_DNS[0, 0, :, :].numpy()
        fV_DNS = fUVP_DNS[0, 1, :, :].numpy()
        fP_DNS = fUVP_DNS[0, 2, :, :].numpy()



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
def find_bracket(F, G, spacingFactor):

    # find pPhiVort_DNS
    Jpp = (tr(F, 0, 1) - tr(F, 0,-1)) * (tr(G, 1, 0) - tr(G,-1, 0)) \
        - (tr(F, 1, 0) - tr(F,-1, 0)) * (tr(G, 0, 1) - tr(G, 0,-1))
    Jpx = (tr(G, 1, 0) * (tr(F, 1, 1) - tr(F, 1,-1)) - tr(G,-1, 0) * (tr(F,-1, 1) - tr(F,-1,-1)) \
         - tr(G, 0, 1) * (tr(F, 1, 1) - tr(F,-1, 1)) + tr(G, 0,-1) * (tr(F, 1,-1) - tr(F,-1,-1)))
    Jxp = (tr(G, 1, 1) * (tr(F, 0, 1) - tr(F, 1, 0)) - tr(G,-1,-1) * (tr(F,-1, 0) - tr(F, 0,-1)) \
         - tr(G,-1, 1) * (tr(F, 0, 1) - tr(F,-1, 0)) + tr(G, 1,-1) * (tr(F, 1, 0) - tr(F, 0,-1)))

    pPhi_DNS = (Jpp + Jpx + Jxp) * spacingFactor

    # filter
    fpPhi_DNS = filter(pPhi_DNS[tf.newaxis,tf.newaxis,:,:], training=False)
    fpPhi_DNS = fpPhi_DNS[0,0,:,:]

    return fpPhi_DNS








#---------------------------------------------------------------------- find missing LES sub-grid scale terms 
def findLESTerms(pLES):

    tstart2 = time.time()

    global wto, w0, w1, lr, U_LESm, V_LESm, P_LESm, pPrint
    global rLES, firstNewz, z0, z1, z1o



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
    
    # print("L, delx and delx_LES are: ", L, delx, delx_LES)

    BOUT_U_LES = pLES[4+0*N_LES*N_LES:4+1*N_LES*N_LES]
    BOUT_V_LES = pLES[4+1*N_LES*N_LES:4+2*N_LES*N_LES]
    BOUT_P_LES = pLES[4+2*N_LES*N_LES:4+3*N_LES*N_LES]

    U_LES = np.reshape(BOUT_U_LES, (N_LES, N_LES))
    V_LES = np.reshape(BOUT_V_LES, (N_LES, N_LES))
    P_LES = np.reshape(BOUT_P_LES, (N_LES, N_LES))


    if (pStep==pStepStart):
        pPrint = pStepStart
        maxit = lr_LES_maxIt
        firstNewz = True

        managerCheckpoint_wl.save()
        managerCheckpoint_kl.save()

        # find z0
        z0 = z0.numpy()
        w0 = w0.numpy()

        # find kDNS
        LES_coarse = layer_LES_coarse.trainable_variables[0].numpy()
        LES_medium = layer_LES_medium.trainable_variables[0].numpy()
        LES_finest = layer_LES_finest.trainable_variables[0].numpy()

        # find noise_DNS
        it=0
        noise_DNS=[]
        for layer in wl_synthesis.layers:
            if "layer_noise_constants" in layer.name:
                noise_DNS.append(layer.trainable_variables[:].numpy())

        if (TESTCASE=='HIT_2D'):
            filename = "results_latentSpace/z0.npz"
        elif(TESTCASE=='HW' or TESTCASE=='mHW'):
            os.system("mkdir -p ../bout_interfaces/restart_fromGAN/")
            filename = PATH_StylES + "bout_interfaces/restart_fromGAN/z0_fromBOUT.npz"
        
        np.savez(filename,
                z0=z0, \
                w0=w0, \
                LES_coarse=LES_coarse, \
                LES_medium=LES_medium, \
                LES_finest=LES_finest, \
                noise_DNS=noise_DNS)
        
        # input("Press any key to continue, pStep... " + str(pStep))

    else:

        maxit = LES_pass

    #------------------------------------- preprare target image
    U_LES = tf.convert_to_tensor(U_LES, dtype=DTYPE)
    V_LES = tf.convert_to_tensor(V_LES, dtype=DTYPE)
    P_LES = tf.convert_to_tensor(P_LES, dtype=DTYPE)

    # concatenate
    U_LES = U_LES[tf.newaxis,tf.newaxis,:,:]
    V_LES = V_LES[tf.newaxis,tf.newaxis,:,:]
    P_LES = P_LES[tf.newaxis,tf.newaxis,:,:]

    fimgA = tf.concat([U_LES, V_LES, P_LES], 1)

    # print("preprare    ", time.time() - tstart2)


 


    #------------------------------------- find reconstructed field
    it = 0
    tstart = time.time()
    # resREC = tollLES+1
    resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_residuals(wl_synthesis, filter, w0, fimgA, INIT_SCAL)
    # print("Starting residuals:  step {0:3d}   simtime {1:3e}   resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} " \
    #      .format(pStep, simtime, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))
            
    # # to make sure we use the same restarting point as comparison between difference tollerances...
    # if (pStep==pStepStart):
    #     tollLES = 1.0e-3
    # else:
    #     tollLES = 1.0e-4

    pStepNewz = pStep+1
    while (resREC.numpy()>tollLES and it<maxit):

        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
            step_find_latents_LES_coarse(wl_synthesis, filter, opt_LES_coarse, w0, fimgA, ltv_LES_coarse, INIT_SCAL)

        LESc = layer_LES_coarse.trainable_variables[0]
        LESc = tf.clip_by_value(LESc, 0.0, 1.0)
        layer_LES_coarse.trainable_variables[0].assign(LESc)

        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
            step_find_latents_LES_medium(wl_synthesis, filter, opt_LES_medium, w0, fimgA, ltv_LES_medium, INIT_SCAL)

        LESm = layer_LES_medium.trainable_variables[0]
        LESm = tf.clip_by_value(LESm, 0.0, 1.0)
        layer_LES_medium.trainable_variables[0].assign(LESm)

        # resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
        #     step_find_latents_LES_finest(wl_synthesis, filter, opt_LES_finest, w0, fimgA, ltv_LES_finest, INIT_SCAL)

        # LESf = layer_LES_finest.trainable_variables[0]
        # LESf = tf.clip_by_value(LESf, 0.0, 1.0)
        # layer_LES_finest.trainable_variables[0].assign(LESf)
        

        # change next z0
        if ((it==maxitLES or firstNewz) and (pStepNewz!=pStep)):
            
            pStepNewz = pStep

            # find inference with correct LESm variables
            resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_residuals(wl_synthesis, filter, w0, fimgA, INIT_SCAL)
            
            # loop over target DNS image
            kDNS = tf.fill((NWTOT, LATENT_SIZE), 1.0)
            layer_kDNS.trainable_variables[0].assign(kDNS)

            if (firstNewz):
                z1p = tf.identity(z0)
                z1o = tf.identity(z0)
            else:
                z1p = (z1 - z1o) + z1 # linear extrapolation
                z1o = tf.identity(z1)

                # inject some randomness on the linear extrapolation
                z2p = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
                z2p = z2p[:,tf.newaxis,:]
                for nl in range(1, NWTOT):
                    z  = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
                    z  = z[:,tf.newaxis,:]
                    z2p = tf.concat([z2p,z], axis=1)
                z1p = z1p*0.9 + z2p*0.1

            itDNS = 0
            resDNS, wn, UVP_DNSn = step_find_kDNS_noGradients(kl_synthesis, filter, opt_kDNS, z1p, UVP_DNS, ltv_DNS, INIT_SCAL)
            while (resDNS>tollDNS and itDNS<maxitDNS):
                resDNS, wn, UVP_DNSn = step_find_kDNS(kl_synthesis, filter, opt_kDNS, z1p, UVP_DNS, ltv_DNS, INIT_SCAL)
                z1p = z1p*layer_kDNS.trainable_variables[0]
                z1p = tf.clip_by_value(z1p, -1.0, 1.0)
                itDNS = itDNS+1

            # print final residuals
            # print("Residuals DNS search ", resDNS.numpy())

            # save old value
            z1 = tf.identity(z1p)

            # find latest corrected inference
            resDNS, wn, UVP_DNSn = step_find_kDNS_noGradients(kl_synthesis, filter, opt_kDNS, z1, UVP_DNS, ltv_DNS, INIT_SCAL)
            
            # # save fields from DNS search
            # filename = "./results_StylES/fields/fields_DNS_search_org_" + str(pStep).zfill(7)
            # np.savez(filename, simtime=simtime, U=UVP_DNS[0,0,:,:].numpy(), V=UVP_DNS[0,1,:,:].numpy(), P=UVP_DNS[0,2,:,:].numpy())

            # filename = "./results_StylES/fields/fields_DNS_search_" + str(pStep).zfill(7)
            # np.savez(filename, simtime=simtime, U=UVP_DNSn[0,0,:,:].numpy(), V=UVP_DNSn[0,1,:,:].numpy(), P=UVP_DNSn[0,2,:,:].numpy())

            # replace last latent space
            if (not firstNewz):
                print("Change next z0 at step " + str(pStep))

                LESc = layer_LES_coarse.trainable_variables[0]
                LESm = layer_LES_medium.trainable_variables[0]
                LESf = layer_LES_finest.trainable_variables[0]
                nl = NWTOT-2
                wpc = LESc[nl,:,:]*w0[:,nl,       0:C_LAYERS,:] + (1.0-LESc[nl,:,:])*w0[:,nl+1,       0:C_LAYERS,:]
                wpm = LESm[nl,:,:]*w0[:,nl,C_LAYERS:M_LAYERS,:] + (1.0-LESm[nl,:,:])*w0[:,nl+1,C_LAYERS:M_LAYERS,:]
                wpf = LESf[nl,:,:]*w0[:,nl,M_LAYERS:G_LAYERS,:] + (1.0-LESf[nl,:,:])*w0[:,nl+1,M_LAYERS:G_LAYERS,:]
                wp  = tf.concat([wpc, wpm, wpf], axis=1)
                wp  = wp[:,tf.newaxis,:,:]
                wn  = wn[:,tf.newaxis,:,:]

                if (NWTOT>2):
                    tmc = (LESc[nl-1,:,:]*w0[:,nl-1,       0:C_LAYERS,:] + (1-LESc[nl-1,:,:])*w0[:,nl,       0:C_LAYERS,:])
                    tmm = (LESm[nl-1,:,:]*w0[:,nl-1,C_LAYERS:M_LAYERS,:] + (1-LESm[nl-1,:,:])*w0[:,nl,C_LAYERS:M_LAYERS,:])
                    tmf = (LESf[nl-1,:,:]*w0[:,nl-1,M_LAYERS:G_LAYERS,:] + (1-LESf[nl-1,:,:])*w0[:,nl,M_LAYERS:G_LAYERS,:])

                    w0  = tf.concat([w0[:,0:nl,:,:], wp, wn], axis=1)

                    LESc1 = LESc[0:nl-1, :, :]
                    LESm1 = LESm[0:nl-1, :, :]
                    LESf1 = LESf[0:nl-1, :, :]
                    
                    LESc2 = (tmc - w0[:,nl,       0:C_LAYERS,:])/(w0[:,nl-1,       0:C_LAYERS,:] - w0[:,nl,       0:C_LAYERS,:])
                    LESm2 = (tmm - w0[:,nl,C_LAYERS:M_LAYERS,:])/(w0[:,nl-1,C_LAYERS:M_LAYERS,:] - w0[:,nl,C_LAYERS:M_LAYERS,:])
                    LESf2 = (tmf - w0[:,nl,M_LAYERS:G_LAYERS,:])/(w0[:,nl-1,M_LAYERS:G_LAYERS,:] - w0[:,nl,M_LAYERS:G_LAYERS,:])
                    
                    LESc3 = tf.fill((1, C_LAYERS         , LATENT_SIZE), 1.0)
                    LESm3 = tf.fill((1, M_LAYERS-C_LAYERS, LATENT_SIZE), 1.0)
                    LESf3 = tf.fill((1, G_LAYERS-M_LAYERS, LATENT_SIZE), 1.0)
                    
                    LESc = tf.concat([LESc1, LESc2, LESc3], axis=0)
                    LESm = tf.concat([LESm1, LESm2, LESm3], axis=0)
                    LESf = tf.concat([LESf1, LESf2, LESf3], axis=0)

                else:

                    w0  = tf.concat([wp, wn], axis=1)
                    
                    LESc = tf.fill((1, C_LAYERS         , LATENT_SIZE), 1.0)
                    LESm = tf.fill((1, M_LAYERS-C_LAYERS, LATENT_SIZE), 1.0)
                    LESf = tf.fill((1, G_LAYERS-M_LAYERS, LATENT_SIZE), 1.0)
                
                layer_LES_coarse.trainable_variables[0].assign(LESc)
                layer_LES_medium.trainable_variables[0].assign(LESm)
                layer_LES_finest.trainable_variables[0].assign(LESf)

                # reset counter
                it = 0

            else:

                firstNewz = False

        else:
            
            it = it+1

        # if (it%1==0):
        if (it!=0 and (it+1)%100==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   step {1:6d}   it {2:6d}  residuals {3:3e} resLES {4:3e} resDNS {5:3e} loss_fil {6:3e}" \
                .format(tend-tstart, pStep, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))


    #------------------------------------- find new terms if adjustment occurred
    if (it>0 or pStep==pStepStart):

        # repeat inference to make sure values are clipped
        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_residuals(wl_synthesis, filter, w0, fimgA, INIT_SCAL)
        
        # print final residuals
        # tend = time.time()
        print("Finishing residuals:  step {0:5d}   it {1:3d}   simtime {2:3e}   resREC {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fil {6:3e} " \
            .format(pStep, it, simtime, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))    

        # find derivatives
        spacingFactor = tf.constant(1.0/(12.0*delx*dely), dtype=DTYPE)        
        F = UVP_DNS[0, 1, :, :]
        G = UVP_DNS[0, 2, :, :]
        fpPhiVort_DNS = find_bracket(F, G, spacingFactor)
        G = UVP_DNS[0, 0, :, :]
        fpPhiN_DNS = find_bracket(F, G, spacingFactor)

        # print("derivatives ", time.time() - tstart2)

        
        # pass back diffusion terms
        fpPhiVort_DNS = tf.reshape(fpPhiVort_DNS, [-1])
        fpPhiN_DNS    = tf.reshape(fpPhiN_DNS,    [-1])
        
        fpPhiVort_DNS = tf.cast(fpPhiVort_DNS, dtype="float64")
        fpPhiN_DNS    = tf.cast(fpPhiN_DNS, dtype="float64")

        fpPhiVort_DNS = fpPhiVort_DNS.numpy()
        fpPhiN_DNS    = fpPhiN_DNS.numpy()

                
        if (it>=LES_pass):
            print("Reset LES fields")
            U_LES = fUVP_DNS[0,0,:,:]
            V_LES = fUVP_DNS[0,1,:,:]
            P_LES = fUVP_DNS[0,2,:,:]
                    
            U_LES = tf.reshape(U_LES, [-1])
            V_LES = tf.reshape(V_LES, [-1])
            P_LES = tf.reshape(P_LES, [-1])
            
            U_LES = tf.cast(U_LES, dtype="float64")
            V_LES = tf.cast(V_LES, dtype="float64")
            P_LES = tf.cast(P_LES, dtype="float64")
            
            U_LES = U_LES.numpy()
            V_LES = V_LES.numpy()
            P_LES = P_LES.numpy()

            LES_it = np.asarray([-1], dtype="float64")
            rLES = np.concatenate((LES_it, fpPhiVort_DNS, fpPhiN_DNS, U_LES, V_LES, P_LES), axis=0)
        else:
            LES_it = np.asarray([it], dtype="float64")
            rLES = np.concatenate((LES_it, fpPhiVort_DNS, fpPhiN_DNS), axis=0)    
            

        # print("concatenate ", time.time() - tstart2)


    #------------------------------------- print values
    if (pStep>=pPrint):
        pPrint = pPrint + pPrintFreq
        filename = "./results_StylES/fields/fields_DNS_" + str(pStep).zfill(7)
        np.savez(filename, simtime=simtime, U=UVP_DNS[0,0,:,:].numpy(), V=UVP_DNS[0,1,:,:].numpy(), P=UVP_DNS[0,2,:,:].numpy())

    return rLES
    


# test
if (RUN_TEST):
    dummy=np.array([0.2, 0.2])
    initFlow(dummy)






#---------------------extra pieces

        # lrc = lr_schedule_LES_finest(it)
        # lrm = lr_schedule_LES_finest(it)
        # lrf = lr_schedule_LES_finest(it)

        # # if (it%10==0):
        # if (it!=0 and it%100==0):
        #     tend = time.time()
        #     print("LES iterations:  time {0:3e}   step {1:6d}   it {2:6d}  residuals {3:3e} resLES {4:3e} resDNS {5:3e} loss_fil {6:3e}  lrc {7:3e}  lrm {8:3e}  lrf {9:3e} " \
        #         .format(tend-tstart, pStep, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lrc, lrm, lrf))

        #     # resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_residuals(wl_synthesis, filter, w0, fimgA, INIT_SCAL)
        #     # LESc = layer_LES_coarse.trainable_variables[0].numpy()
        #     # print(np.min(LESc), np.max(LESc))
        #     # LESm = layer_LES_medium.trainable_variables[0].numpy()
        #     # print(np.min(LESm), np.max(LESm))
        #     # LESf = layer_LES_finest.trainable_variables[0].numpy()
        #     # print(np.min(LESf), np.max(LESf))
        #     # filename = "./results_StylES/fields/fields_DNS_it_" + str(it).zfill(7)
        #     # np.savez(filename, simtime=simtime, U=UVP_DNS[0,0,:,:].numpy(), V=UVP_DNS[0,1,:,:].numpy(), P=UVP_DNS[0,2,:,:].numpy())

        # LES_1 = layer_LES.trainable_variables[0]
        # LES_2 = layer_LES.trainable_variables[1]
        # if ((tf.reduce_min(LES_1)<-0.5 or tf.reduce_max(LES_1)>1.5) or \
        #     (tf.reduce_min(LES_2)<-0.5 or tf.reduce_max(LES_2)>1.5)):
        #     # print("Find new w1...", it)
        #     # waa = LESo*w0[:,0:M_LAYERS,:] + (1.0-LESo)*w1[:,0:M_LAYERS,:]
        #     # wbb = w0[:,M_LAYERS-1:M_LAYERS,:]
        #     # wbb = tf.tile(wbb, [1,G_LAYERS-M_LAYERS,1])
        #     # waa = waa[:,0:M_LAYERS,:]
        #     # wt  = tf.concat([waa,wbb], axis=1)
                        
        #     # wa = LESo*w0[:,0:M_LAYERS,:] + (1.0-LESo)*w1[:,0:M_LAYERS,:]
        #     # wb = wa[:,M_LAYERS-1:M_LAYERS,:]
        #     # wb = tf.tile(wb, [1,G_LAYERS-M_LAYERS,1])
        #     # wa = wa[:,0:M_LAYERS,:]
        #     # wt = tf.concat([wa,wb], axis=1)

        #     # wa = LESo_1*w0[:,0:M_LAYERS-1       ,:] + (1.0-LESo_1)*w1[:,0:M_LAYERS-1       ,:]
        #     # me = tf.tile(LESo_2, [G_LAYERS-(M_LAYERS-1),1])
        #     # wb = me*w0[:,M_LAYERS-1:G_LAYERS,:] + (1.0-me)*w1[:,M_LAYERS-1:G_LAYERS,:]
        #     # wt = tf.concat([wa,wb], axis=1)

        #     # for iSubLES in range(10):
        #     #     resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
        #     #         step_find_latents_subLES(wl_synthesis, filter, opt_LES, w2, fimgA, ltv_LES, INIT_SCAL)

        #     z2 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART+2)
        #     w2 = mapping(z2, training=False)

        #     w0 = tf.identity(wt)
        #     w1 = tf.identity(w2)


        #     # save values
        #     managerCheckpoint_wl.save()
        #     dz0 = z0.numpy()
        #     dw0 = w0.numpy()
        #     dw1 = w1.numpy()
        #     dw2 = w2.numpy()
        #     dwt  = wt.numpy()
        #     dwto = wto.numpy()
        #     dLES_1 = LES_1
        #     dLES_2 = LES_2
        #     dLESo_1 = LESo_1
        #     dLESo_2 = LESo_2
            
        #     it=0
        #     dnoise_DNS=[]
        #     for layer in wl_synthesis.layers:
        #         if "layer_noise_constants" in layer.name:
        #             print(layer.trainable_variables)
        #             noise_DNS.append(layer.trainable_variables[0].numpy())
        #     print("\nsave new latent spaces")
        #     np.savez(PATH_StylES + "bout_interfaces/restart_fromGAN/z0_fromBOUT.npz", \
        #         z0=dz0, w0=dw0, w1=dw1, w2=dw2, wt=dwt, wto=dwto, \
        #         LES_1=dLES_1, LES_2=dLES_2, LESo_1=dLESo_1, LESo_2=dLESo_2, noise_DNS=dnoise_DNS)

        #     input("Press any key to continue, pStep... " + str(pStep))

        #     # reset variables
        #     LESn_1 = tf.fill((M_LAYERS-1, LATENT_SIZE), 1.0)
        #     LESn_2 = tf.fill((1,          LATENT_SIZE), 1.0)
        #     LESn_1 = tf.cast(LESn_1, dtype=DTYPE)
        #     LESn_2 = tf.cast(LESn_2, dtype=DTYPE)
        #     layer_LES.trainable_variables[0].assign(LESn_1)
        #     layer_LES.trainable_variables[1].assign(LESn_2)        
                    
        # else:
        #     LESo_1 = tf.identity(LES_1)
        #     LESo_2 = tf.identity(LES_2)


