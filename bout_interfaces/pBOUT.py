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
tollLES     = 5.0e-4
FILTER_SIG  = 2
step        = 0
lr_kMat_It  = 100
lr_kDNS_It  = 100
lr          = lr_DNS
INIT_SCAL   = 10.0
PATH_StylES = "../../../../StylES/"
CHKP_DIR_WL = PATH_StylES + "utilities/checkpoints_wl"



# clean up and prepare folders
os.system("rm -rf results_bout")

os.system("mkdir -p results_bout/plots")
os.system("mkdir -p results_bout/fields")

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
checkpoint_wl = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
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


# loading wl_synthesis checkpoint and zlatents
if managerCheckpoint_wl.latest_checkpoint:
    print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing wl_synthesis from scratch.")

data      = np.load(PATH_StylES + "/bout_interfaces/restart_fromGAN/z0.npz")
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

# set average values
U_LESm = 0.0
V_LESm = 0.0
P_LESm = 0.0





#---------------------------------------------------------------------- initialize the flow taking the LES field from a GAN inference
def initFlow(npv):

    global U_LESm, V_LESm, P_LESm
    
    # pass delx and dely
    delx_LES = npv[0]
    dely_LES = npv[1]

    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
    
    print("delx, delx_LES ", delx, delx_LES)
    

    # load fields from file or from inference
    if (LOAD_DATA):

        data = np.load('../../../../StylES/bout_interfaces/restart_fromGAN/restart_UVPLES.npz')
        U_LES = data['U']
        V_LES = data['V']
        P_LES = data['P']

        # convert to TensorFlow
        U_LES = tf.convert_to_tensor(U_LES)
        V_LES = tf.convert_to_tensor(V_LES)
        P_LES = tf.convert_to_tensor(P_LES)
        
    else:

        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_latents_LES_restart_A(wl_synthesis, filter, w0, w1)
        print("Starting residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fill {3:3e} " \
            .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

        # find fields
        U_DNS = UVP_DNS[0, 0, :, :]
        V_DNS = UVP_DNS[0, 1, :, :]
        P_DNS = UVP_DNS[0, 2, :, :]

        U_DNS = U_DNS*INIT_SCAL
        V_DNS = V_DNS*INIT_SCAL
        P_DNS = P_DNS*INIT_SCAL

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

    fpPhi_LES = (Jpp + Jpx + Jxp) * spacingFactor

    return fpPhi_LES




@tf.function
def find_bracket_diff(F, G, spacingFactor, pPhi_LES):

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


    # find diffusion terms
    pD = fpPhi_DNS - pPhi_LES
    
    return fpPhi_DNS, pD








#---------------------------------------------------------------------- find missing LES sub-grid scale terms 
def findLESTerms(pLES):

    tstart2 = time.time()

    global wto, w0, w1, lr, U_LESm, V_LESm, P_LESm, mLESo, pPrint


    # pass values from BOUT++
    pLES = pLES.astype("float32")
    
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
    BOUT_Q_LES = pLES[4+3*N_LES*N_LES:4+4*N_LES*N_LES]
    BOUT_R_LES = pLES[4+4*N_LES*N_LES:4+5*N_LES*N_LES]        

    U_LES        = np.reshape(BOUT_U_LES, (N_LES, N_LES))
    V_LES        = np.reshape(BOUT_V_LES, (N_LES, N_LES))
    P_LES        = np.reshape(BOUT_P_LES, (N_LES, N_LES))
    pPHiVort_LES = np.reshape(BOUT_Q_LES, (N_LES, N_LES))        
    pPhiN_LES    = np.reshape(BOUT_R_LES, (N_LES, N_LES))


    # save values from BOUT++
    U_LES_fromBOUT = tf.convert_to_tensor(U_LES)
    V_LES_fromBOUT = tf.convert_to_tensor(V_LES)
    P_LES_fromBOUT = tf.convert_to_tensor(P_LES)

    U_LES_fromBOUT = U_LES_fromBOUT[tf.newaxis,tf.newaxis,:,:]
    V_LES_fromBOUT = V_LES_fromBOUT[tf.newaxis,tf.newaxis,:,:]
    P_LES_fromBOUT = P_LES_fromBOUT[tf.newaxis,tf.newaxis,:,:]

    fimgA_fromBOUT = tf.concat([U_LES_fromBOUT, V_LES_fromBOUT, P_LES_fromBOUT], 1)
    

    # normalize values for StyleGAN
    # # print("Average values are: ", U_LESm, V_LESm, P_LESm)
    # U_LES = U_LES + U_LESm
    # V_LES = V_LES + V_LESm
    # P_LES = P_LES + P_LESm

    # U_LES = U_LES/INIT_SCAL
    # V_LES = V_LES/INIT_SCAL
    # P_LES = P_LES/INIT_SCAL


    # # find min/max
    # U_min = np.min(U_LES)
    # U_max = np.max(U_LES)
    # V_min = np.min(V_LES)
    # V_max = np.max(V_LES)
    # P_min = np.min(P_LES)
    # P_max = np.max(P_LES)

    # # print("Min/max values are: ", U_min, U_max, V_min, V_max, P_min, P_max)
    
    # UVP_minmax = np.asarray([U_min, U_max, V_min, V_max, P_min, P_max])
    # UVP_minmax = tf.convert_to_tensor(UVP_minmax, dtype=DTYPE)



    # preprare target image
    U_LES = tf.convert_to_tensor(U_LES)
    V_LES = tf.convert_to_tensor(V_LES)
    P_LES = tf.convert_to_tensor(P_LES)

    pPHiVort_LES = tf.convert_to_tensor(pPHiVort_LES)
    pPhiN_LES    = tf.convert_to_tensor(pPhiN_LES)        

    # concatenate
    U_LES = U_LES[tf.newaxis,tf.newaxis,:,:]
    V_LES = V_LES[tf.newaxis,tf.newaxis,:,:]
    P_LES = P_LES[tf.newaxis,tf.newaxis,:,:]

    fimgA = tf.concat([U_LES, V_LES, P_LES], 1)
    # fimgA = normalize(fimgA)

    # print("preprare    ", time.time() - tstart2)


    # save old w
    wto = tf.identity(w0)    

   
    # find reconstructed field
    it = 0
    resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
        step_find_residuals(wl_synthesis, filter, w0, w1, fimgA, INIT_SCAL)
    tstart = time.time()
    # print("Starting residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fill {3:3e} " \
    #     .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

    while (resREC.numpy()>tollLES and it<lr_LES_maxIt): 

        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
            step_find_latents_LES(wl_synthesis, filter, opt_LES, w0, w1, fimgA, ltv_LES, INIT_SCAL)
        
        lr = lr_schedule_LES(it)
        mLES = layer_LES.trainable_variables[0]
        if (tf.reduce_min(mLES)<0 or tf.reduce_max(mLES)>1):
            # print("Find new w1...")
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

        if ((it+1)%100==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   step {1:6d}   it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
                .format(tend-tstart, pStep, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

        it = it+1


    #------------------------------------- find new terms if adjustment occurred
    if (pStep==pStepStart and it==0):

        pDvort = np.zeros((N_LES, N_LES), dtype="float64")
        pDn    = np.zeros((N_LES, N_LES), dtype="float64")

        pDvort = tf.reshape(pDvort, [-1])
        pDn    = tf.reshape(pDn,    [-1])

        pPrint = 0
        
    else:

        if (it>0):

            # # print final residuals
            # tend = time.time()
            # print("LES iterations:  time {0:3e}   step {1:6d}   it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
            #     .format(tend-tstart, pStep, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))
        
            # save old w
            mLES = layer_LES.trainable_variables[0]
            wa = mLES*w0[:,0:M_LAYERS,:] + (1.0-mLES)*w1[:,0:M_LAYERS,:]
            wb = wa[:,M_LAYERS-1:M_LAYERS,:]
            wb = tf.tile(wb, [1,G_LAYERS-M_LAYERS,1])
            wa = wa[:,0:M_LAYERS,:]
            wto = tf.concat([wa,wb], axis=1)


        # #------------------------------------- find DNS terms
        # # scaling
        # # UVP_DNS = rescale(UVP_DNS, UVP_minmax)

        # # make sure average is zero
        # U_DNS = UVP_DNS[0,0,:,:]
        # V_DNS = UVP_DNS[0,1,:,:]
        # P_DNS = UVP_DNS[0,2,:,:]

        # tU_DNSm = tf.reduce_mean(U_DNS)
        # tV_DNSm = tf.reduce_mean(V_DNS)
        # tP_DNSm = tf.reduce_mean(P_DNS)
        
        # U_DNS = U_DNS - tU_DNSm
        # V_DNS = V_DNS - tV_DNSm
        # P_DNS = P_DNS - tP_DNSm
        
        # U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
        # V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
        # P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]

        # UVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], 1)


        # #------------------------------------- find LES terms
        # # scaling
        # # UVP_LES = rescale(UVP_LES, UVP_minmax)
            
        # # make sure average is zero
        # U_LES = UVP_LES[0,0,:,:]
        # V_LES = UVP_LES[0,1,:,:]
        # P_LES = UVP_LES[0,2,:,:]

        # tU_LESm = tf.reduce_mean(U_LES)
        # tV_LESm = tf.reduce_mean(V_LES)
        # tP_LESm = tf.reduce_mean(P_LES)
        
        # U_LES = U_LES - tU_LESm
        # V_LES = V_LES - tV_LESm
        # P_LES = P_LES - tP_LESm
        
        # U_LES = U_LES[tf.newaxis, tf.newaxis, :, :]
        # V_LES = V_LES[tf.newaxis, tf.newaxis, :, :]
        # P_LES = P_LES[tf.newaxis, tf.newaxis, :, :]
        
        # UVP_LES = tf.concat([U_LES, V_LES, P_LES], axis=1)

        # U_LESm = tU_LESm.numpy()
        # V_LESm = tV_LESm.numpy()
        # P_LESm = tP_LESm.numpy()


        # #------------------------------------- find fDNS terms
        # # scaling
        # # fUVP_DNS = rescale(fUVP_DNS, UVP_minmax)        

        # # make sure average is zero
        # U_DNS = fUVP_DNS[0,0,:,:]
        # V_DNS = fUVP_DNS[0,1,:,:]
        # P_DNS = fUVP_DNS[0,2,:,:]

        # tU_DNSm = tf.reduce_mean(U_DNS)
        # tV_DNSm = tf.reduce_mean(V_DNS)
        # tP_DNSm = tf.reduce_mean(P_DNS)
        
        # U_DNS = U_DNS - tU_DNSm
        # V_DNS = V_DNS - tV_DNSm
        # P_DNS = P_DNS - tP_DNSm
        
        # U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
        # V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
        # P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]

        # fUVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], 1)
    


        #------------------------------------- find derivatives
        # # find pPHiVort_LES and pPhiN_LES
        # spacingFactor = tf.constant(1.0/(12.0*delx_LES*dely_LES), dtype="float32")
        # F = UVP_LES[0, 1, :, :]
        # G = UVP_LES[0, 2, :, :]
        # pPHiVort_LES2 = find_bracket(F, G, spacingFactor)
        # G = UVP_LES[0, 0, :, :]
        # pPhiN_LES2 = find_bracket(F, G, spacingFactor)


        # find pDvort and pDn
        spacingFactor = tf.constant(1.0/(12.0*delx*dely), dtype="float32")        
        F = UVP_DNS[0, 1, :, :]
        G = UVP_DNS[0, 2, :, :]
        fpPhiVort_DNS, pDvort = find_bracket_diff(F, G, spacingFactor, pPHiVort_LES)
        G = UVP_DNS[0, 0, :, :]
        fpPhiN_DNS, pDn = find_bracket_diff(F, G, spacingFactor, pPhiN_LES)

        # print("derivatives ", time.time() - tstart2)


        #------------------------------------- print values
        # print Poisson brackets
        if (pStep==pStepStart):
            
            pPrint=0
            
            filename = "./results_bout/plots/Plots_fpPhiVort_" + str(pStep).zfill(7)
            nc.savez(filename, U=fpPhiVort_DNS.numpy(), V=pPHiVort_LES.numpy(), P=(fpPhiVort_DNS-pPHiVort_LES).numpy())

            filename = "./results_bout/plots/Plots_fpPhiN_" + str(pStep).zfill(7)
            nc.savez(filename, U=fpPhiN_DNS.numpy(), V=pPhiN_LES.numpy(), P=(fpPhiN_DNS-pPhiN_LES).numpy())

            # filename = "./results_bout/plots/Plots_fpPhiVort_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(fpPhiVort_DNS.numpy(), pPHiVort_LES.numpy(), (fpPhiVort_DNS-pPHiVort_LES).numpy(), N=N_LES, filename=filename)
            # print_fields_3(fpPhiVort_DNS.numpy(), pPHiVort_LES.numpy(), (fpPhiVort_DNS-pPHiVort_LES).numpy(), N=N_LES, filename=filename, \
            #                Umin=-0.2, Umax=0.2, Vmin=-0.2, Vmax=0.2, Pmin=-0.2, Pmax=0.2)
            
            # filename = "./results_bout/plots/Plots_fpPhiN_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(fpPhiN_DNS.numpy(), pPhiN_LES.numpy(), (fpPhiN_DNS-pPhiN_LES).numpy(), N=N_LES, filename=filename)
            # print_fields_3(fpPhiN_DNS.numpy(), pPhiN_LES.numpy(), (fpPhiN_DNS-pPhiN_LES).numpy(), N=N_LES, filename=filename, \
            #                Umin=-0.005, Umax=0.005, Vmin=-0.005, Vmax=0.005, Pmin=-0.005, Pmax=0.005)


        # print values
        if (pStep>=pPrint):

            pPrint = pPrint+100

            filename = "./results_bout/fields/fields_DNS_" + str(pStep).zfill(7)
            nc.savez(filename, simtime=simtime, U=UVP_DNS[0,0,:,:].numpy(), V=UVP_DNS[0,1,:,:].numpy(), P=UVP_DNS[0,2,:,:].numpy())

            # filename = "./results_bout/fields/fields_LES_" + str(pStep).zfill(7)
            # nc.savez(filename, U=UVP_LES[0,0,:,:].numpy(), V=UVP_LES[0,1,:,:].numpy(), P=UVP_LES[0,2,:,:].numpy())

            # filename = "./results_bout/fields/fields_diffLES_" + str(pStep).zfill(7)
            # diff =(UVP_LES-fUVP_DNS)
            # nc.savez(filename, U=diff[0,0,:,:].numpy(), V=diff[0,1,:,:].numpy(), P=diff[0,2,:,:].numpy())

            # filename = "./results_bout/fields/fields_diffLESBOUT_" + str(pStep).zfill(7)
            # diff =(UVP_LES-fimgA)
            # nc.savez(filename, U=diff[0,0,:,:].numpy(), V=diff[0,1,:,:].numpy(), P=diff[0,2,:,:].numpy())

            # filename = "./results_bout/fields/fields_fromBOUT_" + str(pStep).zfill(7)
            # nc.savez(filename, U=fimgA_fromBOUT[0,0,:,:].numpy(), V=fimgA_fromBOUT[0,1,:,:].numpy(), P=fimgA_fromBOUT[0,2,:,:].numpy())

            # filename = "./results_bout/plots/Plots_DNS_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(UVP_DNS[0,0,:,:].numpy(), UVP_DNS[0,1,:,:].numpy(), UVP_DNS[0,2,:,:].numpy(), N=N_DNS, filename=filename)

            # filename = "./results_bout/plots/Plots_LES_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(UVP_LES[0,0,:,:].numpy(), UVP_LES[0,1,:,:].numpy(), UVP_LES[0,2,:,:].numpy(), N=N_LES, filename=filename)

            # filename = "./results_bout/plots/Plots_diffLES_" + str(pStep).zfill(7) + ".png"
            # diff =(UVP_LES-fUVP_DNS)
            # print_fields_3(diff[0,0,:,:].numpy(), diff[0,1,:,:].numpy(), diff[0,2,:,:].numpy(), N=N_LES, filename=filename)

            # filename = "./results_bout/plots/Plots_diffLESBOUT_" + str(pStep).zfill(7) + ".png"
            # diff =(UVP_LES-fimgA)
            # print_fields_3(diff[0,0,:,:].numpy(), diff[0,1,:,:].numpy(), diff[0,2,:,:].numpy(), N=N_LES, filename=filename)

            # filename = "./results_bout/plots/Plots_fromBOUT_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(fimgA[0,0,:,:].numpy(), fimgA[0,1,:,:].numpy(), fimgA[0,2,:,:].numpy(), N=N_LES, filename=filename)


        #------------------------------------- pass back diffusion terms
        pDvort = tf.reshape(pDvort, [-1])
        pDn    = tf.reshape(pDn,    [-1])
        
        pDvort = tf.cast(pDvort, dtype="float64")
        pDn    = tf.cast(pDn, dtype="float64")

        pDvort = pDvort.numpy()
        pDn    = pDn.numpy()

        # print(np.min(pDvort), np.max(pDvort))
        # print(np.min(pDn),    np.max(pDn))

    rLES = np.concatenate((pDvort, pDn), axis=0)

    # print("concatenate ", time.time() - tstart2)
    
    return rLES
    


# test
if (RUN_TEST):
    dummy=np.array([0.2, 0.2])
    initFlow(dummy)




#--------------------------------Extra pieces-----------------------------


# #------------------------------------------ match LES fields
# def matchLES(pLES):

#     global z1, w2, lr

#     # pass values from BOUT++
#     pLES = pLES.astype("float32")
    
#     pStep      = int(pLES[0])
#     pStepMatch = int(pLES[1])
#     residuals  = pLES[2]
    
#     BOUT_U_LES = pLES[3+0*N_LES*N_LES:3+1*N_LES*N_LES]
#     BOUT_V_LES = pLES[3+1*N_LES*N_LES:3+2*N_LES*N_LES]
#     BOUT_P_LES = pLES[3+2*N_LES*N_LES:3+3*N_LES*N_LES]

#     U_LES = np.reshape(BOUT_U_LES, (N_LES, N_LES))
#     V_LES = np.reshape(BOUT_V_LES, (N_LES, N_LES))
#     P_LES = np.reshape(BOUT_P_LES, (N_LES, N_LES))


#     # find min/max which are the same for DNS and LES
#     U_min = np.min(U_LES)
#     U_max = np.max(U_LES)
#     V_min = np.min(V_LES)
#     V_max = np.max(V_LES)
#     P_min = np.min(P_LES)
#     P_max = np.max(P_LES)
    
#     # print("Min/Max n",    U_min, U_max)
#     # print("Min/Max phi",  V_min, V_max)
#     # print("Min/Max vort", P_min, P_max)
#     # print("Total vort",   np.sum(P_LES))

#     UVP_minmax = np.asarray([U_min, U_max, V_min, V_max, P_min, P_max])
#     UVP_minmax = tf.convert_to_tensor(UVP_minmax)

#     # normalize
#     U_LES = 2.0*(U_LES - U_min)/(U_max - U_min) - 1.0
#     V_LES = 2.0*(V_LES - V_min)/(V_max - V_min) - 1.0
#     P_LES = 2.0*(P_LES - P_min)/(P_max - P_min) - 1.0

#     # preprare target image
#     U_LES = tf.convert_to_tensor(U_LES)
#     V_LES = tf.convert_to_tensor(V_LES)
#     P_LES = tf.convert_to_tensor(P_LES)

#     # concatenate
#     U_LES = U_LES[tf.newaxis,tf.newaxis,:,:]
#     V_LES = V_LES[tf.newaxis,tf.newaxis,:,:]
#     P_LES = P_LES[tf.newaxis,tf.newaxis,:,:]

#     fimgA = tf.concat([U_LES, V_LES, P_LES], 1)

#     # find reconstructed field
#     it = 0
#     resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_residuals(wl_synthesis, filter, z0, w0, fimgA)

#     # if (pStep==pStepStart):
#     #     resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_residuals(wl_synthesis, filter, z0, w0, fimgA)
#     # else:
#     #     resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_residuals(wl_synthesis, filter, z1, w2, fimgA)
    
#     tstart = time.time()
#     while (resREC.numpy()>tollLES and it<lr_LES_maxIt): 

#         # if (pStep==pStepStart):

#         #     opt_k.initial_learning_rate = lr_DNS      # reload initial learning rate
#         #     opt_m.initial_learning_rate = lr_LES      # reload initial learning rate

#         #     # iterate on zlatent 1
#         #     if (it<lr_kDNS_It):
#         #         resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS \
#         #             = step_find_latents_kLES(wl_synthesis, filter, opt_k, z0, w0, fimgA, ltv_DNS)

#         #     if (it==lr_kDNS_It):
#         #         # find z1,w1
#         #         k_new = layer_k.trainable_variables[0]
#         #         z1 = k_new*z0
#         #         layer_k.trainable_variables[0].assign(tf.fill((LATENT_SIZE), 1.0))
#         #         w1 = mapping(z1, training=False)
#         #         z2 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
#         #         w2 = mapping(z2, training=False)

#         #     # iterate on zlatent 2
#         #     if (it>=lr_kDNS_It and it<2*lr_kDNS_It):
#         #         resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS \
#         #             = step_find_latents_kLES(wl_synthesis, filter, opt_k, z2, w2, fimgA, ltv_DNS_noNoise)

#         #     if (it==2*lr_kDNS_It):
#         #         # find z2,w2
#         #         k_new = layer_k.trainable_variables[0]
#         #         z2 = k_new*z2
#         #         layer_k.trainable_variables[0].assign(tf.fill((LATENT_SIZE), 1.0))
#         #         w2 = mapping(z2, training=False)
                
#         #     # find new M
#         #     if (it>=2*lr_kDNS_It):
#         #         resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS \
#         #             = step_find_latents_mLES(wl_synthesis, filter, opt_m, z1, w2, fimgA, ltv_LES)
            
#         #     # make sure z1 and w2 values are set
#         #     if (it<lr_kDNS_It):
#         #         z1 = z0
#         #         w2 = w0
#         #     elif (it<2*lr_kDNS_It):
#         #         z1 = z2
#         #         w2 = w2

#         #     # save lr
#         #     if (it<2*lr_kDNS_It):
#         #         lr = lr_schedule_k(it)
#         #     else:
#         #         lr = lr_schedule_m(it)
    
#         # else:

#         #     opt_k.initial_learning_rate = lr      # reload initial learning rate
#         #     opt_m.initial_learning_rate = lr      # reload initial learning rate
    
#         resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS \
#             = step_find_latents_mLES(wl_synthesis, filter, opt_m, z0, w0, fimgA, ltv_LES)

#         if ((it+1)%100==0):
#             tend = time.time()
#             print("LES iterations:  time {0:3e}   step {1:6d}   it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
#                 .format(tend-tstart, pStep, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

#         it = it+1


#     # rescale according to LES min/max all fields
#     UVP_LES  = rescale(UVP_LES,  UVP_minmax)    


#     # print values
#     if (pStep%100==0):

#         UVP_DNS  = rescale(UVP_DNS,  UVP_minmax)
#         fUVP_DNS = rescale(fUVP_DNS, UVP_minmax)
#         fimgA    = rescale(fimgA, UVP_minmax)

#         # print
#         filename = "./results_bout/plots/Plots_DNS_" + str(pStep).zfill(7)
#         nc.savez(filename, U=UVP_DNS[0,0,:,:].numpy(), V=UVP_DNS[0,1,:,:].numpy(), P=UVP_DNS[0,2,:,:].numpy())

#         filename = "./results_bout/plots/Plots_LES_" + str(pStep).zfill(7)
#         nc.savez(filename, U=UVP_LES[0,0,:,:].numpy(), V=UVP_LES[0,1,:,:].numpy(), P=UVP_LES[0,2,:,:].numpy())

#         filename = "./results_bout/plots/Plots_diffLES_" + str(pStep).zfill(7)
#         diff =(UVP_LES-fUVP_DNS)
#         nc.savez(filename, U=diff[0,0,:,:].numpy(), V=diff[0,1,:,:].numpy(), P=diff[0,2,:,:].numpy())

#         filename = "./results_bout/plots/Plots_diffLESBOUT_" + str(pStep).zfill(7)
#         diff =(UVP_LES-fimgA)
#         nc.savez(filename, U=diff[0,0,:,:].numpy(), V=diff[0,1,:,:].numpy(), P=diff[0,2,:,:].numpy())

#         filename = "./results_bout/plots/Plots_fromBOUT_" + str(pStep).zfill(7)
#         nc.savez(filename, U=fimgA[0,0,:,:].numpy(), V=fimgA[0,1,:,:].numpy(), P=fimgA[0,2,:,:].numpy())
        

#         # filename = "./results_bout/plots/Plots_DNS_" + str(pStep).zfill(7) + ".png"
#         # print_fields_3(UVP_DNS[0,0,:,:].numpy(), UVP_DNS[0,1,:,:].numpy(), UVP_DNS[0,2,:,:].numpy(), N=N_DNS, filename=filename)

#         # filename = "./results_bout/plots/Plots_LES_" + str(pStep).zfill(7) + ".png"
#         # print_fields_3(UVP_LES[0,0,:,:].numpy(), UVP_LES[0,1,:,:].numpy(), UVP_LES[0,2,:,:].numpy(), N=N_LES, filename=filename)

#         # filename = "./results_bout/plots/Plots_diffLES_" + str(pStep).zfill(7) + ".png"
#         # diff =(UVP_LES-fUVP_DNS)
#         # print_fields_3(diff[0,0,:,:].numpy(), diff[0,1,:,:].numpy(), diff[0,2,:,:].numpy(), N=N_LES, filename=filename)

#         # filename = "./results_bout/plots/Plots_diffLESBOUT_" + str(pStep).zfill(7) + ".png"
#         # diff =(UVP_LES-fimgA)
#         # print_fields_3(diff[0,0,:,:].numpy(), diff[0,1,:,:].numpy(), diff[0,2,:,:].numpy(), N=N_LES, filename=filename)

#         # filename = "./results_bout/plots/Plots_fromBOUT_" + str(pStep).zfill(7) + ".png"
#         # print_fields_3(fimgA[0,0,:,:].numpy(), fimgA[0,1,:,:].numpy(), fimgA[0,2,:,:].numpy(), N=N_LES, filename=filename)


#     # pass back new LES terms
#     resid = resREC
#     U_LES = UVP_LES[0,0,:,:]
#     V_LES = UVP_LES[0,1,:,:]
#     P_LES = UVP_LES[0,2,:,:]

#     resid = tf.reshape(resid, [-1])
#     U_LES = tf.reshape(U_LES, [-1])
#     V_LES = tf.reshape(V_LES, [-1])
#     P_LES = tf.reshape(P_LES, [-1])
    
#     resid = tf.cast(resid, dtype="float64")
#     U_LES = tf.cast(U_LES, dtype="float64")
#     V_LES = tf.cast(V_LES, dtype="float64")
#     P_LES = tf.cast(P_LES, dtype="float64")

#     resid = resid.numpy()
#     U_LES = U_LES.numpy()
#     V_LES = V_LES.numpy()
#     P_LES = P_LES.numpy()

#     mLES = np.concatenate((resid, U_LES, V_LES, P_LES), axis=0)
        
#     return mLES
