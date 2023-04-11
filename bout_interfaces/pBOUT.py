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
TUNE_NOISE  = True
NITEZ       = 0   # number of attempts to find a closer z. When restart from a GAN field, use NITEZ=0
RELOAD_FREQ = 10000
N_DNS       = 2**RES_LOG2
N_LES       = 2**(RES_LOG2-FIL)
RUN_TEST    = False
delx        = 1.0
dely        = 1.0
delx_LES    = 1.0
dely_LES    = 1.0
tollLES     = 1.e-2  # to do: if this is too large it may fail as w1 and z2 are not set!
FILTER_SIG  = 2
step        = 0
INI_SCALING = 10.0  # to do: define a better rescaling
lr_kDNS_It  = 100
lr          = lr_DNS

BOUT_U_LES  = np.zeros((N_LES,N_LES), dtype="float32")
BOUT_V_LES  = np.zeros((N_LES,N_LES), dtype="float32")
BOUT_P_LES  = np.zeros((N_LES,N_LES), dtype="float32")



# clean up and prepare folders
os.system("rm -rf results_bout")

os.system("mkdir -p results_bout/plots")
os.system("mkdir -p results_bout/fields")

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(SEED_RESTART)




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
layer_k = layer_klatent()
layer_m = layer_mlatent_LES()

zlatents     = tf.keras.Input(shape=([LATENT_SIZE]),           dtype=DTYPE)
w2           = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
zlatents_DNS = layer_k(zlatents)
w1           = mapping(zlatents_DNS)
w            = layer_m(w1, w2)
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=[zlatents, w2], outputs=outputs)



# define optimizer for kDNS search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_k  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_k = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_k = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_k)

# define optimizer for mDNS search
if (lr_LES_POLICY=="EXPONENTIAL"):
    lr_schedule_m  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_LES,
        decay_steps=lr_LES_STEP,
        decay_rate=lr_LES_RATE,
        staircase=lr_LES_EXP_ST)
elif (lr_LES_POLICY=="PIECEWISE"):
    lr_schedule_m = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_LES_BOUNDS, lr_LES_VALUES)
opt_m = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_m)


# set trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    ltv_LES = []

for variable in layer_k.trainable_variables:
    ltv_DNS.append(variable)
    ltv_LES.append(variable)

for variable in layer_m.trainable_variables:
    ltv_LES.append(variable)
    
print("\n ltv_DNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)

print("\n ltv_LES variables:")
for variable in ltv_LES:
    print(variable.name, variable.shape)

time.sleep(3)


print("\n DNS variables:")
for variable in ltv_DNS:
    print(variable.name)

print("\n LES variables:")
for variable in ltv_LES:
    print(variable.name)


# set z latent space
zlatents = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
wlatents = mapping(zlatents, training=False)









#---------------------------------------------------------------------- initialize the flow taking the LES field from a GAN inference
def initFlow(npv):

    # pass delx and dely
    delx_LES = npv[0]
    dely_LES = npv[1]

    L = (delx_LES + dely_LES)/2.0*N_LES

    delx = delx_LES*N_LES/N_DNS
    dely = dely_LES*N_LES/N_DNS
    
    print("delx, delx_LES ", delx, delx_LES)
    
    

    # predictions from GAN
    predictions = wl_synthesis([zlatents, wlatents], training=False)



    #---------------- find LES fields from GAN
    UVP_LES = predictions[RES_LOG2-FIL-2]*INI_SCALING    # to do: need a better init values

    # split fields
    U_LES = UVP_LES[0,0,:,:]*0.0  # to do: need a better init values
    V_LES = UVP_LES[0,1,:,:]
    P_LES = UVP_LES[0,2,:,:]

    # make sure vorticity is overall null
    P_LES = P_LES - tf.reduce_mean(P_LES)

    # make sure vorticity is overall null and matches laplacian of phi min/max
    # P_LES = (-tr(V_LES, 2, 0) + 16*tr(V_LES, 1, 0) - 30*V_LES + 16*tr(V_LES,-1, 0) - tr(V_LES,-2, 0))/(12*delx_LES**2) \
    #       + (-tr(V_LES, 0, 2) + 16*tr(V_LES, 0, 1) - 30*V_LES + 16*tr(V_LES, 0,-1) - tr(V_LES, 0,-2))/(12*delx_LES**2)
    # cP_LES = sc.ndimage.gaussian_filter(P_LES.numpy(), 1, mode=['wrap','wrap'])
    # # # filtered DNS fields
    # UVP_DNS = predictions[RES_LOG2-2]
    # P_DNS   = UVP_DNS[0,2,:,:]
    # fP_DNS  = (filter(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
    # cP_LES = (cP_LES - np.min(cP_LES))/(np.max(cP_LES) - np.min(cP_LES))  \
    #        * (np.max(fP_DNS.numpy()) - np.min(fP_DNS.numpy())) + np.min(fP_DNS.numpy())
    # P_LES = tf.convert_to_tensor(cP_LES)

    # find grad Phi for energy estimation
    gradV_LES = tf.sqrt(((tr(V_LES, 1, 0) - tr(V_LES, -1, 0))/(2.0*delx))**2 + ((tr(V_LES, 0, 1) - tr(V_LES, 0, -1))/(2.0*dely))**2)

    # pass to Numpy
    U_LES     = U_LES.numpy()
    V_LES     = V_LES.numpy()
    P_LES     = P_LES.numpy()
    gradV_LES = gradV_LES.numpy()

    # find min/max
    U_min = np.min(U_LES)
    U_max = np.max(U_LES)
    V_min = np.min(V_LES)
    V_max = np.max(V_LES)
    P_min = np.min(P_LES)
    P_max = np.max(P_LES)

    # find min/max which are the same for DNS and LES
    UVP_minmax = np.asarray([U_min, U_max, V_min, V_max, P_min, P_max])
    UVP_minmax = tf.convert_to_tensor(UVP_minmax)
    
    
    #---------------- print min/max values and shape
    print("Min/Max n",    U_min, U_max)
    print("Min/Max phi",  V_min, V_max)
    print("Min/Max vort", P_min, P_max)
    print("Tot vort",     np.sum(P_LES))
    
    print(U_LES.shape)



        
    #---------------- find DNS fields from StyleGAN
    UVP_DNS = predictions[RES_LOG2-2]*INI_SCALING  # to do: need a better init values

    # split fields
    U_DNS = UVP_DNS[0,0,:,:]*0.0  # to do: need a better init values
    V_DNS = UVP_DNS[0,1,:,:]
    P_DNS = UVP_DNS[0,2,:,:]
    
    # make sure vorticity is overall null
    P_DNS = P_DNS - tf.math.reduce_mean(P_DNS)

    # rescale to LES min/max
    UVP_DNS = rescale(UVP_DNS, UVP_minmax)
    
    # find grad Phi for energy estimation
    gradV_DNS = tf.sqrt(((tr(V_DNS, 1, 0) - tr(V_DNS, -1, 0))/(2.0*delx))**2 + ((tr(V_DNS, 0, 1) - tr(V_DNS, 0, -1))/(2.0*dely))**2)

    # pass to Numpy
    U_DNS     = U_DNS.numpy()
    V_DNS     = V_DNS.numpy()
    P_DNS     = P_DNS.numpy()
    gradV_DNS = gradV_DNS.numpy()



    #---------------- find also the LES fields filtering DNS 
    fUVP_DNS = predictions[RES_LOG2-2]*INI_SCALING  # to do: need a better init values

    # filter fields
    fU_DNS = fUVP_DNS[:,0,:,:]*0.0  # to do: need a better init values
    fU_DNS = fU_DNS[:,tf.newaxis,:,:]
    fU_DNS = filter(fU_DNS, training = False)

    fV_DNS = fUVP_DNS[:,1,:,:]
    fV_DNS = fV_DNS[:,tf.newaxis,:,:]
    fV_DNS = filter(fV_DNS, training = False)

    fP_DNS = fUVP_DNS[:,2,:,:]
    fP_DNS = fP_DNS[:,tf.newaxis,:,:]
    fP_DNS = filter(fP_DNS, training = False)

    # make sure vorticity is overall null
    fP_DNS = fP_DNS - tf.math.reduce_mean(fP_DNS)

    # concatenate
    fUVP_DNS =  tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)

    # rescale to LES min/max
    fUVP_DNS = rescale(fUVP_DNS, UVP_minmax)

    # separate fields
    fU_DNS = fU_DNS[0,0,:,:]
    fV_DNS = fV_DNS[0,0,:,:]
    fP_DNS = fP_DNS[0,0,:,:]
    
    # find grad Phi for energy estimation
    fgradV_DNS = tf.sqrt(((tr(fV_DNS, 1, 0) - tr(fV_DNS, -1, 0))/(2.0*delx))**2 + ((tr(fV_DNS, 0, 1) - tr(fV_DNS, 0, -1))/(2.0*dely))**2)

    # pass to Numpy
    fU_DNS     = fU_DNS.numpy()
    fV_DNS     = fV_DNS.numpy()
    fP_DNS     = fP_DNS.numpy()
    fgradV_DNS = fgradV_DNS.numpy()



    # ---------------- plot values
    if (RUN_TEST):

        filename = "./results_bout/Plots_DNS_fromGAN.png"
        print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename)
        # print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename, \
        #                Umin=-10.0, Umax=10.0, Vmin=-10.0, Vmax=10, Pmin=-10.0, Pmax=10.0)

        filename = "./results_bout/Plots_LES_fromGAN.png"
        print_fields_3(U_LES, V_LES, P_LES, N_LES, filename)
        # print_fields_3(U_LES, V_LES, P_LES, N_LES, filename, \
        #                Umin=-10.0, Umax=10.0, Vmin=-10.0, Vmax=10, Pmin=-10.0, Pmax=10.0)
        
        filename = "./results_bout/Plots_diff_LES_fDNS.png"
        print_fields_3(U_LES-fU_DNS, V_LES-fV_DNS, P_LES-fP_DNS, N_LES, filename)
        # print_fields_3(U_LES-fU_DNS, V_LES-fV_DNS, P_LES-fP_DNS, N_LES, filename, \
        #                Umin=-10.0, Umax=10.0, Vmin=-10.0, Vmax=10, Pmin=-10.0, Pmax=10.0)


        filename = "./results_bout/EnergySpectrum_DNS.png"
        plot_spectrum(U_DNS, gradV_DNS, L, filename, close=False)
        
        filename = "./results_bout/EnergySpectrum_LES.png"
        plot_spectrum(U_LES, gradV_LES, L, filename, close=False)

        filename = "./results_bout/EnergySpectrum_fDNS.png"
        plot_spectrum(fU_DNS, fgradV_DNS, L, filename, close=True)

    else:        

        filename = "./results_bout/Plots_DNS_fromGAN"
        nc.savez(filename, U=U_DNS, V=V_DNS, P=P_DNS)

        filename = "./results_bout/Plots_LES_fromGAN"
        nc.savez(filename, U=U_LES, V=V_LES, P=P_LES)

        filename = "./results_bout/Plots_diff_LES_fDNS"
        nc.savez(filename, U=U_LES-fU_DNS, V=V_LES-fV_DNS, P=P_LES-fP_DNS)
        
        filename = "./results_bout/EnergySpectrum_DNS"
        nc.savez(filename, U=U_DNS, grad=gradV_DNS)

        filename = "./results_bout/EnergySpectrum_LES"
        nc.savez(filename, U=U_LES, grad=gradV_LES)

        filename = "./results_bout/EnergySpectrum_fDNS"
        nc.savez(filename, U=fU_DNS, grad=fgradV_DNS)


    #---------------- pass values back
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
def step_find_residuals(wl_synthesis, filter, latents, w, fimgA):
        
    # find predictions
    predictions = wl_synthesis([latents, w], training=False)
    UVP_DNS = predictions[RES_LOG2-2]
    UVP_LES = predictions[RES_LOG2-FIL-2]

    # filter DNS field
    fU_DNS = UVP_DNS[:,0,:,:]
    fU_DNS = fU_DNS[:,tf.newaxis,:,:]
    fU_DNS = filter(fU_DNS, training = False)

    fV_DNS = UVP_DNS[:,1,:,:]
    fV_DNS = fV_DNS[:,tf.newaxis,:,:]
    fV_DNS = filter(fV_DNS, training = False)

    fP_DNS = UVP_DNS[:,2,:,:]
    fP_DNS = fP_DNS[:,tf.newaxis,:,:]
    fP_DNS = filter(fP_DNS, training = False)
    
    fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)

    # find residuals
    resDNS = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))
    resLES = tf.math.reduce_mean(tf.math.squared_difference( UVP_LES, fimgA))
    resREC = resDNS + resLES

    # find filter loss
    loss_fil = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))

    return resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS




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

    global z2, w1, lr

    # pass values from BOUT++
    pLES = pLES.astype("float32")
    
    pStep      = int(pLES[0])
    pStepStart = int(pLES[1])
    
    if (pStep==pStepStart):   # to do: fix return in BOUT++
        print("\n")

    delx_LES = pLES[2]
    dely_LES = pLES[3]

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


    # find min/max which are the same for DNS and LES
    U_min = np.min(U_LES)
    U_max = np.max(U_LES)
    V_min = np.min(V_LES)
    V_max = np.max(V_LES)
    P_min = np.min(P_LES)
    P_max = np.max(P_LES)
    
    # print("Min/Max n",    U_min, U_max)
    # print("Min/Max phi",  V_min, V_max)
    # print("Min/Max vort", P_min, P_max)
    # print("Total vort",     np.sum(P_LES))

    UVP_minmax = np.asarray([U_min, U_max, V_min, V_max, P_min, P_max])
    UVP_minmax = tf.convert_to_tensor(UVP_minmax)

    # normalize
    U_LES = 2.0*(U_LES - U_min)/(U_max - U_min) - 1.0
    V_LES = 2.0*(V_LES - V_min)/(V_max - V_min) - 1.0
    P_LES = 2.0*(P_LES - P_min)/(P_max - P_min) - 1.0


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

    # print("preprare    ", time.time() - tstart2)


   
    # find reconstructed field
    it = 0
    if (pStep==pStepStart):
        resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_residuals(wl_synthesis, filter, zlatents, wlatents, fimgA)
    else:
        resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_residuals(wl_synthesis, filter, z2, w1, fimgA)
        
    tstart = time.time()
    while (resREC.numpy()>tollLES and it<lr_LES_maxIt): 

        if (pStep==pStepStart):

            opt_k.initial_learning_rate = lr_DNS      # reload initial learning rate
            opt_m.initial_learning_rate = lr_LES      # reload initial learning rate

            # iterate on zlatent 1
            if (it<lr_kDNS_It):
                resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS \
                    = step_find_latents_kLES(wl_synthesis, filter, opt_k, zlatents, wlatents, fimgA, ltv_DNS)

            if (it==lr_kDNS_It):
                # find z1,w1
                k_new = layer_k.trainable_variables[0]
                z1 = k_new*zlatents
                layer_k.trainable_variables[0].assign(tf.fill((LATENT_SIZE), 1.0))
                w1 = mapping(z1, training=False)
                zlatents2 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
                wlatents2 = mapping(zlatents2, training=False)

            # iterate on zlatent 2
            if (it>=lr_kDNS_It and it<2*lr_kDNS_It):
                resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS \
                    = step_find_latents_kLES(wl_synthesis, filter, opt_k, zlatents2, wlatents2, fimgA, ltv_DNS)

            if (it==2*lr_kDNS_It):
                # find z2,w2
                k_new = layer_k.trainable_variables[0]
                z2 = k_new*zlatents2
                layer_k.trainable_variables[0].assign(tf.fill((LATENT_SIZE), 1.0))
                w2 = mapping(z2, training=False)
                
            # find new M
            if (it>=2*lr_kDNS_It):            
                resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS \
                    = step_find_latents_mLES(wl_synthesis, filter, opt_m, z2, w1, fimgA, ltv_LES)
                    
            if (it<2*lr_kDNS_It):
                lr = lr_schedule_k(it)
            else:
                lr = lr_schedule_m(it)
    
        else:

            opt_k.initial_learning_rate = lr      # reload initial learning rate
            opt_m.initial_learning_rate = lr      # reload initial learning rate
    
            resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS \
                = step_find_latents_mLES(wl_synthesis, filter, opt_m, z2, w1, fimgA, ltv_LES)

        if ((it+1)%100==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   step {1:6d}   it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
                .format(tend-tstart, pStep, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

        it = it+1

    # print("iterations  ", time.time() - tstart2)
    # print("LES iterations:  it {0:6d}  residuals {1:3e}".format(it, resREC.numpy()))


    resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_residuals(wl_synthesis, filter, z2, w1, fimgA)

    if (it>0):

        # # rescale according to LES min/max all fields
        # UVP_DNS  = rescale(UVP_DNS,  UVP_minmax)
        # UVP_LES  = rescale(UVP_LES,  UVP_minmax)    
        # fUVP_DNS = rescale(fUVP_DNS, UVP_minmax)    


        # print values
        if (pStep%1==0):
            filename = "./results_bout/plots/Plots_DNS_" + str(pStep).zfill(7)
            nc.savez(filename, U=UVP_DNS[0,0,:,:].numpy(), V=UVP_DNS[0,1,:,:].numpy(), P=UVP_DNS[0,2,:,:].numpy())

            filename = "./results_bout/plots/Plots_LES_" + str(pStep).zfill(7)
            nc.savez(filename, U=UVP_LES[0,0,:,:].numpy(), V=UVP_LES[0,1,:,:].numpy(), P=UVP_LES[0,2,:,:].numpy())

            filename = "./results_bout/plots/Plots_diffLES_" + str(pStep).zfill(7)
            diff =(UVP_LES-fUVP_DNS)
            nc.savez(filename, U=diff[0,0,:,:].numpy(), V=diff[0,1,:,:].numpy(), P=diff[0,2,:,:].numpy())

            filename = "./results_bout/plots/Plots_diffLESBOUT_" + str(pStep).zfill(7)
            diff =(UVP_LES-fimgA)
            nc.savez(filename, U=diff[0,0,:,:].numpy(), V=diff[0,1,:,:].numpy(), P=diff[0,2,:,:].numpy())

            filename = "./results_bout/plots/Plots_fromBOUT_" + str(pStep).zfill(7)
            nc.savez(filename, U=fimgA[0,0,:,:].numpy(), V=fimgA[0,1,:,:].numpy(), P=fimgA[0,2,:,:].numpy())
            

            # filename = "./results_bout/plots/Plots_DNS_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(UVP_DNS[0,0,:,:].numpy(), UVP_DNS[0,1,:,:].numpy(), UVP_DNS[0,2,:,:].numpy(), N_DNS, filename)

            # filename = "./results_bout/plots/Plots_LES_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(UVP_LES[0,0,:,:].numpy(), UVP_LES[0,1,:,:].numpy(), UVP_LES[0,2,:,:].numpy(), N_LES, filename)

            # filename = "./results_bout/plots/Plots_diffLES_" + str(pStep).zfill(7) + ".png"
            # diff =(UVP_LES-fUVP_DNS)
            # print_fields_3(diff[0,0,:,:].numpy(), diff[0,1,:,:].numpy(), diff[0,2,:,:].numpy(), N_LES, filename)

            # filename = "./results_bout/plots/Plots_diffLESBOUT_" + str(pStep).zfill(7) + ".png"
            # diff =(UVP_LES-fimgA)
            # print_fields_3(diff[0,0,:,:].numpy(), diff[0,1,:,:].numpy(), diff[0,2,:,:].numpy(), N_LES, filename)

            # filename = "./results_bout/plots/Plots_fromBOUT_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(fimgA[0,0,:,:].numpy(), fimgA[0,1,:,:].numpy(), fimgA[0,2,:,:].numpy(), N_LES, filename)


    

        #------------------------------------- find derivatives

        spacingFactor = tf.constant(1.0/(12.0*delx_LES*dely_LES), dtype="float32")

        # find pPHiVort_LES and pPhiN_LES
        F = UVP_LES[0, 1, :, :]
        G = UVP_LES[0, 2, :, :]
        pPHiVort_LES2 = find_bracket(F, G, spacingFactor)
        G = UVP_LES[0, 0, :, :]
        pPhiN_LES2 = find_bracket(F, G, spacingFactor)

        # find pDvort and pDn
        F = UVP_DNS[0, 1, :, :]
        G = UVP_DNS[0, 2, :, :]
        fpPhiVort_DNS, pDvort = find_bracket_diff(F, G, spacingFactor, pPHiVort_LES2)
        G = UVP_DNS[0, 0, :, :]
        fpPhiN_DNS, pDn = find_bracket_diff(F, G, spacingFactor, pPhiN_LES2)

        # print("derivatives ", time.time() - tstart2)


        
        # print Poisson brackets
        if (pStep==pStepStart):
            
            filename = "./results_bout/plots/Plots_fpPhiVort_" + str(pStep).zfill(7)
            nc.savez(filename, U=fpPhiVort_DNS.numpy(), V=pPHiVort_LES.numpy(), P=(fpPhiVort_DNS-pPHiVort_LES).numpy())

            filename = "./results_bout/plots/Plots_fpPhiN_" + str(pStep).zfill(7)
            nc.savez(filename, U=fpPhiN_DNS.numpy(), V=pPhiN_LES.numpy(), P=(fpPhiN_DNS-pPhiN_LES).numpy())

            # filename = "./results_bout/plots/Plots_fpPhiVort_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(fpPhiVort_DNS.numpy(), pPHiVort_LES.numpy(), (fpPhiVort_DNS-pPHiVort_LES).numpy(), N_LES, filename)
            # print_fields_3(fpPhiVort_DNS.numpy(), pPHiVort_LES.numpy(), (fpPhiVort_DNS-pPHiVort_LES).numpy(), N_LES, filename, \
            #                Umin=-0.2, Umax=0.2, Vmin=-0.2, Vmax=0.2, Pmin=-0.2, Pmax=0.2)
            
            # filename = "./results_bout/plots/Plots_fpPhiN_" + str(pStep).zfill(7) + ".png"
            # print_fields_3(fpPhiN_DNS.numpy(), pPhiN_LES.numpy(), (fpPhiN_DNS-pPhiN_LES).numpy(), N_LES, filename)
            # print_fields_3(fpPhiN_DNS.numpy(), pPhiN_LES.numpy(), (fpPhiN_DNS-pPhiN_LES).numpy(), N_LES, filename, \
            #                Umin=-0.005, Umax=0.005, Vmin=-0.005, Vmax=0.005, Pmin=-0.005, Pmax=0.005)


        
        
        # pass back diffusion terms
        pDvort = tf.reshape(pDvort, [-1])
        pDn    = tf.reshape(pDn,    [-1])
        
        pDvort = tf.cast(pDvort, dtype="float64")
        pDn    = tf.cast(pDn, dtype="float64")

        pDvort = 0.0*pDvort.numpy()
        pDn    = 0.0*pDn.numpy()

        # print(np.min(pDvort), np.max(pDvort))
        # print(np.min(pDn),    np.max(pDn))

    else:

        pDvort = np.zeros((N_LES, N_LES), dtype="float64")
        pDn    = np.zeros((N_LES, N_LES), dtype="float64")

        pDvort = tf.reshape(pDvort, [-1])
        pDn    = tf.reshape(pDn,    [-1])
                
    rLES = np.concatenate((pDvort, pDn), axis=0)

    # print("concatenate ", time.time() - tstart2)
    
    return rLES
    


# test
if (RUN_TEST):
    dummy=np.array([0.2, 0.2])
    initFlow(dummy)




#--------------------------------Extra pieces-----------------------------
