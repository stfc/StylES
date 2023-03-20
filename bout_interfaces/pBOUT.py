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

from LES_constants import *
from LES_parameters import *
from LES_plot import *

from MSG_StyleGAN_tf2 import *



# local parameters
TUNE_NOISE  = True
NITEZ       = 0   # number of attempts to find a closer z. When restart from a GAN field, use NITEZ=0
RELOAD_FREQ = 10
N_DNS       = 2**RES_LOG2
N_LES       = 2**(RES_LOG2-FIL)
RUN_REST    = False
delx        = 1.0
dely        = 1.0
delx_LES    = 1.0
dely_LES    = 1.0
tollLES     = 1.e-1
FILTER_SIG  = 2
step        = 0
INI_SCALING = 10.0  # to do: define a better rescaling

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
    '/lustre/scafellpike/local/HT04543/jxc06/jxc74-jxc06/projects/Turbulence_with_Style/PhaseII_FARSCAPE2/codes/StylES/checkpoints/',
    max_to_keep=1)
checkpoint.restore(managerCheckpoint.latest_checkpoint)

if managerCheckpoint.latest_checkpoint:
    print("StyleGAN restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing StyleGAN from scratch.")




# create variable synthesis model
layer_LES = layer_wlatent_LES()

zlatents     = tf.keras.Input(shape=([LATENT_SIZE]), dtype=DTYPE)
wlatents     = mapping(zlatents)
wlatents_LES = layer_LES(wlatents)
outputs      = synthesis(wlatents_LES, training=False)
wl_synthesis = tf.keras.Model(inputs=zlatents, outputs=outputs)



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




# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    ltv_LES = []

for variable in layer_LES.trainable_variables:
    ltv_DNS.append(variable)
    # ltv_LES.append(variable)


print("\n DNS variables:")
for variable in ltv_DNS:
    print(variable.name)

print("\n LES variables:")
for variable in ltv_LES:
    print(variable.name)


# set z latent space
zlatent = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)












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
    predictions = wl_synthesis(zlatent, training=False)



    #---------------- find LES fields from GAN
    UVP_LES = predictions[RES_LOG2-FIL-2]*INI_SCALING    # to do: need a better init values

    # split fields
    U_LES = 0.0*UVP_LES[0,0,:,:]  # to do: need a better init values
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

    #---------------- print min/max values and shape
    print("Min/Max n",    U_min, U_max)
    print("Min/Max phi",  V_min, V_max)
    print("Min/Max vort", P_min, P_max)
    print("Tot vort",     np.sum(P_LES))
    
    print(U_LES.shape)



        
    #---------------- find DNS fields from StyleGAN
    UVP_DNS = predictions[RES_LOG2-2]

    # split fields
    U_DNS = UVP_DNS[0,0,:,:]
    V_DNS = UVP_DNS[0,1,:,:]
    P_DNS = UVP_DNS[0,2,:,:]
    
    # make sure vorticity is overall null
    P_DNS = P_DNS - tf.math.reduce_mean(P_DNS)
    
    # rescale according to LES min/max (as this will be used in BOUT++)
    u_min =  tf.reduce_min(U_DNS)
    v_min =  tf.reduce_min(V_DNS)
    p_min =  tf.reduce_min(P_DNS)

    U_DNS = (U_DNS - u_min)/(tf.reduce_max(U_DNS) - u_min)
    V_DNS = (V_DNS - v_min)/(tf.reduce_max(V_DNS) - v_min)
    P_DNS = (P_DNS - p_min)/(tf.reduce_max(P_DNS) - p_min)

    U_DNS = (U_max - U_min)*U_DNS + U_min
    V_DNS = (V_max - V_min)*V_DNS + V_min
    P_DNS = (P_max - P_min)*P_DNS + P_min
    
    # find grad Phi for energy estimation
    gradV_DNS = tf.sqrt(((tr(V_DNS, 1, 0) - tr(V_DNS, -1, 0))/(2.0*delx))**2 + ((tr(V_DNS, 0, 1) - tr(V_DNS, 0, -1))/(2.0*dely))**2)

    # pass to Numpy
    U_DNS     = U_DNS.numpy()
    V_DNS     = V_DNS.numpy()
    P_DNS     = P_DNS.numpy()
    gradV_DNS = gradV_DNS.numpy()



    #---------------- find also the LES fields filtering DNS 
    fUVP_DNS = predictions[RES_LOG2-2]

    # filter fields
    fU_DNS = fUVP_DNS[:,0,:,:]
    fU_DNS = fU_DNS[:,tf.newaxis,:,:]
    fU_DNS = filter(fU_DNS, training = False)

    fV_DNS = fUVP_DNS[:,1,:,:]
    fV_DNS = fV_DNS[:,tf.newaxis,:,:]
    fV_DNS = filter(fV_DNS, training = False)

    fP_DNS = fUVP_DNS[:,2,:,:]
    fP_DNS = fP_DNS[:,tf.newaxis,:,:]
    fP_DNS = filter(fP_DNS, training = False)

    fUVP_DNS =  tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)

    fU_DNS = fU_DNS[0,0,:,:]
    fV_DNS = fV_DNS[0,0,:,:]
    fP_DNS = fP_DNS[0,0,:,:]
    
    
    # make sure vorticity is overall null
    fP_DNS = fP_DNS - tf.math.reduce_mean(fP_DNS)
    
    # rescale according to LES min/max (as this will be used in BOUT++)
    u_min =  tf.reduce_min(fU_DNS)
    v_min =  tf.reduce_min(fV_DNS)
    p_min =  tf.reduce_min(fP_DNS)

    fU_DNS = (fU_DNS - u_min)/(tf.reduce_max(fU_DNS) - u_min)
    fV_DNS = (fV_DNS - v_min)/(tf.reduce_max(fV_DNS) - v_min)
    fP_DNS = (fP_DNS - p_min)/(tf.reduce_max(fP_DNS) - p_min)

    fU_DNS = (U_max - U_min)*fU_DNS + U_min
    fV_DNS = (V_max - V_min)*fV_DNS + V_min
    fP_DNS = (P_max - P_min)*fP_DNS + P_min
    
    # find grad Phi for energy estimation
    fgradV_DNS = tf.sqrt(((tr(fV_DNS, 1, 0) - tr(fV_DNS, -1, 0))/(2.0*delx))**2 + ((tr(fV_DNS, 0, 1) - tr(fV_DNS, 0, -1))/(2.0*dely))**2)

    # pass to Numpy
    fU_DNS     = fU_DNS.numpy()
    fV_DNS     = fV_DNS.numpy()
    fP_DNS     = fP_DNS.numpy()
    fgradV_DNS = fgradV_DNS.numpy()



    #---------------- plot values
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
def step_find_residuals(latents, fimgA, ltv, UVP_minmax):

    # pass min/max values
    U_min = UVP_minmax[0]
    U_max = UVP_minmax[1]
    V_min = UVP_minmax[2]
    V_max = UVP_minmax[3]
    P_min = UVP_minmax[4]
    P_max = UVP_minmax[5]
    
    # find predictions
    predictions = wl_synthesis(latents, training=False)
    UVP_DNS = predictions[RES_LOG2-2]
    UVP_LES = predictions[RES_LOG2-FIL-2]

    # filter DNS field
    fUVP_DNS = predictions[RES_LOG2-2]


    #-------------- rescale DNS fields
    U_DNS = UVP_DNS[0,0,:,:]
    V_DNS = UVP_DNS[0,1,:,:]
    P_DNS = UVP_DNS[0,2,:,:]
    
    # make sure vorticity is overall null
    vort_tot = tf.math.reduce_mean(P_DNS)
    P_DNS = P_DNS-vort_tot

    # find min/max
    u_min =  tf.reduce_min(U_DNS)
    v_min =  tf.reduce_min(V_DNS)
    p_min =  tf.reduce_min(P_DNS)

    # rescale
    U_DNS = (U_DNS - u_min)/(tf.reduce_max(U_DNS) - u_min)
    V_DNS = (V_DNS - v_min)/(tf.reduce_max(V_DNS) - v_min)
    P_DNS = (P_DNS - p_min)/(tf.reduce_max(P_DNS) - p_min)

    U_DNS = (U_max - U_min)*U_DNS + U_min
    V_DNS = (V_max - V_min)*V_DNS + V_min
    P_DNS = (P_max - P_min)*P_DNS + P_min
    
    UVP_DNS = tf.concat([U_DNS[tf.newaxis,tf.newaxis,:,:], V_DNS[tf.newaxis,tf.newaxis,:,:], P_DNS[tf.newaxis,tf.newaxis,:,:]], axis=1)


    #-------------- rescale LES fields
    U_LES = UVP_LES[0,0,:,:]
    V_LES = UVP_LES[0,1,:,:]
    P_LES = UVP_LES[0,2,:,:]
    
    # make sure vorticity is overall null
    vort_tot = tf.math.reduce_mean(P_LES)
    P_LES = P_LES-vort_tot

    # find min/max
    u_min =  tf.reduce_min(U_LES)
    v_min =  tf.reduce_min(V_LES)
    p_min =  tf.reduce_min(P_LES)

    # rescale
    U_LES = (U_LES - u_min)/(tf.reduce_max(U_LES) - u_min)
    V_LES = (V_LES - v_min)/(tf.reduce_max(V_LES) - v_min)
    P_LES = (P_LES - p_min)/(tf.reduce_max(P_LES) - p_min)

    U_LES = (U_max - U_min)*U_LES + U_min
    V_LES = (V_max - V_min)*V_LES + V_min
    P_LES = (P_max - P_min)*P_LES + P_min
    
    UVP_LES = tf.concat([U_LES[tf.newaxis,tf.newaxis,:,:], V_LES[tf.newaxis,tf.newaxis,:,:], P_LES[tf.newaxis,tf.newaxis,:,:]], axis=1)


    #-------------- rescale filtered DNS fields
    fU_DNS = fUVP_DNS[:,0,:,:]
    fU_DNS = fU_DNS[:,tf.newaxis,:,:]
    fU_DNS = filter(fU_DNS, training = False)

    fV_DNS = fUVP_DNS[:,1,:,:]
    fV_DNS = fV_DNS[:,tf.newaxis,:,:]
    fV_DNS = filter(fV_DNS, training = False)

    fP_DNS = fUVP_DNS[:,2,:,:]
    fP_DNS = fP_DNS[:,tf.newaxis,:,:]
    fP_DNS = filter(fP_DNS, training = False)

    fU_DNS = fU_DNS[0,0,:,:]
    fV_DNS = fV_DNS[0,0,:,:]
    fP_DNS = fP_DNS[0,0,:,:]
    
    # make sure vorticity is overall null
    vort_tot = tf.math.reduce_mean(fP_DNS)
    fP_DNS = fP_DNS-vort_tot

    # find min/max
    u_min =  tf.reduce_min(fU_DNS)
    v_min =  tf.reduce_min(fV_DNS)
    p_min =  tf.reduce_min(fP_DNS)

    # rescale
    fU_DNS = (fU_DNS - u_min)/(tf.reduce_max(fU_DNS) - u_min)
    fV_DNS = (fV_DNS - v_min)/(tf.reduce_max(fV_DNS) - v_min)
    fP_DNS = (fP_DNS - p_min)/(tf.reduce_max(fP_DNS) - p_min)

    fU_DNS = (U_max - U_min)*fU_DNS + U_min
    fV_DNS = (V_max - V_min)*fV_DNS + V_min
    fP_DNS = (P_max - P_min)*fP_DNS + P_min
    
    fUVP_DNS = tf.concat([fU_DNS[tf.newaxis,tf.newaxis,:,:], fV_DNS[tf.newaxis,tf.newaxis,:,:], fP_DNS[tf.newaxis,tf.newaxis,:,:]], axis=1)


    # find residuals
    resDNS = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))
    resLES = tf.math.reduce_mean(tf.math.squared_difference( UVP_LES, fimgA))

    resREC = resDNS + resLES

    # find filter loss
    loss_fil = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))

    return resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS




@tf.function
def step_find_latents_LES(latents, fimgA, ltv, UVP_minmax):
    with tf.GradientTape() as tape_LES:

        # pass min/max values
        U_min = UVP_minmax[0]
        U_max = UVP_minmax[1]
        V_min = UVP_minmax[2]
        V_max = UVP_minmax[3]
        P_min = UVP_minmax[4]
        P_max = UVP_minmax[5]
        
        # find predictions
        predictions = wl_synthesis(latents, training=False)
        UVP_DNS = predictions[RES_LOG2-2]
        UVP_LES = predictions[RES_LOG2-FIL-2]

        # filter DNS field
        fUVP_DNS = predictions[RES_LOG2-2]


        #-------------- rescale DNS fields
        U_DNS = UVP_DNS[0,0,:,:]
        V_DNS = UVP_DNS[0,1,:,:]
        P_DNS = UVP_DNS[0,2,:,:]
        
        # make sure vorticity is overall null
        vort_tot = tf.math.reduce_mean(P_DNS)
        P_DNS = P_DNS-vort_tot

        # find min/max
        u_min =  tf.reduce_min(U_DNS)
        v_min =  tf.reduce_min(V_DNS)
        p_min =  tf.reduce_min(P_DNS)

        # rescale
        U_DNS = (U_DNS - u_min)/(tf.reduce_max(U_DNS) - u_min)
        V_DNS = (V_DNS - v_min)/(tf.reduce_max(V_DNS) - v_min)
        P_DNS = (P_DNS - p_min)/(tf.reduce_max(P_DNS) - p_min)

        U_DNS = (U_max - U_min)*U_DNS + U_min
        V_DNS = (V_max - V_min)*V_DNS + V_min
        P_DNS = (P_max - P_min)*P_DNS + P_min
        
        UVP_DNS = tf.concat([U_DNS[tf.newaxis,tf.newaxis,:,:], V_DNS[tf.newaxis,tf.newaxis,:,:], P_DNS[tf.newaxis,tf.newaxis,:,:]], axis=1)


        #-------------- rescale LES fields
        U_LES = UVP_LES[0,0,:,:]
        V_LES = UVP_LES[0,1,:,:]
        P_LES = UVP_LES[0,2,:,:]
        
        # make sure vorticity is overall null
        vort_tot = tf.math.reduce_mean(P_LES)
        P_LES = P_LES-vort_tot

        # find min/max
        u_min =  tf.reduce_min(U_LES)
        v_min =  tf.reduce_min(V_LES)
        p_min =  tf.reduce_min(P_LES)

        # rescale
        U_LES = (U_LES - u_min)/(tf.reduce_max(U_LES) - u_min)
        V_LES = (V_LES - v_min)/(tf.reduce_max(V_LES) - v_min)
        P_LES = (P_LES - p_min)/(tf.reduce_max(P_LES) - p_min)

        U_LES = (U_max - U_min)*U_LES + U_min
        V_LES = (V_max - V_min)*V_LES + V_min
        P_LES = (P_max - P_min)*P_LES + P_min
        
        UVP_LES = tf.concat([U_LES[tf.newaxis,tf.newaxis,:,:], V_LES[tf.newaxis,tf.newaxis,:,:], P_LES[tf.newaxis,tf.newaxis,:,:]], axis=1)


        #-------------- rescale filtered DNS fields
        fU_DNS = fUVP_DNS[:,0,:,:]
        fU_DNS = fU_DNS[:,tf.newaxis,:,:]
        fU_DNS = filter(fU_DNS, training = False)

        fV_DNS = fUVP_DNS[:,1,:,:]
        fV_DNS = fV_DNS[:,tf.newaxis,:,:]
        fV_DNS = filter(fV_DNS, training = False)

        fP_DNS = fUVP_DNS[:,2,:,:]
        fP_DNS = fP_DNS[:,tf.newaxis,:,:]
        fP_DNS = filter(fP_DNS, training = False)

        fU_DNS = fU_DNS[0,0,:,:]
        fV_DNS = fV_DNS[0,0,:,:]
        fP_DNS = fP_DNS[0,0,:,:]
        
        # make sure vorticity is overall null
        vort_tot = tf.math.reduce_mean(fP_DNS)
        fP_DNS = fP_DNS-vort_tot

        # find min/max
        u_min =  tf.reduce_min(fU_DNS)
        v_min =  tf.reduce_min(fV_DNS)
        p_min =  tf.reduce_min(fP_DNS)

        # rescale
        fU_DNS = (fU_DNS - u_min)/(tf.reduce_max(fU_DNS) - u_min)
        fV_DNS = (fV_DNS - v_min)/(tf.reduce_max(fV_DNS) - v_min)
        fP_DNS = (fP_DNS - p_min)/(tf.reduce_max(fP_DNS) - p_min)

        fU_DNS = (U_max - U_min)*fU_DNS + U_min
        fV_DNS = (V_max - V_min)*fV_DNS + V_min
        fP_DNS = (P_max - P_min)*fP_DNS + P_min
        
        fUVP_DNS = tf.concat([fU_DNS[tf.newaxis,tf.newaxis,:,:], fV_DNS[tf.newaxis,tf.newaxis,:,:], fP_DNS[tf.newaxis,tf.newaxis,:,:]], axis=1)


        # find residuals
        resDNS = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))
        resLES = tf.math.reduce_mean(tf.math.squared_difference( UVP_LES, fimgA))

        resREC = resDNS + resLES

        # apply gradients
        gradients_LES = tape_LES.gradient(resREC, ltv)
        opt_LES.apply_gradients(zip(gradients_LES, ltv))

        # find filter loss
        loss_fil = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))

        
    return resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS





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


    # preprare target image
    U_LES = tf.convert_to_tensor(U_LES)
    V_LES = tf.convert_to_tensor(V_LES)
    P_LES = tf.convert_to_tensor(P_LES)

    pPHiVort_LES = tf.convert_to_tensor(pPHiVort_LES)
    pPhiN_LES    = tf.convert_to_tensor(pPhiN_LES)        

    U_LES = U_LES[tf.newaxis,tf.newaxis,:,:]
    V_LES = V_LES[tf.newaxis,tf.newaxis,:,:]
    P_LES = P_LES[tf.newaxis,tf.newaxis,:,:]

    fimgA  = tf.concat([U_LES, V_LES, P_LES], 1)

    # print("preprare    ", time.time() - tstart2)

    # if (pStep%1000==0):
    #     checkpoint.restore(managerCheckpoint.latest_checkpoint)
    
    # find reconstructed field
    it = 0
    resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_residuals(zlatent, fimgA, ltv_DNS, UVP_minmax)
    opt_LES.initial_learning_rate = lr_LES      # reload initial learning rate
    tstart = time.time()
    while (resREC.numpy()>tollLES and it<100000):   # to do: set back to lr_LES_maxIt
        lr = lr_schedule_LES(it)

        resREC, resLES, resDNS, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_latents_LES(zlatent, fimgA, ltv_DNS, UVP_minmax)
            
        if ((it+1)%100==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   it {1:6d}  residuals {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fill {5:3e}  lr {6:3e} " \
                .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

        it = it+1

    # print("iterations  ", time.time() - tstart2)

    # print("LES iterations:  it {0:6d}  residuals {1:3e}".format(it, resREC.numpy()))

   
    # print values
    if (pStep%50==0):
        filename = "./results_bout/plots/Plots_DNS_" + str(pStep).zfill(7) + ".png"
        print_fields_3(UVP_DNS[0,0,:,:].numpy(), UVP_DNS[0,1,:,:].numpy(), UVP_DNS[0,2,:,:].numpy(), N_DNS, filename)

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


    #------------------------- find pDvort
    F = UVP_DNS[0, 1, :, :]
    G = UVP_DNS[0, 2, :, :]

    fpPhiVort_DNS, pDvort = find_bracket_diff(F, G, spacingFactor, pPHiVort_LES)


    #------------------------- find pDn
    G = UVP_DNS[0, 0, :, :]

    fpPhiN_DNS, pDn = find_bracket_diff(F, G, spacingFactor, pPhiN_LES)

    # print("derivatives ", time.time() - tstart2)


    
    # print Poisson brackets
    if (pStep==pStepStart):
        
        filename = "./results_bout/plots/Plots_fpPhiVort_" + str(pStep).zfill(7) + ".png"
        print_fields_3(fpPhiVort_DNS.numpy(), pPHiVort_LES.numpy(), (fpPhiVort_DNS-pPHiVort_LES).numpy(), N_LES, filename)
        # print_fields_3(fpPhiVort_DNS.numpy(), pPHiVort_LES.numpy(), (fpPhiVort_DNS-pPHiVort_LES).numpy(), N_LES, filename, \
        #                Umin=-0.2, Umax=0.2, Vmin=-0.2, Vmax=0.2, Pmin=-0.2, Pmax=0.2)
        
        filename = "./results_bout/plots/Plots_fpPhiN_" + str(pStep).zfill(7) + ".png"
        print_fields_3(fpPhiN_DNS.numpy(), pPhiN_LES.numpy(), (fpPhiN_DNS-pPhiN_LES).numpy(), N_LES, filename)
        # print_fields_3(fpPhiN_DNS.numpy(), pPhiN_LES.numpy(), (fpPhiN_DNS-pPhiN_LES).numpy(), N_LES, filename, \
        #                Umin=-0.005, Umax=0.005, Vmin=-0.005, Vmax=0.005, Pmin=-0.005, Pmax=0.005)


    
    
    # pass back diffusion terms
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
if (RUN_REST):
    dummy=np.array([0.2, 0.2])
    initFlow(dummy)




#--------------------------------Extra pieces-----------------------------
