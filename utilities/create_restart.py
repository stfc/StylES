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

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *

os.chdir('../')
from MSG_StyleGAN_tf2 import *
os.chdir('./utilities')


# parameters
TUNE        = False
TUNE_NOISE  = False
tollLES     = 1.e-3
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2-FIL
zero_DNS    = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
CHKP_DIR_WL = "./checkpoints_wl"
RESTART_WL  = False
INIT_SCAL   = 10.0


if (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = 50.176

DELX  = L/N_DNS
DELY  = L/N_DNS


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



# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)


# create variable synthesis model
layer_LES = layer_wlatent_mLES()

w0           = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
w1           = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
w            = layer_LES(w0, w1)
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=[w0, w1], outputs=outputs)


# define checkpoints wl_synthesis and filter
checkpoint_wl        = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
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

# restart from defined values
if (RESTART_WL):

    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    data      = np.load("results_latentSpace/z0.npz")
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

else:             

    # set z
    z0 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
    z1 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART+1)
    w0 = mapping(z0, training=False)
    w1 = mapping(z1, training=False)


# find inference
resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_latents_LES_restart_A(wl_synthesis, filter, w0, w1)
print("Initial residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fill {3:3e} " \
        .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))



# tune to given tollerance
if (TUNE):
    # save old w
    wto = tf.identity(w0)

    # start search
    it     = 0
    tstart = time.time()
    while (resREC>tollLES and it<lr_LES_maxIt):                

        lr = lr_schedule_LES(it)
        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
            step_find_latents_LES_restart_B(wl_synthesis, filter, opt_LES, w0, w1, ltv_LES)
        
        mLES = layer_LES.trainable_variables[0]
        if (tf.reduce_min(mLES)<0 or tf.reduce_max(mLES)>1):
            print("Find new w1...")
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

        # print residuals
        if (it%100==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   it {1:6d}  residuals {2:3e}   lr {3:3e} ".format(tend-tstart, it, resREC.numpy(), lr))
                
            U_DNS = fUVP_DNS[0, 0, :, :].numpy()
            V_DNS = fUVP_DNS[0, 1, :, :].numpy()
            P_DNS = fUVP_DNS[0, 2, :, :].numpy()            

            if (TESTCASE=='HIT_2D'):
                filename = "results_reconstruction/plots/plots_restart.png"
                # filename = "results_reconstruction/plots/plots_restart_" + str(it) + ".png"
            elif(TESTCASE=='HW' or TESTCASE=='mHW'):
                os.system("mkdir -p ../bout_interfaces/restart_fromGAN/")
                filename = "../bout_interfaces/restart_fromGAN/plots_DNS_restart.png"
                # filename = "../bout_interfaces/plots_DNS_restart_" + str(it) + ".png"
            print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename)
                                    
        it = it+1

    tend = time.time()
    lr = lr_schedule_LES(it)
    print("LES iterations:  time {0:3e}   it {1:6d}  residuals {2:3e}   lr {3:3e} ".format(tend-tstart, it, resREC.numpy(), lr))

            
# save NN configuration
if (not RESTART_WL):
    managerCheckpoint_wl.save()

    # find z0
    z0 = z0.numpy()
    w1 = w1.numpy()

    # find kDNS
    mLES = layer_LES.trainable_variables[0].numpy()

    # find noise_DNS
    it=0
    noise_DNS=[]
    for layer in synthesis.layers:
        if "layer_noise_constants" in layer.name:
            noise_DNS.append(layer.trainable_variables[0].numpy())

    if (TESTCASE=='HIT_2D'):
        np.savez("results_latentSpace/z0.npz", z0=z0, w1=w1, mLES=mLES, noise_DNS=noise_DNS)
    elif(TESTCASE=='HW' or TESTCASE=='mHW'):
        os.system("mkdir -p ../bout_interfaces/restart_fromGAN/")
        np.savez("../bout_interfaces/restart_fromGAN/z0.npz", z0=z0, w1=w1, mLES=mLES, noise_DNS=noise_DNS)



# find fields
U_DNS = UVP_DNS[0, 0, :, :]
V_DNS = UVP_DNS[0, 1, :, :]
P_DNS = UVP_DNS[0, 2, :, :]

U_DNS = U_DNS*INIT_SCAL
V_DNS = V_DNS*INIT_SCAL
P_DNS = P_DNS*INIT_SCAL

U_DNS = U_DNS - tf.reduce_mean(U_DNS)
V_DNS = V_DNS - tf.reduce_mean(V_DNS)
P_DNS = P_DNS - tf.reduce_mean(P_DNS)

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




# print fields and energy spectra
if (TESTCASE=='HIT_2D'):

    filename = "../LES_Solvers/plots_restart.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, N=OUTPUT_DIM, filename=filename, testcase=TESTCASE)

    filename = "../LES_Solvers/restart"
    save_fields(0.6, U_DNS, V_DNS, P_DNS, filename=filename)  # Note: t=0.6 is the corrisponding time to t=545 tau_e

    filename = "../LES_Solvers/energy_spectrum_restart.png"
    closePlot=True
    plot_spectrum(U_DNS, V_DNS, L, filename, close=closePlot)

elif(TESTCASE=='HW' or TESTCASE=='mHW'):

    DELX = L/N_LES
    DELY = L/N_LES
    
    filename = "../bout_interfaces/restart_fromGAN/plots_DNS_restart.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename, testcase=TESTCASE)

    filename = "../bout_interfaces/restart_fromGAN/plots_LES_restart.png"
    print_fields_3(U_LES, V_LES, P_LES, N=N_LES, filename=filename, testcase=TESTCASE)

    filename = "../bout_interfaces/restart_fromGAN/plots_fDNS_restart.png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, N=N_LES, filename=filename, testcase=TESTCASE)

    filename = "../bout_interfaces/restart_fromGAN/restart_UVPLES"
    save_fields(0.0, U_LES, V_LES, P_LES, filename=filename)

    filename = "../bout_interfaces/restart_fromGAN/energy_spectrum_restart.png"
    closePlot=False
    gradV_DNS = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*DELX))**2 \
                      + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*DELY))**2)
    plot_spectrum(U_DNS, gradV_DNS, L, filename, close=closePlot)

    filename = "../bout_interfaces/restart_fromGAN/energy_spectrum_restart.png"
    closePlot=True
    gradV_LES = np.sqrt(((cr(V_LES, 1, 0) - cr(V_LES, -1, 0))/(2.0*DELX))**2 \
                      + ((cr(V_LES, 0, 1) - cr(V_LES, 0, -1))/(2.0*DELY))**2)
    plot_spectrum(U_LES, gradV_LES, L, filename, close=closePlot)
