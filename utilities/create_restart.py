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

tf.random.set_seed(SEED_RESTART)


# parameters
TUNE        = False
TUNE_NOISE  = False
tollDNS     = 1.e-2
tollLES     = 1.e-2
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2-FIL
zero_DNS    = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
RESTART_WL  = False
maxitLES    = 1000

if (TESTCASE=='HIT_2D'):
    from HIT_2D import L
    os.system("mkdir -p ../LES_Solvers/restart_fromGAN/")
    Z0_DIR_WL = "../LES_Solvers/restart_fromGAN/"
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = 50.176
    os.system("mkdir -p ../bout_interfaces/restart_fromGAN/")    
    Z0_DIR_WL = "../bout_interfaces/restart_fromGAN/"

os.system("rm -rf " + Z0_DIR_WL)
os.system("mkdir -p " + Z0_DIR_WL)
CHKP_DIR_WL = Z0_DIR_WL + "checkpoints_wl/"
DELX  = L/N_DNS
DELY  = L/N_DNS



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

if (lr_LES_POLICY=="EXPONENTIAL"):
    lr_schedule_LES  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_LES,
        decay_steps=lr_LES_STEP,
        decay_rate=lr_LES_RATE,
        staircase=lr_LES_EXP_ST)
elif (lr_LES_POLICY=="PIECEWISE"):
    lr_schedule_LES = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_LES_BOUNDS, lr_LES_VALUES)
opt_mLES = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_LES)



# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)


# create variable synthesis model
layer_kDNS = layer_zlatent_kDNS()
layer_mLES = layer_wlatent_mLES()

z            = tf.keras.Input(shape=([4, LATENT_SIZE]), dtype=DTYPE)
z0, z1       = layer_kDNS(z)
w0           = mapping(z0, training=False)
w1           = mapping(z1, training=False)
w            = layer_mLES(w0, w1)
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=z, outputs=[outputs, w])


# define checkpoints wl_synthesis and filter
checkpoint_wl        = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    ltv_LES = []
    
for variable in layer_kDNS.trainable_variables:
    ltv_DNS.append(variable)

for variable in layer_mLES.trainable_variables:
    ltv_LES.append(variable)

print("\n kDNS variables:")
for variable in ltv_DNS:
    print(variable.name)

print("\n mLES variables:")
for variable in ltv_LES:
    print(variable.name)



# restart from defined values
if (RESTART_WL):

    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    filename = Z0_DIR_WL + "z0.npz"
                
    data = np.load(filename)

    z          = data["z"]
    kDNS       = data["kDNS"]
    mLES       = data["mLES"]
    noise_LES  = data["noise_LES"]

    print("z",    z.shape,    np.min(z),    np.max(z))
    print("kDNS", kDNS.shape, np.min(kDNS), np.max(kDNS))
    print("mLES", mLES.shape, np.min(mLES), np.max(mLES))
    if (noise_LES.size>0):
        print("noise_LES",  noise_LES.shape,  np.min(noise_LES),  np.max(noise_LES))

    # convert to TensorFlow tensors
    z          = tf.convert_to_tensor(z,          dtype=DTYPE)
    kDNS       = tf.convert_to_tensor(kDNS,       dtype=DTYPE)
    mLES       = tf.convert_to_tensor(mLES,       dtype=DTYPE)
    noise_LES  = tf.convert_to_tensor(noise_LES,  dtype=DTYPE)

    # assign kDNS
    layer_kDNS.trainable_variables[0].assign(kDNS)
    layer_mLES.trainable_variables[0].assign(mLES)

    # assign variable noise
    it=0
    for layer in wl_synthesis.layers:
        if "layer_noise_constants" in layer.name:
            print(layer.trainable_variables)
            layer.trainable_variables[0].assign(noise_LES[it])
            it=it+1

else:             

    # set z
    z = tf.random.uniform([BATCH_SIZE, 4, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)


# find inference
UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, filter, z, INIT_SCAL)
resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS, UVP_LES, typeRes=0)
print("Initial residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
        .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))


# tune to given tollerance
if (TUNE):

    # save old w
    z0to = tf.identity(z[:,0,:])
    z2to = tf.identity(z[:,2,:])
    k0DNSo = layer_kDNS.trainable_variables[0][0,:]
    k1DNSo = layer_kDNS.trainable_variables[0][1,:]
    
    # start search
    it     = 0
    tstart = time.time()
    while (resREC>tollLES and it<lr_LES_maxIt):

        if (resREC>tollDNS and it<lr_DNS_maxIt):

            resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
                step_find_zlatents_kDNS(wl_synthesis, filter, opt_kDNS, z, UVP_DNS, UVP_LES, ltv_DNS, INIT_SCAL, typeRes=0)

            # check z0 and z1
            k0DNS = layer_kDNS.trainable_variables[0][0,:]
            k1DNS = layer_kDNS.trainable_variables[0][1,:]
            if (tf.reduce_min(k0DNS)<0 or tf.reduce_max(k0DNS)>1 or tf.reduce_min(k1DNS)<0 or tf.reduce_max(k1DNS)>1):

                print("Find new z...")
                
                zt = k0DNSo*z[:,0,:] + (1.0-k0DNSo)*z[:,1,:]
                z1 = 2.0*zt - z0
                    
                k0DNSo = tf.fill((LATENT_SIZE), 0.5)
                k0DNSo = tf.cast(k0DNSo, dtype=DTYPE)

                zt = k1DNSo*z[:,2,:] + (1.0-k1DNSo)*z[:,3,:]
                z3 = 2.0*zt - z2
                
                k1DNSo = tf.fill((LATENT_SIZE), 0.5)
                k1DNSo = tf.cast(k1DNSo, dtype=DTYPE)
            else:
                k0DNSo = tf.identity(k0DNS)
                k1DNSo = tf.identity(k1DNS)

                z0 = tf.identity(z[:,0,:])
                z1 = tf.identity(z[:,1,:])
                z2 = tf.identity(z[:,2,:])
                z3 = tf.identity(z[:,3,:])
            
            z = tf.concat([z0,z1,z2,z3], axis=0)
            z = z[tf.newaxis,:,:]

            k0DNS = k0DNSo[tf.newaxis,:]
            k1DNS = k1DNSo[tf.newaxis,:]
            kDNSn = tf.concat([k0DNS, k1DNS], axis=0)
            layer_kDNS.trainable_variables[0].assign(kDNSn)

        else:

            resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
                step_find_wlatents_mLES(wl_synthesis, filter, opt_mLES, z, UVP_DNS, UVP_LES, ltv_LES, INIT_SCAL, typeRes=0)

            mLESc = layer_mLES.trainable_variables[0]
            mLESc = tf.clip_by_value(mLESc, 0.0, 1.0)
            layer_mLES.trainable_variables[0].assign(mLESc)

        # find correct inference
        UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, filter, z, INIT_SCAL)
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS, UVP_LES, typeRes=0)

        # print fields
        if (it%1==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   it {1:6d}  resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} " \
                .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))
            
            if (it%100==0):
                U_DNS = UVP_DNS[0, 0, :, :].numpy()
                V_DNS = UVP_DNS[0, 1, :, :].numpy()
                P_DNS = UVP_DNS[0, 2, :, :].numpy()

                # filename =  Z0_DIR_WL + "plots_restart.png"
                filename =  Z0_DIR_WL + "plots_restart_" + str(it).zfill(4) + ".png"

                print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename)
    
        it = it+1

    # print final iteration
    tend = time.time()
    print("LES iterations:  time {0:3e}   it {1:6d}  resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} " \
        .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))
            

# save NN configuration
if (not RESTART_WL):
    managerCheckpoint_wl.save()

    # find z
    z = z.numpy()

    # find kDNS
    kDNS = layer_kDNS.trainable_variables[0].numpy()
    mLES = layer_mLES.trainable_variables[0].numpy()

    # find noise_LES
    it=0
    noise_LES=[]
    for layer in wl_synthesis.layers:
        if "layer_noise_constants" in layer.name:
            noise_LES.append(layer.trainable_variables[:].numpy())

    filename =  Z0_DIR_WL + "z0.npz"

    np.savez(filename,
            z=z, \
            kDNS=kDNS, \
            mLES=mLES, \
            noise_LES=noise_LES)


# find fields
U_DNS = UVP_DNS[0, 0, :, :]
V_DNS = UVP_DNS[0, 1, :, :]
P_DNS = UVP_DNS[0, 2, :, :]

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

    filename = Z0_DIR_WL + "plots_restart.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, N=OUTPUT_DIM, filename=filename, testcase=TESTCASE)

    filename = Z0_DIR_WL + "restart"
    save_fields(0.6, U_DNS, V_DNS, P_DNS, filename=filename)  # Note: t=0.6 is the corrisponding time to t=545 tau_e

    filename = Z0_DIR_WL + "energy_spectrum_restart.png"
    closePlot=True
    plot_spectrum(U_DNS, V_DNS, L, filename, close=closePlot)

elif(TESTCASE=='HW' or TESTCASE=='mHW'):

    DELX = L/N_LES
    DELY = L/N_LES
    
    filename = Z0_DIR_WL + "plots_DNS_restart.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename, testcase=TESTCASE) #, \
                   #Umin=-13.0, Umax=13.0, Vmin=-13.0, Vmax=13.0, Pmin=-13.0, Pmax=13.0)

    filename = Z0_DIR_WL + "plots_LES_restart.png"
    print_fields_3(U_LES, V_LES, P_LES, N=N_LES, filename=filename, testcase=TESTCASE) #, \
                    #Umin=-13.0, Umax=13.0, Vmin=-13.0, Vmax=13.0, Pmin=-13.0, Pmax=13.0)

    filename = Z0_DIR_WL + "plots_fDNS_restart.png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, N=N_LES, filename=filename, testcase=TESTCASE) #, \
                    #Umin=-13.0, Umax=13.0, Vmin=-13.0, Vmax=13.0, Pmin=-13.0, Pmax=13.0)

    filename = Z0_DIR_WL + "restart_UVPLES"
    save_fields(0.0, U_LES, V_LES, P_LES, filename=filename)

    filename = Z0_DIR_WL + "energy_spectrum_restart.png"
    closePlot=False
    gradV_DNS = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*DELX))**2 \
                      + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*DELY))**2)
    plot_spectrum(U_DNS, gradV_DNS, L, filename, close=closePlot)

    filename = Z0_DIR_WL + "energy_spectrum_restart.png"
    closePlot=True
    gradV_LES = np.sqrt(((cr(V_LES, 1, 0) - cr(V_LES, -1, 0))/(2.0*DELX))**2 \
                      + ((cr(V_LES, 0, 1) - cr(V_LES, 0, -1))/(2.0*DELY))**2)
    plot_spectrum(U_LES, gradV_LES, L, filename, close=closePlot)



