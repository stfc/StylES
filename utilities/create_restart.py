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
from MSG_StyleGAN_tf2 import *

tf.random.set_seed(SEED_RESTART)


# parameters
TUNE        = True
TUNE_NOISE  = False
tollDNS     = 1.0e-2
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2-FIL
RESTART_WL  = False
INIT_SCAL   = 10

if (TESTCASE=='HIT_2D'):
    from HIT_2D import L
    os.system("mkdir -p ../LES_Solvers/restart_fromGAN/")
    Z0_DIR_WL = "../LES_Solvers/restart_fromGAN/"
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = 50.176
    os.system("mkdir -p ../bout_interfaces/restart_fromGAN/")    
    Z0_DIR_WL = "../bout_interfaces/restart_fromGAN/"

if (not RESTART_WL):
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


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)




# create variable synthesis model

#--------------------------------------- model 1
# layer_kDNS = layer_zlatent_kDNS()

# z            = tf.keras.Input(shape=([2*M_LAYERS+1, LATENT_SIZE]), dtype=DTYPE)
# w            = layer_kDNS(mapping, z)
# outputs      = synthesis(w, training=False)
# wl_synthesis = tf.keras.Model(inputs=z, outputs=[outputs, w])


#--------------------------------------- model 2
# layer_kDNS = layer_create_noise_z0([M_LAYERS, LATENT_SIZE], 0, randomize_noise=False, name="layer_noise_constants_z0")
# layer_kLES = layer_zlatent_kDNS2()
# noise      = tf.constant(1, shape=[1, 1, M_LAYERS, LATENT_SIZE], dtype=DTYPE)
# noise_z0   = tf.random.uniform([BATCH_SIZE, 1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART, dtype=DTYPE)

# phi_noise_in = tf.keras.Input(shape=([NC2_NOISE,1]), dtype=DTYPE)
# z            = layer_kDNS(noise, phi_noise_in)
# z            = z[tf.newaxis,:,:]
# z            = tf.concat([z, noise_z0], axis=1)
# w            = layer_kLES(mapping, z)
# outputs      = synthesis(w, training=False)
# wl_synthesis = tf.keras.Model(inputs=phi_noise_in, outputs=[outputs, w])


# #--------------------------------------- model 3
# layer_kDNS = layer_wlatent_mLES()

# z_in         = tf.keras.Input(shape=([LATENT_SIZE]), dtype=DTYPE)
# w0           = mapping( z_in, training=False)
# w1           = mapping(-z_in, training=False)  
# w            = layer_kDNS(w0, w1)
# outputs      = synthesis(w, training=False)
# wl_synthesis = tf.keras.Model(inputs=z_in, outputs=[outputs, w])


#--------------------------------------- model 4
layer_kDNS = layer_wlatent_mLES()

z_in         = tf.keras.Input(shape=([G_LAYERS+1, LATENT_SIZE]), dtype=DTYPE)
w0           = mapping(z_in[:,0,:], training=False)
w            = layer_kDNS(w0, z_in[:,1:G_LAYERS+1,:])
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=z_in, outputs=[outputs, w])





# create filter model
if (USE_GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = gaussian_filter(x_in[0,0,:,:], rs=1, rsca=int(2**FIL))
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



# restart from defined values
if (RESTART_WL):

    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    filename = Z0_DIR_WL + "z0.npz"
                
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
        noise_DNS  = tf.convert_to_tensor(noise_DNS,  dtype=DTYPE)
        it=0
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                print(layer.trainable_variables)
                layer.trainable_variables[:].assign(noise_DNS[it])
                it=it+1

else:             

    # set z
    #--------------------------------------- model 1
    # z0min = tf.constant(MINVALRAN, shape=[BATCH_SIZE, M_LAYERS, LATENT_SIZE], dtype=DTYPE)
    # z0max = tf.constant(MAXVALRAN, shape=[BATCH_SIZE, M_LAYERS, LATENT_SIZE], dtype=DTYPE)
    # z0med = tf.random.normal(shape=[BATCH_SIZE, 1, LATENT_SIZE], mean=0.0, stddev=1.0, dtype=DTYPE)
    # z0med = tf.clip_by_value(z0med, clip_value_min=MINVALRAN, clip_value_max=MAXVALRAN)
    # z0    = tf.concat([z0min, z0max, z0med], axis=1)

    #--------------------------------------- model 2
    # z0 = tf.random.uniform(shape=[BATCH_SIZE, NC2_NOISE,1], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)

    #--------------------------------------- model 3
    # z0 = tf.random.uniform(shape=[BATCH_SIZE, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)

    #--------------------------------------- model 4
    zn = tf.random.uniform(shape=[BATCH_SIZE, 1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
    zw = tf.random.uniform(shape=[BATCH_SIZE, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
    wn = mapping(zw, training=False)
    z0 = tf.concat([zn, wn], axis=1)


# find inference
UVP_DNS, UVP_LES, fUVP_DNS, wno, _ = find_predictions(wl_synthesis, gfilter, z0)
resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS, UVP_LES, typeRes=0)
print("Initial residuals:  resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
        .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))

imgA  = tf.identity(UVP_DNS)
fimgA = tf.identity(UVP_LES)

# tune to given tollerance
if (TUNE):

    it     = 0
    tstart = time.time()
    while (resREC>tollDNS and it<lr_LES_maxIt):

        lr = lr_schedule_DNS(it)
        UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, wn = \
            step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, z0, imgA, fimgA, ltv_DNS, typeRes=0)

        valid_wn = True
        for nvars in range(len(layer_kDNS.trainable_variables[:])):
            kDNS  = layer_kDNS.trainable_variables[nvars]
            kDNSn = tf.clip_by_value(kDNS, -1, 2)
            if (tf.reduce_any((kDNS-kDNSn)>0)):
                print("reset values")
                kDNSn = tf.zeros_like(kDNSn)
                zn = tf.random.uniform(shape=[BATCH_SIZE, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
                z0 = tf.concat([zn[:,tf.newaxis,:], wno], axis=1)
                layer_kDNS.trainable_variables[nvars].assign(kDNSn)
                UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, z0)
                resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=0)        
        
        if (valid_wn):
            wno = tf.identity(wn)
 
        # print fields
        if (it%10==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   it {1:6d}  resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} lr {6:3e}" \
                .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy(), lr))

            if (it%1000==0):
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

    # load z
    z0 = z0.numpy()

    # load parameters
    kDNS = []
    for nvars in range(len(layer_kDNS.trainable_variables[:])):
        kDNS.append(layer_kDNS.trainable_variables[nvars].numpy())

    # load noise
    if (TUNE_NOISE):
        it=0
        noise_DNS=[]
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                noise_DNS.append(layer.trainable_variables[:].numpy())
    else:
        noise_DNS=[]

    filename =  Z0_DIR_WL + "z0.npz"

    np.savez(filename,
            z0=z0, \
            kDNS=kDNS, \
            noise_DNS=noise_DNS)


# find DNS, LES and filtered fields
if (TESTCASE=='HW' or TESTCASE=='mHW'):
    UVP_DNS = UVP_DNS*INIT_SCAL

    print("min/max/average U ", tf.reduce_mean(UVP_DNS[0, 0, :, :]))
    print("min/max/average V ", tf.reduce_mean(UVP_DNS[0, 1, :, :]))
    print("min/max/average P ", tf.reduce_mean(UVP_DNS[0, 2, :, :]))

U_DNS = UVP_DNS[0, 0, :, :]
V_DNS = UVP_DNS[0, 1, :, :]
P_DNS = UVP_DNS[0, 2, :, :]

U_DNS = U_DNS[tf.newaxis, tf.newaxis, :, :]
V_DNS = V_DNS[tf.newaxis, tf.newaxis, :, :]
P_DNS = P_DNS[tf.newaxis, tf.newaxis, :, :]

fU_DNS = gfilter(U_DNS, training = False)
fV_DNS = gfilter(V_DNS, training = False)
fP_DNS = gfilter(P_DNS, training = False)

UVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], 1)
fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)

U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
P_DNS = UVP_DNS[0, 2, :, :].numpy()

U_LES = fUVP_DNS[0, 0, :, :].numpy()  # note as we take here the LES as the filtered fields, not the internal of the GAN!
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
    print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename, testcase=TESTCASE)

    filename = Z0_DIR_WL + "plots_LES_restart.png"
    print_fields_3(U_LES, V_LES, P_LES, N=N_LES, filename=filename, testcase=TESTCASE)

    filename = Z0_DIR_WL + "plots_fDNS_restart.png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, N=N_LES, filename=filename, testcase=TESTCASE)

    cP_DNS = (-tr(V_DNS, 2, 0) + 16*tr(V_DNS, 1, 0) - 30*V_DNS + 16*tr(V_DNS,-1, 0) - tr(V_DNS,-2, 0))/(12*DELX**2) \
           + (-tr(V_DNS, 0, 2) + 16*tr(V_DNS, 0, 1) - 30*V_DNS + 16*tr(V_DNS, 0,-1) - tr(V_DNS, 0,-2))/(12*DELY**2)
    filename = Z0_DIR_WL + "plots_diffPhi.png"
    print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, N=N_DNS, filename=filename, testcase=TESTCASE, diff=True)


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
