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
tollLES     = 1.e-3
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2-FIL
zero_DNS    = np.zeros([N_DNS,N_DNS], dtype=DTYPE)
RESTART_WL  = False
Z0_DIR_WL   = "../bout_interfaces/restart_fromGAN/"
CHKP_DIR_WL = Z0_DIR_WL + "checkpoints_wl/"
CHKP_DIR_KL = Z0_DIR_WL + "checkpoints_kl/"
maxitLES    = 10
maxitDNS    = 100

if (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = 50.176

DELX  = L/N_DNS
DELY  = L/N_DNS

firstNewz = True

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



# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)


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



# restart from defined values
if (RESTART_WL):

    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    if managerCheckpoint_kl.latest_checkpoint:
        print("kl_synthesis restored from {}".format(managerCheckpoint_kl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing kl_synthesis from scratch.")

    if (TESTCASE=='HIT_2D'):
        filename = "results_latentSpace/z0.npz"
    elif(TESTCASE=='HW' or TESTCASE=='mHW'):
        filename = Z0_DIR_WL + "z0_fromBOUT.npz"
                
    data = np.load(filename)

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

    # assign kDNS
    layer_LES_coarse.trainable_variables[0].assign(LES_coarse)
    layer_LES_medium.trainable_variables[0].assign(LES_medium)
    layer_LES_finest.trainable_variables[0].assign(LES_finest)

    # assign variable noise
    it=0
    for layer in wl_synthesis.layers:
        if "layer_noise_constants" in layer.name:
            print(layer.trainable_variables)
            layer.trainable_variables[:].assign(noise_DNS[it])
            it=it+1

else:             

    # set z
    z0 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
    w0 = mapping(z0, training=False)
    z0 = z0[:,tf.newaxis,:]
    w0 = w0[:,tf.newaxis,:,:]
    for nl in range(1, NWTOT):
        z  = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
        w  = mapping(z, training=False)
        z  = z[:,tf.newaxis,:]
        w  = w[:,tf.newaxis,:,:]
        z0 = tf.concat([z0,z], axis=1)
        w0 = tf.concat([w0,w], axis=1)


# find inference
resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_latents_LES_restart_A(wl_synthesis, filter, w0, INIT_SCAL)
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

        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
            step_find_latents_LES_restart_B_coarse(wl_synthesis, filter, opt_LES_coarse, w0, ltv_LES_coarse, INIT_SCAL)

        LESc = layer_LES_coarse.trainable_variables[0]
        LESc = tf.clip_by_value(LESc, 0.0, 1.0)
        layer_LES_coarse.trainable_variables[0].assign(LESc)

        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
            step_find_latents_LES_restart_B_medium(wl_synthesis, filter, opt_LES_medium, w0, ltv_LES_medium, INIT_SCAL)

        LESm = layer_LES_medium.trainable_variables[0]
        LESm = tf.clip_by_value(LESm, 0.0, 1.0)
        layer_LES_medium.trainable_variables[0].assign(LESm)

        resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
            step_find_latents_LES_restart_B_finest(wl_synthesis, filter, opt_LES_finest, w0, ltv_LES_finest, INIT_SCAL)

        LESf = layer_LES_finest.trainable_variables[0]
        LESf = tf.clip_by_value(LESf, 0.0, 1.0)
        layer_LES_finest.trainable_variables[0].assign(LESf)
        

        # change next z0
        if (it%maxitLES==0 and it!=0):
            print("change next z0")
            
            # find inference with correct LESm variables
            resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_latents_LES_restart_A(wl_synthesis, filter, w0, INIT_SCAL)
            print(resREC.numpy())
             
            # loop over target DNS image
            kDNS = tf.fill((NWTOT, LATENT_SIZE), 1.0)
            layer_kDNS.trainable_variables[0].assign(kDNS)

            if (firstNewz):
                z1p = tf.identity(z0)
                z1o = tf.identity(z0)
                firstNewz = False
            else:
                z1p = (z1 - z1o) + z1
                z1o = tf.identity(z1)

            itDNS = 0
            resDNS, wn, UVP_DNSn = step_find_kDNS_noGradients(kl_synthesis, filter, opt_kDNS, z1p, UVP_DNS, ltv_DNS, INIT_SCAL)

            while (resDNS>tollDNS and itDNS<maxitDNS):
                resDNS, wn, UVP_DNSn = step_find_kDNS(kl_synthesis, filter, opt_kDNS, z1p, UVP_DNS, ltv_DNS, INIT_SCAL)
                z1p = z1p*layer_kDNS.trainable_variables[0]
                z1p = tf.clip_by_value(z1p, -1.0, 1.0)
                itDNS = itDNS+1

            # print final residuals
            print("Residuals DNS search ", resDNS.numpy())

            # save old value
            z1 = tf.identity(z1p)

            # find latest corrected inference
            resDNS, wn, UVP_DNSn = step_find_kDNS_noGradients(kl_synthesis, filter, opt_kDNS, z1, UVP_DNS, ltv_DNS, INIT_SCAL)


            # save fields from DNS search
            filename = "../bout_interfaces/restart_fromGAN/plots_DNS_search_org_" + str(it).zfill(7) + ".png"
            print_fields_3(UVP_DNS[0,0,:,:].numpy(), UVP_DNS[0,1,:,:].numpy(), UVP_DNS[0,2,:,:].numpy(), filename=filename)

            filename = "../bout_interfaces/restart_fromGAN/plots_DNS_search_" + str(it).zfill(7) + ".png"
            print_fields_3(UVP_DNSn[0,0,:,:].numpy(), UVP_DNSn[0,1,:,:].numpy(), UVP_DNSn[0,2,:,:].numpy(), filename=filename)


            # replace last latent space
            LESc = layer_LES_coarse.trainable_variables[0]
            LESm = layer_LES_medium.trainable_variables[0]
            LESf = layer_LES_finest.trainable_variables[0]

            if (NWTOT>2):

                if (NWTOT==3):
                    ptc = tf.identity(w0[:,0,       0:C_LAYERS,:])
                    ptm = tf.identity(w0[:,0,C_LAYERS:M_LAYERS,:])
                    ptf = tf.identity(w0[:,0,M_LAYERS:G_LAYERS,:])
                tc = LESc[0,:,:]*w0[:,0,       0:C_LAYERS,:] + (1.0 - LESc[0,:,:])*w0[:,1,       0:C_LAYERS,:]
                tm = LESm[0,:,:]*w0[:,0,C_LAYERS:M_LAYERS,:] + (1.0 - LESm[0,:,:])*w0[:,1,C_LAYERS:M_LAYERS,:]
                tf = LESf[0,:,:]*w0[:,0,M_LAYERS:G_LAYERS,:] + (1.0 - LESf[0,:,:])*w0[:,1,M_LAYERS:G_LAYERS,:]
                for nl in range(1,NWTOT-1):
                    tc = LESc[nl,:,:]*tc + (1.0 - LESc[nl,:,:])*w0[:,nl+1,       0:C_LAYERS,:]
                    tm = LESm[nl,:,:]*tm + (1.0 - LESm[nl,:,:])*w0[:,nl+1,C_LAYERS:M_LAYERS,:]
                    tf = LESf[nl,:,:]*tf + (1.0 - LESf[nl,:,:])*w0[:,nl+1,M_LAYERS:G_LAYERS,:]
                    if (NWTOT>3 and nl==NWTOT-3):
                        ptc = tf.identity(tc)
                        ptm = tf.identity(tm)
                        ptf = tf.identity(tf)

                nl = NWTOT-3
                wpc = (tc - ptc*LESc[nl,:,:])/(1-LESc[nl,:,:])
                wpm = (tm - ptm*LESm[nl,:,:])/(1-LESm[nl,:,:])
                wpf = (tf - ptf*LESf[nl,:,:])/(1-LESf[nl,:,:])

                wp  = tf.concat([wpc, wpm, wpf], axis=1)
                wp  = wp[:,tf.newaxis,:,:]
                wn  = wn[:,tf.newaxis,:,:]
                w0  = tf.concat([w0[:,0:nl,:,:], wp, wn], axis=1)

                LESc1 = LESc[0:nl, :, :]
                LESm1 = LESm[0:nl, :, :]
                LESf1 = LESf[0:nl, :, :]
                
                LESc2 = tf.fill((1, C_LAYERS         , LATENT_SIZE), 1.0)
                LESm2 = tf.fill((1, M_LAYERS-C_LAYERS, LATENT_SIZE), 1.0)
                LESf2 = tf.fill((1, G_LAYERS-M_LAYERS, LATENT_SIZE), 1.0)
                
                LESc = tf.concat([LESc1, LESc2], axis=0)
                LESm = tf.concat([LESm1, LESm2], axis=0)
                LESf = tf.concat([LESf1, LESf2], axis=0)

            else:

                nl = 0

                wpc = LESc[nl,:,:]*w0[:,nl,       0:C_LAYERS,:] + (1.0-LESc[nl,:,:])*w0[:,nl+1,       0:C_LAYERS,:]
                wpm = LESm[nl,:,:]*w0[:,nl,C_LAYERS:M_LAYERS,:] + (1.0-LESm[nl,:,:])*w0[:,nl+1,C_LAYERS:M_LAYERS,:]
                wpf = LESf[nl,:,:]*w0[:,nl,M_LAYERS:G_LAYERS,:] + (1.0-LESf[nl,:,:])*w0[:,nl+1,M_LAYERS:G_LAYERS,:]

                wp  = tf.concat([wpc, wpm, wpf], axis=1)

                wp  = wp[:,tf.newaxis,:,:]
                wn  = wn[:,tf.newaxis,:,:]

                w0  = tf.concat([wp, wn], axis=1)
                
                LESc = tf.fill((1, C_LAYERS         , LATENT_SIZE), 1.0)
                LESm = tf.fill((1, M_LAYERS-C_LAYERS, LATENT_SIZE), 1.0)
                LESf = tf.fill((1, G_LAYERS-M_LAYERS, LATENT_SIZE), 1.0)
            
            layer_LES_coarse.trainable_variables[0].assign(LESc)
            layer_LES_medium.trainable_variables[0].assign(LESm)
            layer_LES_finest.trainable_variables[0].assign(LESf)

            # reset counter        
            it = 0
            resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = step_find_latents_LES_restart_A(wl_synthesis, filter, w0, INIT_SCAL)
            print(resREC.numpy())
            exit()
            
        else:
            
            it = it+1

        #if (it%1==0):
        if (it!=0 and (it+1)%100==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   it {1:6d}  residuals {2:3e} ".format(tend-tstart, it, resREC.numpy()))
                
            U_DNS = UVP_DNS[0, 0, :, :].numpy()
            V_DNS = UVP_DNS[0, 1, :, :].numpy()
            P_DNS = UVP_DNS[0, 2, :, :].numpy()            

            if (TESTCASE=='HIT_2D'):
                filename = "results_reconstruction/plots/plots_restart.png"
                # filename = "results_reconstruction/plots/plots_restart_" + str(it) + ".png"
            elif(TESTCASE=='HW' or TESTCASE=='mHW'):
                os.system("mkdir -p ../bout_interfaces/restart_fromGAN/")
                # filename = "../bout_interfaces/restart_fromGAN/plots_DNS_restart.png"
                filename = "../bout_interfaces/restart_fromGAN/plots_DNS_restart_" + str(it).zfill(7) + ".png"
            print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename)
                   

    tend = time.time()
    lr = lr_schedule_LES_finest(it)
    print("LES iterations:  time {0:3e}   it {1:6d}  residuals {2:3e}   lr {3:3e} ".format(tend-tstart, it, resREC.numpy(), lr))



            
# save NN configuration
if (not RESTART_WL):
    managerCheckpoint_wl.save()
    managerCheckpoint_kl.save()

    # find z0, w0
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
        filename = "../bout_interfaces/restart_fromGAN/z0.npz"
    
    np.savez(filename,
            z0=z0, \
            w0=w0, \
            LES_coarse=LES_coarse, \
            LES_medium=LES_medium, \
            LES_finest=LES_finest, \
            noise_DNS=noise_DNS)



# find fields
U_DNS = UVP_DNS[0, 0, :, :]
V_DNS = UVP_DNS[0, 1, :, :]
P_DNS = UVP_DNS[0, 2, :, :]

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
    print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename, testcase=TESTCASE) #, \
                   #Umin=-13.0, Umax=13.0, Vmin=-13.0, Vmax=13.0, Pmin=-13.0, Pmax=13.0)

    filename = "../bout_interfaces/restart_fromGAN/plots_LES_restart.png"
    print_fields_3(U_LES, V_LES, P_LES, N=N_LES, filename=filename, testcase=TESTCASE) #, \
                    #Umin=-13.0, Umax=13.0, Vmin=-13.0, Vmax=13.0, Pmin=-13.0, Pmax=13.0)

    filename = "../bout_interfaces/restart_fromGAN/plots_fDNS_restart.png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, N=N_LES, filename=filename, testcase=TESTCASE) #, \
                    #Umin=-13.0, Umax=13.0, Vmin=-13.0, Vmax=13.0, Pmin=-13.0, Pmax=13.0)

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



