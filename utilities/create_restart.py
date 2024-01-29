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
tollDNS     = 1.0e+0
N_DNS       = 2**RES_LOG2
N_LES       = 2**(RES_LOG2-FIL)
N2L         = int(N_LES/2)
RS          = int(2**FIL)
RESTART_WL  = False


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

DELX     = L/N_DNS
DELY     = L/N_DNS
DELX_LES = L/N_LES
DELY_LES = L/N_LES



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

train_summary_writer = tf.summary.create_file_writer(Z0_DIR_WL)


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
    z0 = tf.random.uniform(shape=[BATCH_SIZE, 2*M_LAYERS+1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)



#-------------- load DNS field
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

imgA = tf.concat([U_DNS_org, V_DNS_org, P_DNS_org], axis=1)

filename = Z0_DIR_WL + "plots_DNS_orig_restart.png"
print_fields_3(imgA[0,0,:,:], imgA[0,1,:,:], imgA[0,2,:,:], filename=filename, testcase=TESTCASE)


# find filtered field
fU = gfilter(U_DNS_org)[0,0,:,:]
fV = gfilter(V_DNS_org)[0,0,:,:]
fP = gfilter(P_DNS_org)[0,0,:,:]

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

fimgA = tf.concat([fU[tf.newaxis,tf.newaxis,:,:], fV[tf.newaxis,tf.newaxis,:,:], fP[tf.newaxis,tf.newaxis,:,:]], axis=1)

nfU = fU/fU_amax
nfV = fV/fV_amax
nfP = fP/fP_amax

nfimgA = tf.concat([nfU[tf.newaxis,tf.newaxis,:,:], nfV[tf.newaxis,tf.newaxis,:,:], nfP[tf.newaxis,tf.newaxis,:,:]], axis=1)

filename = Z0_DIR_WL + "plots_fDNSn.png"
print_fields_3(nfU, nfV, nfP, filename=filename, testcase=TESTCASE)
    


#--------------  find multiplier for DNS field
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

filename = Z0_DIR_WL + "plots_nDNSf.png"
print_fields_3(fnU, fnV, fnP, filename=filename, testcase=TESTCASE)

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
print("UVP_max are :", UVP_max[0].numpy(), UVP_max[1].numpy(), UVP_max[2].numpy())



# find inference
UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, z0, UVP_max)
resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=1)
print("\nInitial residuals ------------------------:     resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
        .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))

# tune to given tollerance
if (TUNE):

    it     = 0
    kDNSo = layer_kDNS.trainable_variables[0]
    tstart = time.time()
    while (resREC>tollDNS and it<lr_DNS_maxIt):

        lr = lr_schedule_DNS(it)
        UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, _, _ = \
            step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, z0, imgA, fimgA, ltv_DNS, UVP_max, typeRes=1)

        # adjust variables
        kDNS = layer_kDNS.trainable_variables[1]
        kDNS = tf.clip_by_value(kDNS, 0.0, 1.0)
        layer_kDNS.trainable_variables[1].assign(kDNS)

        # valid_zn = True
        # kDNS  = layer_kDNS.trainable_variables[0]
        # kDNSc = tf.clip_by_value(kDNS, 0.0, 1.0)
        # if (tf.reduce_any((kDNS-kDNSc)>0) or (it%10000==0 and it!=0)):
        #     print("reset values")
        #     z0p = tf.random.uniform(shape=[BATCH_SIZE, M_LAYERS, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
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

        # write losses to tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('resREC',   resREC,   step=it)
            tf.summary.scalar('resLES',   resLES,   step=it)
            tf.summary.scalar('resDNS',   resDNS,   step=it)
            tf.summary.scalar('loss_fil', loss_fil, step=it)
            tf.summary.scalar('lr',       lr,       step=it)

        # print fields
        if (it%100==0):
            tend = time.time()
            print("LES iterations:  time {0:3e}   it {1:6d}  resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} lr {6:3e}" \
                .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy(), lr))

            if (it%1000==0):
                U_DNS = UVP_DNS[0, 0, :, :].numpy()
                V_DNS = UVP_DNS[0, 1, :, :].numpy()
                P_DNS = UVP_DNS[0, 2, :, :].numpy()
    
                filename =  Z0_DIR_WL + "plots_restart_" + str(0).zfill(4) + ".png"
                print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename)

                filename = Z0_DIR_WL + "plots_DNSdiff_restart_" + str(0).zfill(4) + ".png"
                print_fields_3(P_DNS_org[0,0,:,:], P_DNS, P_DNS_org[0,0,:,:]-P_DNS, filename=filename, testcase=TESTCASE, diff=True, \
                    Umin=-nU_amax, Umax=nU_amax, Vmin=-nV_amax, Vmax=nV_amax, Pmin=-nP_amax, Pmax=nP_amax)

                filename = Z0_DIR_WL + "plots_LESdiff_restart_" + str(0).zfill(4) + ".png"
                print_fields_3(fimgA[0,2,:,:], fUVP_DNS[0,2,:,:], fimgA[0,2,:,:]-fUVP_DNS[0,2,:,:], \
                    filename=filename, testcase=TESTCASE, diff=True, \
                    Umin=-fU_amax, Umax=fU_amax, Vmin=-fV_amax, Vmax=fV_amax, Pmin=-fP_amax, Pmax=fP_amax)
    
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


# check average fields
if (TESTCASE=='HW' or TESTCASE=='mHW'):
    print("min/max/average U ", tf.reduce_mean(UVP_DNS[0, 0, :, :]))
    print("min/max/average V ", tf.reduce_mean(UVP_DNS[0, 1, :, :]))
    print("min/max/average P ", tf.reduce_mean(UVP_DNS[0, 2, :, :]))


#--------------------- find DNS, LES and filtered fields
# DNS
U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
P_DNS = UVP_DNS[0, 2, :, :].numpy()

# filtered
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
    
    filename = Z0_DIR_WL + "plots_DNSfromGAN_restart.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE)

    filename = Z0_DIR_WL + "plots_fDNSfromGAN_restart.png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, filename=filename, testcase=TESTCASE)

    filename = Z0_DIR_WL + "plots_DNS_diff_DNSfromGAN_restart.png"
    print_fields_3(P_DNS_org[0,0,:,:], P_DNS, P_DNS_org[0,0,:,:]-P_DNS, filename=filename, testcase=TESTCASE, diff=True, \
                   Umin=-nU_amax, Umax=nU_amax, Vmin=-nV_amax, Vmax=nV_amax, Pmin=-nP_amax, Pmax=nP_amax)

    filename = Z0_DIR_WL + "plots_fDNS_diff_fDNSfromGAN_restart.png"
    print_fields_3(fimgA[0,2,:,:], fP_DNS, fimgA[0,2,:,:]-fP_DNS, filename=filename, testcase=TESTCASE, diff=True, \
                   Umin=-fU_amax, Umax=fU_amax, Vmin=-fV_amax, Vmax=fV_amax, Pmin=-fP_amax, Pmax=fP_amax)

    cP_DNS = (-tr(V_DNS, 2, 0) + 16*tr(V_DNS, 1, 0) - 30*V_DNS + 16*tr(V_DNS,-1, 0) - tr(V_DNS,-2, 0))/(12*DELX**2) \
           + (-tr(V_DNS, 0, 2) + 16*tr(V_DNS, 0, 1) - 30*V_DNS + 16*tr(V_DNS, 0,-1) - tr(V_DNS, 0,-2))/(12*DELY**2)
    filename = Z0_DIR_WL + "plots_diffPhi.png"
    print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, filename=filename, testcase=TESTCASE, diff=True)


    filename = Z0_DIR_WL + "energy_spectrum_restart.png"
    closePlot=False
    gradV_DNS = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*DELX))**2 \
                      + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*DELY))**2)
    plot_spectrum(U_DNS, gradV_DNS, L, filename, close=closePlot)

    filename = Z0_DIR_WL + "energy_spectrum_fDNS_restart.png"
    closePlot=True
    gradV_fDNS = np.sqrt(((cr(fV_DNS, 1, 0) - cr(fV_DNS, -1, 0))/(2.0*DELX_LES))**2 \
                       + ((cr(fV_DNS, 0, 1) - cr(fV_DNS, 0, -1))/(2.0*DELY_LES))**2)
    plot_spectrum(fU_DNS, gradV_fDNS, L, filename, close=closePlot)
