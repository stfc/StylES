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
import glob
import imageio

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from LES_constants import *
from LES_parameters import *
from LES_plot import *
from MSG_StyleGAN_tf2 import *

tf.random.set_seed(SEED_RESTART+2)


#------------------------------------------------------ parameters
TUNE        = False
TUNE_NOISE  = True
tollDNS     = 1.0e-2
RESTART_WL  = False

# check that randomization is off
noise_DNS=[]
for layer in synthesis.layers:
    if "layer_noise_constants" in layer.name:
        if len(layer.trainable_variables[:]) == 0:
            print("Carefull! Noise randomization is on!! Swith it to off in ../parameters.py")
            exit()

# set folders and paths
if (TESTCASE=='HIT_2D'):
    from HIT_2D import L
    os.system("mkdir -p ../LES_Solvers/restart_fromGAN/")
    Z0_DIR_WL = "../LES_Solvers/restart_fromGAN/"
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = 50.176
    os.system("mkdir -p ../bout_interfaces/restart_fromGAN/org/")
    Z0_DIR_WL     = "../bout_interfaces/restart_fromGAN/"
    Z0_DIR_WL_ORG = "../bout_interfaces/restart_fromGAN/org/"

if (not RESTART_WL):
    os.system("rm -rf " + Z0_DIR_WL)
    os.system("mkdir -p " + Z0_DIR_WL)
CHKP_DIR_WL = Z0_DIR_WL + "checkpoints_wl/"



#------------------------------------------------------ define optimizer for z and w search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_DNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_DNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_kDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_DNS)

train_summary_writer = tf.summary.create_file_writer(Z0_DIR_WL_ORG + "logs/")


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)


# create fixed synthesis model
flayer_kDNS   = layer_zlatent_kDNS()

fz_in         = tf.keras.Input(shape=([1+2*(G_LAYERS-M_LAYERS), LATENT_SIZE]),  dtype=DTYPE)
fw            = flayer_kDNS(mapping, fz_in)
fpre_w        = pre_synthesis(fw)
foutputs      = synthesis([fw, fpre_w], training=False)
fwl_synthesis = tf.keras.Model(inputs=fz_in, outputs=[foutputs, fw])


# create variable synthesis model
layer_kDNS   = layer_zlatent_kDNS()

z_in   = tf.keras.Input(shape=([1+2*(G_LAYERS-M_LAYERS), LATENT_SIZE]),  dtype=DTYPE)
img_in = []
for res in range(2,RES_LOG2-FIL+1):
    img_in.append(tf.keras.Input(shape=([NUM_CHANNELS, 2**res, 2**res]), dtype=DTYPE))
w            = layer_kDNS(mapping, z_in)
outputs      = synthesis([w, img_in], training=False)
wl_synthesis = tf.keras.Model(inputs=[z_in, img_in], outputs=[outputs, w])


# create filter model
if (GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = gaussian_filter(x_in[0,0,:,:], rs=RS2, rsca=RS)
    gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    x_in              = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out               = gaussian_filter(x_in[0,0,:,:], rs=RS2, rsca=1)
    gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)
else:
    gfilter = filters[IFIL]


# define checkpoints wl_synthesis and filter
checkpoint_wl        = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    
# for variable in layer_kDNS.trainable_variables:
#     ltv_DNS.append(variable)

print("\n kDNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)


# set initial scaling coefficients
UVP_max = [INIT_SCA, INIT_SCA, INIT_SCA]

time.sleep(3)



#------------------------------------------------------ restart from defined values
if (RESTART_WL):

    # loading wl_synthesis checkpoint
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    filename = Z0_DIR_WL + "z0.npz"
                
    data = np.load(filename)

    z0      = data["z0"]
    kDNS    = data["kDNS"]
    LES_in0 = data["LES_in0"]

    print("z0",      z0.shape,      np.min(z0),      np.max(z0))
    print("kDNS",    kDNS.shape,    np.min(kDNS),    np.max(kDNS))
    print("LES_in0", LES_in0.shape, np.min(LES_in0), np.max(LES_in0))

    # assign variables
    z0 = tf.convert_to_tensor(z0, dtype=DTYPE)

    for nvars in range(len(kDNS)):
        tkDNS = tf.convert_to_tensor(kDNS[nvars], dtype=DTYPE)
        layer_kDNS.trainable_variables[nvars].assign(tkDNS)

    LES_all = []
    for res in range(2,RES_LOG2-FIL+1):
        rs = 2**(RES_LOG2-FIL-res)
        LES_all.append(tf.convert_to_tensor(LES_in0[:,:,::rs,::rs], dtype=DTYPE))

    # assign variable noise
    if (TUNE_NOISE):
        noise_DNS = data["noise_DNS"]
        print("noise_DNS", noise_DNS.shape, np.min(noise_DNS), np.max(noise_DNS))
        noise_DNS = tf.convert_to_tensor(noise_DNS,  dtype=DTYPE)
        it=0
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                layer.trainable_variables[0].assign(noise_DNS[it])
                it=it+1

else:

    # set z
    z0i = tf.random.uniform(shape=[BATCH_SIZE,                 1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
    z0p = tf.random.uniform(shape=[BATCH_SIZE, G_LAYERS-M_LAYERS, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
    z0m = -z0p
    z0  = tf.concat([z0i, z0p, z0m], axis=1)

    UVP_DNS, UVP_LES, fUVP_DNS, _, predictions = find_predictions(fwl_synthesis, gfilter, z0, UVP_max)

    LES_all = []
    for res in range(2,RES_LOG2-FIL+1):
        LES_all.append(predictions[res-2])

print ("============================Completed setup!\n\n")



#------------------------------------------------------ find DNS field target
# find inference
UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)
resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS, UVP_LES, typeRes=0)
print("\nInitial residuals ------------------------:     resREC {0:3e} resLES {1:3e}  resDNS {2:3e} loss_fil {3:3e} " \
        .format(resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))

imgA   = tf.identity(UVP_DNS)
fimgA  = tf.identity(UVP_LES)



#------------------------------------------------------ tune to given tollerance
if (TUNE):

    it     = 0
    kDNSo = layer_kDNS.trainable_variables[0]
    tstart = time.time()
    while (resREC>tollDNS and it<lr_DNS_maxIt):

        lr = lr_schedule_DNS(it)
        UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, _, _ = \
            step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, [z0, LES_all], imgA, fimgA, ltv_DNS, UVP_max, typeRes=0)


        # adjust variables
        kDNS  = layer_kDNS.trainable_variables[0]
        kDNSn = tf.clip_by_value(kDNS, 0.0, 1.0)
        if (tf.reduce_any((kDNS-kDNSn)>0)):
            layer_kDNS.trainable_variables[0].assign(kDNSn)
            UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)
            resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=0)


        # # adjust variables
        # valid_z0 = True
        # kDNS     = layer_kDNS.trainable_variables[0]
        # kDNSt = tf.clip_by_value(kDNS, 0.0, 1.0)
        # if (tf.reduce_any((kDNS-kDNSt)>0)):  # find new left and right z0
        #     if (valid_z0):
        #         print("reset z at iteration", it)
        #         valid_z0 = False
        #     z0o = kDNSo*z0p + (1.0-kDNSo)*z0m
        #     z0n = tf.random.uniform(shape=[BATCH_SIZE, G_LAYERS-M_LAYERS, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
        #     z0p = (z0o+z0n)/2.0
        #     z0m = (z0o-z0n)/2.0
        #     z0  = tf.concat([z0i, z0p, z0m], axis=1)
        #     kDNSn = 0.5*tf.ones_like(kDNS)
        #     layer_kDNS.trainable_variables[0].assign(kDNSn)
        #     UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_all], UVP_max)
        #     resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=0)

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
    
                filename = Z0_DIR_WL + "plots_diff_fDNS_LES_" + str(it).zfill(5) + ".png"
                print_fields_3(fimgA[0,2,:,:], fUVP_DNS[0,2,:,:], fimgA[0,2,:,:]-fUVP_DNS[0,2,:,:],
                    filename=filename, testcase=TESTCASE, diff=True) #,\
                    # Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

                filename = Z0_DIR_WL + "plots_DNS_" + str(it).zfill(5) + ".png"
                print_fields_3(UVP_DNS[0,0,:,:], UVP_DNS[0,1,:,:], UVP_DNS[0,2,:,:],
                    filename=filename, testcase=TESTCASE) #, \
                    # Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)
    
        it = it+1

    # print final iteration
    tend = time.time()
    print("LES iterations:  time {0:3e}   it {1:6d}  resREC {2:3e} resLES {3:3e}  resDNS {4:3e} loss_fil {5:3e} " \
        .format(tend-tstart, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil.numpy()))
            


#------------------------------------------------------ save NN configuration
if (not RESTART_WL):
    managerCheckpoint_wl.save()

    # save z
    z0 = z0.numpy()

    # save parameters
    kDNS = []
    for nvars in range(len(layer_kDNS.trainable_variables[:])):
        kDNS.append(layer_kDNS.trainable_variables[nvars].numpy())

    # load noise
    if (TUNE_NOISE):
        it=0
        noise_DNS=[]
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                noise_DNS.append(layer.trainable_variables[0].numpy())

        filename =  Z0_DIR_WL + "z0.npz"
        np.savez(filename,
                z0        = z0, \
                kDNS      = kDNS, \
                LES_in0   = LES_all[-1], \
                noise_DNS = noise_DNS)
    else:
        filename =  Z0_DIR_WL + "z0.npz"
        np.savez(filename,
                z0      = z0, \
                kDNS    = kDNS, \
                LES_in0 = LES_all[-1])


#------------------------------------------------------ check, find and print fields
if (TESTCASE=='HW' or TESTCASE=='mHW'):
    print("Mean U ", tf.reduce_mean(UVP_DNS[0, 0, :, :]))
    print("Mean V ", tf.reduce_mean(UVP_DNS[0, 1, :, :]))
    print("Mean P ", tf.reduce_mean(UVP_DNS[0, 2, :, :]))


#--------------------------- find DNS, LES and filtered fields
# DNS
U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
P_DNS = UVP_DNS[0, 2, :, :].numpy()

# DNS
U_LES = UVP_LES[0, 0, :, :].numpy()
V_LES = UVP_LES[0, 1, :, :].numpy()
P_LES = UVP_LES[0, 2, :, :].numpy()

# filtered
fU_DNS = fUVP_DNS[0, 0, :, :].numpy()
fV_DNS = fUVP_DNS[0, 1, :, :].numpy()
fP_DNS = fUVP_DNS[0, 2, :, :].numpy()


#--------------------------- print final fields, differences and spectra
if (TESTCASE=='HIT_2D'):

    filename = Z0_DIR_WL + "plots.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, N=OUTPUT_DIM, filename=filename, testcase=TESTCASE)

    filename = Z0_DIR_WL + "restart"
    save_fields(0.6, U_DNS, V_DNS, P_DNS, filename=filename)  # Note: t=0.6 is the corrisponding time to t=545 tau_e

    filename = Z0_DIR_WL + "energy_spectrum.png"
    closePlot=True
    plot_spectrum(U_DNS, V_DNS, L, filename, close=closePlot)

elif(TESTCASE=='HW' or TESTCASE=='mHW'):

    # fields
    filename = Z0_DIR_WL + "plots_DNS.png"
    print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE, \
                Umin=-2*INIT_SCA, Umax=2*INIT_SCA, Vmin=-2*INIT_SCA, Vmax=2*INIT_SCA, Pmin=-2*INIT_SCA, Pmax=2*INIT_SCA)

    filename = Z0_DIR_WL + "plots_LES.png"
    print_fields_3(U_LES, V_LES, P_LES, filename=filename, testcase=TESTCASE, \
                Umin=-2*INIT_SCA, Umax=2*INIT_SCA, Vmin=-2*INIT_SCA, Vmax=2*INIT_SCA, Pmin=-2*INIT_SCA, Pmax=2*INIT_SCA)

    filename = Z0_DIR_WL + "plots_fDNS.png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, filename=filename, testcase=TESTCASE, \
                Umin=-2*INIT_SCA, Umax=2*INIT_SCA, Vmin=-2*INIT_SCA, Vmax=2*INIT_SCA, Pmin=-2*INIT_SCA, Pmax=2*INIT_SCA)


    # differences
    filename = Z0_DIR_WL + "plots_diff_fDNS_LES.png"
    print_fields_3(P_LES, fP_DNS, P_LES-fP_DNS, filename=filename, testcase=TESTCASE, diff=True) #, \
                #Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

    cP_DNS = (-tr(V_DNS, 2, 0) + 16*tr(V_DNS, 1, 0) - 30*V_DNS + 16*tr(V_DNS,-1, 0) - tr(V_DNS,-2, 0))/(12*DELX**2) \
           + (-tr(V_DNS, 0, 2) + 16*tr(V_DNS, 0, 1) - 30*V_DNS + 16*tr(V_DNS, 0,-1) - tr(V_DNS, 0,-2))/(12*DELY**2)
    filename = Z0_DIR_WL + "plots_diff_Phi.png"
    print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, filename=filename, testcase=TESTCASE, diff=True)


    # spectrum
    closePlot=False
    filename = Z0_DIR_WL + "energy_spectrum_DNS.png"
    gradV = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*DELX))**2 \
                  + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*DELY))**2)
    plot_spectrum(U_DNS, gradV, L, filename, label="DNS", close=closePlot)

    filename = Z0_DIR_WL + "energy_spectrum_LES.png"
    gradV = np.sqrt(((cr(V_LES, 1, 0) - cr(V_LES, -1, 0))/(2.0*DELX_LES))**2 \
                  + ((cr(V_LES, 0, 1) - cr(V_LES, 0, -1))/(2.0*DELY_LES))**2)
    plot_spectrum(U_LES, gradV, L, filename, label="LES", close=closePlot)

    closePlot=True
    filename = Z0_DIR_WL + "energy_spectrum_fDNS.png"
    gradV = np.sqrt(((cr(fV_DNS, 1, 0) - cr(fV_DNS, -1, 0))/(2.0*DELX_LES))**2 \
                  + ((cr(fV_DNS, 0, 1) - cr(fV_DNS, 0, -1))/(2.0*DELY_LES))**2)
    plot_spectrum(fU_DNS, gradV, L, filename, label="fDNS", close=closePlot)


print ("============================Completed tuning!\n\n")






#--------------------------- verify filter properties

# find
cf_DNS = 10.0*U_DNS  # conservation
lf_DNS = U_DNS + V_DNS  # linearity
df_DNS = ((tr(P_DNS, 1, 0) - tr(P_DNS,-1, 0))/(2*DELX)) + ((tr(P_DNS, 0, 1) - tr(P_DNS, 0,-1))/(2*DELY))  # commutative
        
cf_DNS = (gfilter(cf_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
lf_DNS = (gfilter(lf_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
df_DNS = (gfilter(df_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]

c_LES = 10.0*fU_DNS
l_LES = fU_DNS + fV_DNS
if (GAUSSIAN_FILTER):
    fP_DNS_noSca = (gfilter_noScaling(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]   # the downscaling must happens after the filtering!!
    d_LES = ((tr(fP_DNS_noSca, 1, 0) - tr(fP_DNS_noSca,-1, 0))/(2*DELX)) + ((tr(fP_DNS_noSca, 0, 1) - tr(fP_DNS_noSca, 0,-1))/(2*DELY))
    d_LES = d_LES[::RS,::RS]
else:
    d_LES = ((tr(fP_DNS, 1, 0) - tr(fP_DNS,-1, 0))/(2*DELX_LES)) + ((tr(fP_DNS, 0, 1) - tr(fP_DNS, 0,-1))/(2*DELY_LES))

# plot
filename = Z0_DIR_WL + "plots_filterProperty_conservation.png"
print("Mean error on conservation: ", tf.reduce_mean(cf_DNS-c_LES).numpy())
print_fields_3(cf_DNS, c_LES, cf_DNS-c_LES, N=res, filename=filename, testcase=TESTCASE, diff=True)

filename = Z0_DIR_WL + "plots_filterProperty_linearity.png"
print("Mean error on linearity: ", tf.reduce_mean(lf_DNS-l_LES).numpy())
print_fields_3(lf_DNS, l_LES, lf_DNS-l_LES, N=res, filename=filename, testcase=TESTCASE, diff=True)

filename = Z0_DIR_WL + "plots_filterProperty_derivative.png"
print("Mean error on derivative: ", tf.reduce_mean(df_DNS-d_LES).numpy())
print_fields_3(df_DNS, d_LES, df_DNS-d_LES, N=res, filename=filename, testcase=TESTCASE, diff=True)

print ("============================Completed filter properties check!\n\n")



print ("Completed all tasks successfully!!")




# ------------- extra pieces

# ltv_gauss = []
# for variable in lgauss.trainable_variables:
#     ltv_gauss.append(variable)

# print("\n filter variables:")
# for variable in ltv_gauss:
#     print(variable.name, variable.shape)




    # #------ Sharp (spectral)
    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # out     = sharp_filter(x_in[0,0,:,:], delta=L/N_LES, size=4, rsca=RS)  # delta = pi/Kc, where Kc is the LES wave number (N_LES/2).
    # gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # out     = sharp_filter(x_in[0,0,:,:], delta=L/N_LES, size=4, rsca=1)  # delta = pi/Kc, where Kc is the LES wave number (N_LES/2).
    # gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)


    # #------ Downscaling
    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # out     = x_in[:,:,::RS,::RS]
    # gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # out     = x_in[:,:,::1,::1]
    # gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)

    # lgauss =  layer_gaussian(rs=RS, rsca=RS)

    # #------ Gaussian normalized
    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # x_max   = tf.abs(tf.reduce_max(x_in))
    # x_min   = tf.abs(tf.reduce_min(x_in))
    # xamax   = tf.maximum(x_max, x_min)
    # x       = x_in/xamax
    # x       = gaussian_filter(x[0,0,:,:], rs=RS, rsca=RS)
    # x_max   = tf.abs(tf.reduce_max(x))
    # x_min   = tf.abs(tf.reduce_min(x))
    # xamaxn  = tf.maximum(x_max, x_min)
    # out     = x/xamaxn * xamax
    # gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    # x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    # x_max   = tf.abs(tf.reduce_max(x_in))
    # x_min   = tf.abs(tf.reduce_min(x_in))
    # xamax   = tf.maximum(x_max, x_min)
    # x       = x_in/xamax
    # x       = gaussian_filter(x[0,0,:,:], rs=RS, rsca=1)
    # x_max   = tf.abs(tf.reduce_max(x))
    # x_min   = tf.abs(tf.reduce_min(x))
    # xamaxn  = tf.maximum(x_max, x_min)
    # out     = x/xamaxn * xamax
    # gfilter_noScaling = tf.keras.Model(inputs=x_in, outputs=out)



        # loss_fill = step_find_gaussianfilter(gfilter, opt_kDNS, UVP_DNS, UVP_LES, ltv_gauss)
        # print(loss_fill)
