#----------------------------------------------------------------------------------------------
#
#    Copyright (C): 2022 UKRI-STFC (Hartree Centre)
#
#    Author: Jony Castagna, Francesca Schiavello, Josh Williams
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
import matplotlib.pyplot as plt

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')
sys.path.insert(0, '../LES_Solvers/testcases/HIT_2D/')

from matplotlib.ticker import FormatStrFormatter
from LES_constants import *
from LES_parameters import *
from LES_plot import *
from MSG_StyleGAN_tf2 import *




#------------------------------------------- set local parameters
NL          = 1     # number of different latent vectors randomly selected
TUNE_NOISE  = False
LOAD_FIELD  = False       # load field from DNS solver (via restart.npz file)
NITEZ       = 0
RESTART_WL  = True
CHKP_DIR_WL = "./checkpoints_wl"
N_DNS       = 2**RES_LOG2
N_LES       = 2**(RES_LOG2-FIL)
N_LES2         = int(N_LES/2)
tollDNS     = 1.0e-3

if (TESTCASE=='HIT_2D'):
    from HIT_2D import L
    FILE_REAL_PATH = "../LES_Solvers/fields/"
    Z0_DIR_WL      = "../LES_Solvers/restart_fromGAN/"
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    L              = 50.176
    Z0_DIR_WL      = "../bout_interfaces/restart_fromGAN/"
    FILE_REAL_PATH = "../bout_interfaces/results/fields/"

DELX = L/N_DNS
DELY = L/N_DNS


#------------------------------------------- initialization
os.system("rm results_latentSpace/*.png")
os.system("rm -rf results_latentSpace/plots")
os.system("rm -rf results_latentSpace/fields")
os.system("rm -rf results_latentSpace/energy")
if (LOAD_FIELD):
    os.system("rm -rf results_latentSpace/plots_org")
    os.system("rm -rf results_latentSpace/fields_org")
    os.system("rm -rf results_latentSpace/energy_org")
os.system("rm -rf results_latentSpace/logs")

os.system("mkdir -p results_latentSpace/plots")
os.system("mkdir -p results_latentSpace/fields")
os.system("mkdir -p results_latentSpace/energy")
os.system("mkdir -p results_latentSpace/plots_org/")
os.system("mkdir -p results_latentSpace/fields_org")
os.system("mkdir -p results_latentSpace/energy_org")

dir_log = 'results_latentSpace/logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(SEED_RESTART)



if (not RESTART_WL):
    os.system("rm -rf " + Z0_DIR_WL)
    os.system("mkdir -p " + Z0_DIR_WL)


#------------------------------------------------------ define optimizer for z and wl_dlatents search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_DNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_DNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_kDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_DNS)

train_summary_writer = tf.summary.create_file_writer(Z0_DIR_WL_LOGS + "search/")


# loading StyleGAN checkpoint and filter
managerCheckpoint = tf.train.CheckpointManager(checkpoint_StylES, '../' + CHKP_DIR, max_to_keep=2)
checkpoint.restore(managerCheckpoint.latest_checkpoint)
if managerCheckpoint.latest_checkpoint:
    print("Net restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=2))
else:
    print("Initializing net from scratch.")
time.sleep(3)



# create filter model
if (GAUSSIAN_FILTER):
    x_in    = tf.keras.Input(shape=([NUM_CHANNELS, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out     = apply_filter_NCH(x_in, size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian', NCH=NUM_CHANNELS)
    gfilter = tf.keras.Model(inputs=x_in, outputs=out)

    x_1ch       = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out_1ch     = apply_filter_NCH(x_1ch, size=4*RS, rsca=RS, mean=0.0, delta=RS, type='Gaussian', NCH=1)
    gfilter_1ch = tf.keras.Model(inputs=x_1ch, outputs=out_1ch)

    x_sub       = tf.keras.Input(shape=([1, RS+1, RS+1]), dtype=DTYPE)
    out_sub     = apply_filter_NCH(x_sub, size=4*RS, rsca=1, mean=0.0, delta=RS, subsection=True, type='Gaussian', NCH=1)
    gfilter_sub = tf.keras.Model(inputs=x_sub, outputs=out_sub)
    
    x_noScaling       = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
    out_noScaling     = apply_filter_NCH(x_noScaling, size=4*RS, rsca=1, mean=0.0, delta=RS, type='Gaussian', NCH=1)
    gfilter_noScaling = tf.keras.Model(inputs=x_noScaling, outputs=out_noScaling)
else:
    gfilter = filters[IFIL]



# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    
print("\n kDNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)


time.sleep(3)


print("============================Finished initialization")



#------------------------------------------------------ set reference DNS and LES
# set z
z0 = tf.random.uniform(shape=[1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)
dlatents = mapping(z0, training=False)

if (LOAD_DNS):
    
    # load numpy array
    U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_DNS)
    U_DNS = np.cast[DTYPE](U_DNS)
    V_DNS = np.cast[DTYPE](V_DNS)
    P_DNS = np.cast[DTYPE](P_DNS)

    U_DNS = U_DNS[:,::6,:]
    V_DNS = V_DNS[:,::6,:]
    P_DNS = P_DNS[:,::6,:]

    NX_DNS = U_DNS.shape[0]
    NY_DNS = U_DNS.shape[1]
    NZ_DNS = U_DNS.shape[2]
    print("Dimensions of original file are: ", NX_DNS, NY_DNS, NZ_DNS)
    
    # convert to tf
    U_DNS = tf.convert_to_tensor(U_DNS, dtype=DTYPE)
    V_DNS = tf.convert_to_tensor(V_DNS, dtype=DTYPE)
    P_DNS = tf.convert_to_tensor(P_DNS, dtype=DTYPE)

    if (NDIMS==3):
        U_DNS = tf.transpose(U_DNS, [1,0,2])
        V_DNS = tf.transpose(V_DNS, [1,0,2])
        P_DNS = tf.transpose(P_DNS, [1,0,2])
        U_DNS = U_DNS[:,tf.newaxis,:,:]
        V_DNS = V_DNS[:,tf.newaxis,:,:]
        P_DNS = P_DNS[:,tf.newaxis,:,:]
    else:
        U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
        V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
        P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]
        
    UVP_DNS_org = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)

    # filter and dowscale if needed
    rsin = int(NX_DNS/OUTPUT_DIM)
    if (rsin>1):
        UVP_DNS     = apply_filter_NCH(UVP_DNS_org, size=4*rsin, rsca=rsin, mean=0.0, delta=rsin, type='Gaussian', NCH=3)
        U_DNS       = UVP_DNS[:,0:1,:,:]
        V_DNS       = UVP_DNS[:,1:2,:,:]
        if (CALC_VORTICITY):
            P_DNS  = apply_filter_NCH(V_DNS, size=2, rsca=1, mean=0.0, delta=DELX*rsin, type='Vorticity', NCH=1)
        else:
            P_DNS = UVP_DNS[:,2:3,:,:]
        UVP_DNS_org = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)

    UVP_LES_org = gfilter(UVP_DNS_org)
    if (CALC_VORTICITY):
        P_DNS = find_vorticity_HW(UVP_LES_org[:,1:2,:,:], DELX_LES, DELY_LES)
        UVP_LES_org = tf.concat([UVP_LES_org[:,0:2,:,:], P_DNS], axis=1)  
        
    # filter image
    rs = 2 
    for reslog in range(RES_LOG2, RES_LOG2-FIL-1, -1):
        res = 2**reslog
        if (reslog==RES_LOG2):
            fUVP_DNS, _ = normalize_max(UVP_DNS_org)
        else:
            fUVP_DNS = apply_filter_NCH(fUVP_DNS, size=4, rsca=rs, mean=0.0, delta=1.0, type='Gaussian', NCH=3)
            U_DNS    = fUVP_DNS[:,0:1,:,:]
            V_DNS    = fUVP_DNS[:,1:2,:,:]
            if (CALC_VORTICITY):        
                P_DNS  = apply_filter_NCH(V_DNS, size=2, rsca=1, mean=0.0, delta=DELX*OUTPUT_DIM/res, type='Vorticity', NCH=1)
                fUVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)
            else:
                P_DNS    = fUVP_DNS[:,2:3,:,:]
            fUVP_DNS = find_centred_fields(fUVP_DNS)
            fUVP_DNS, _ = normalize_max(fUVP_DNS)

        # normalize the data
        fUVP_DNS, _ = normalize_max(fUVP_DNS)
        
    # save LES_in0
    LES_in0 = tf.identity(fUVP_DNS)

else:

    # set UVP max
    UVP_max = tf.constant([INIT_SCA], dtype=DTYPE)
    UVP_max = tf.tile(UVP_max, [3])
    UVP_max = UVP_max[tf.newaxis, :, tf.newaxis, tf.newaxis]
    UVP_max = [UVP_max] + [UVP_max]

    # inference
    if (NUM_CHANNELS==1):
        pre_img, V_LES  = pre_synthesis(dlatents, training = False)
        U_LES = V_LES
        P_LES  = apply_filter_NCH(V_LES, size=2, rsca=1, mean=0.0, delta=DELX*RS, type='Vorticity', NCH=1)
        P_LES = find_centred_fields(P_LES)
        P_LES, _ = normalize_max(P_LES)
        LES_in0 = tf.concat([U_LES, V_LES, P_LES], axis=1)
        zAll = [dlatents, pre_img, LES_in0]
    else:
        pre_img  = pre_synthesis(dlatents, training = False)
        LES_in0  = pre_img[-1]
        zAll     = [dlatents, pre_img, LES_in0]

    if (NDIMS==3):
        LES_in0_init = LES_in0[0:1,:,:,:]
        LES_in0      = LES_in0_init
        for j in range(1, BATCH_SIZE):
            sinLES = int(np.sin(2*np.pi*j/float(BATCH_SIZE-1))*N_LES2)
            sinDNS = sinLES*RS
            LES_in0 = tf.concat([LES_in0, tr(LES_in0_init, 0, sinLES)], axis=0)

    UVP_DNS_org, UVP_LES_org, _ = find_predictions(synthesis, gfilter, zAll, UVP_max)


# find min/max values
fnUVPo, nfUVPo, UVP_amaxo, fUVP_amaxo = find_scaling(UVP_DNS_org, gfilter_1ch)
UVP_max = [UVP_amaxo] + [fUVP_amaxo]
Umax = abs(tf.reduce_max(UVP_amaxo[:,0,:,:]).numpy())
Vmax = abs(tf.reduce_max(UVP_amaxo[:,1,:,:]).numpy())
Pmax = abs(tf.reduce_max(UVP_amaxo[:,2,:,:]).numpy())
print(Umax, Vmax, Pmax)
Umin = -Umax
Vmin = -Vmax
Pmin = -Pmax


# print original plots DNS and LES
U_DNS = UVP_DNS_org[0,0,:,:].numpy()
V_DNS = UVP_DNS_org[0,1,:,:].numpy()
P_DNS = UVP_DNS_org[0,2,:,:].numpy()

filename = Z0_DIR_WL + "plots_DNS_org.png"
print_fields_3(U_DNS, V_DNS, P_DNS, filename=filename, testcase=TESTCASE, \
            Umin=Umin, Umax=Umax, Vmin=Vmin, Vmax=Vmax, Pmin=Pmin, Pmax=Pmax)

U_LES = UVP_LES_org[0,0,:,:].numpy()
V_LES = UVP_LES_org[0,1,:,:].numpy()
P_LES = UVP_LES_org[0,2,:,:].numpy()

filename = Z0_DIR_WL + "plots_LES_org.png"
print_fields_3(U_LES, V_LES, P_LES, filename=filename, testcase=TESTCASE, \
            Umin=Umin, Umax=Umax, Vmin=Vmin, Vmax=Vmax, Pmin=Pmin, Pmax=Pmax)


dVdx = (-cr(V_DNS, 2, 0) + 8*cr(V_DNS, 1, 0) - 8*cr(V_DNS, -1,  0) + cr(V_DNS, -2,  0))/(12.0*DELX)
dVdy = (-cr(V_DNS, 0, 2) + 8*cr(V_DNS, 0, 1) - 8*cr(V_DNS,  0, -1) + cr(V_DNS,  0, -2))/(12.0*DELY)
plot_spectrum_2d_3v(U_DNS, dVdx, dVdy, L, filename_spectra, label="DNS(org)", close=False)

dVdx = (-cr(V_LES, 2, 0) + 8*cr(V_LES, 1, 0) - 8*cr(V_LES, -1,  0) + cr(V_LES, -2,  0))/(12.0*DELX_LES)
dVdy = (-cr(V_LES, 0, 2) + 8*cr(V_LES, 0, 1) - 8*cr(V_LES,  0, -1) + cr(V_LES,  0, -2))/(12.0*DELY_LES)
plot_spectrum_2d_3v(U_LES, dVdx, dVdy, L, filename_spectra, label="LES(org)", close=False)


print("============================Set reference DNS and LES")


#------------------------------------------- loop over different latent spaces
for k in range(NL):
    
    # load initial flow
    if (LOAD_FIELD):

        # load initial flow
        if (TESTCASE=='HIT_2D'):
            tail = str(int(k*100+10000))
            FILE_REAL = FILE_REAL_PATH + "fields_run0_it" + tail + ".npz"

        if (TESTCASE=='HW' or TESTCASE=='mHW'):
            tail = str(int(k)).zfill(5)
            FILE_REAL = FILE_REAL_PATH + "fields_time00000.npz"

        # load numpy array
        U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_REAL)
        U_DNS = np.cast[DTYPE](U_DNS)
        V_DNS = np.cast[DTYPE](V_DNS)
        P_DNS = np.cast[DTYPE](P_DNS)

        # find vorticity
        if (TESTCASE=='HIT_2D'):
            P_DNS  = find_vorticity(U_DNS, V_DNS)
            cP_DNS = find_vorticity(U_DNS, V_DNS)
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            # cP_DNS = (tr(V_DNS, 1, 0) - 2*V_DNS + tr(V_DNS, -1, 0))/(DELX**2) \
            #            + (tr(V_DNS, 0, 1) - 2*V_DNS + tr(V_DNS, 0, -1))/(DELY**2)
            cP_DNS = (-tr(V_DNS, 2, 0) + 16*tr(V_DNS, 1, 0) - 30*V_DNS + 16*tr(V_DNS,-1, 0) - tr(V_DNS,-2, 0))/(12*DELX**2) \
                   + (-tr(V_DNS, 0, 2) + 16*tr(V_DNS, 0, 1) - 30*V_DNS + 16*tr(V_DNS, 0,-1) - tr(V_DNS, 0,-2))/(12*DELY**2)

        # normalize
        U_min = np.min(U_DNS)
        U_max = np.max(U_DNS)
        V_min = np.min(V_DNS)
        V_max = np.max(V_DNS)
        P_min = np.min(P_DNS)
        P_max = np.max(P_DNS)

        U_norm = max(np.absolute(U_min), np.absolute(U_max))
        V_norm = max(np.absolute(V_min), np.absolute(V_max))
        P_norm = max(np.absolute(P_min), np.absolute(P_max))
        
        # print("DNS fields min/max", U_min, U_max, V_min, V_max, P_min, P_max)
        # print("Normalization values", U_norm, V_norm, P_norm)

        U_DNS = U_DNS/U_norm*INIT_SCA
        V_DNS = V_DNS/V_norm*INIT_SCA
        P_DNS = P_DNS/P_norm*INIT_SCA
        
        U_DNS_org = tf.identity(U_DNS)
        V_DNS_org = tf.identity(V_DNS)
        P_DNS_org = tf.identity(P_DNS)
        

        # save org fields
        filename = "results_latentSpace/fields_org/fields_lat" + str(k) + "_res" + str(N_DNS) + ".npz"
        save_fields(0, U_DNS_org, V_DNS_org, P_DNS_org, filename=filename)

        # print plots
        filename = "results_latentSpace/plots_org/Plots_DNS_org_" + tail +".png"
        print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N=N_DNS, filename=filename, testcase=TESTCASE) #, \
                       # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

        # print spectrum
        filename = "results_latentSpace/energy_org/energy_spectrum_org_" + str(k) + ".png"
        closePlot=True
        if (TESTCASE=='HIT_2D'):
            plot_spectrum_2d_3v(U_DNS_org, V_DNS_org, L, filename, close=closePlot, label='DNS')
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            gradV = np.sqrt(((cr(V_DNS_org, 1, 0) - cr(V_DNS_org, -1, 0))/(2.0*DELX))**2 \
                          + ((cr(V_DNS_org, 0, 1) - cr(V_DNS_org, 0, -1))/(2.0*DELY))**2)
            plot_spectrum_2d_3v(U_DNS_org, gradV, L, filename, close=closePlot, label='DNS')



        #-------------- preprare targets
        U_DNS = tf.convert_to_tensor(U_DNS, dtype=DTYPE)
        V_DNS = tf.convert_to_tensor(V_DNS, dtype=DTYPE)
        P_DNS = tf.convert_to_tensor(P_DNS, dtype=DTYPE)

        U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
        V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
        P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]
        
        fU_DNS = gfilter(U_DNS)
        fV_DNS = gfilter(V_DNS)
        fP_DNS = gfilter(P_DNS)
        
        LES_in = []
        for res in range(2,RES_LOG2-FIL+1):
            rs = 2**(RES_LOG2-FIL-res)
            fU_DNS_res = fU_DNS[:,:,::rs,::rs]
            fV_DNS_res = fV_DNS[:,:,::rs,::rs]
            fP_DNS_res = fP_DNS[:,:,::rs,::rs]
            fUVP_res   = tf.concat([fU_DNS_res, fV_DNS_res, fP_DNS_res], axis=1)
            fUVP_res   = normalize_max(fUVP_res)
            LES_in.append(fUVP_res)

        # set targets
        imgA  = tf.concat([U_DNS, V_DNS, P_DNS], axis=1)
        fimgA = tf.identity(LES_in[-1])
        

        # find multiplier for DNS field
        UVP_max = [INIT_SCA, INIT_SCA, INIT_SCA]

       
        # start research on the latent space
        it     = 0
        resREC = large
        tstart = time.time()
        while (resREC>tollDNS and it<lr_DNS_maxIt):

            lr = lr_schedule_DNS(it)
            UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, _, _ = \
                step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, [z0, LES_in], imgA, fimgA, ltv_DNS, UVP_max, typeRes=0)

            # adjust variables
            for lvar in range(len(layer_kDNS.trainable_variables[:])):
                kDNS  = layer_kDNS.trainable_variables[lvar]
                kDNSn = tf.clip_by_value(kDNS, 0.0, 1.0)
                if (tf.reduce_any((kDNS-kDNSn)>0)):
                    layer_kDNS.trainable_variables[lvar].assign(kDNSn)
                    UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, [z0, LES_in], UVP_max)
                    resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=0)

            # print residuals and fields
            if (it%10==0):

                # separate DNS fields from GAN
                U_DNS = UVP_DNS[0, 0, :, :].numpy()
                V_DNS = UVP_DNS[0, 1, :, :].numpy()
                P_DNS = UVP_DNS[0, 2, :, :].numpy()
                
                # print residuals
                tend = time.time()
                print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
                    .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

                # write losses to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('resREC',       resREC,       step=it)
                    tf.summary.scalar('loss_fil',     loss_fil,     step=it)                    
                    tf.summary.scalar('lr',           lr,           step=it)

                if (it%1000==0):
                    # filename = "results_latentSpace/plots/Plots_DNS_fromGAN.png"
                    filename = "results_latentSpace/plots/Plots_DNS_fromGAN_" + str(it).zfill(6) + ".png"
                    print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename, testcase=TESTCASE) #, \
                        # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

                    filename = "results_latentSpace/plots/Plots_DNS_UfromGAN_" + str(it).zfill(6) + ".png"
                    print_fields_3(U_DNS_org, U_DNS, U_DNS_org-U_DNS, N=N_DNS, filename=filename, testcase=TESTCASE, plot='diff', \
                        Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)
                    
                    filename = "results_latentSpace/plots/Plots_DNS_VfromGAN_" + str(it).zfill(6) + ".png"
                    print_fields_3(V_DNS_org, V_DNS, V_DNS_org-V_DNS, N=N_DNS, filename=filename, testcase=TESTCASE, plot='diff', \
                        Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)
                    
                    filename = "results_latentSpace/plots/Plots_DNS_PfromGAN_" + str(it).zfill(6) + ".png"
                    print_fields_3(P_DNS_org, P_DNS, P_DNS_org-P_DNS, N=N_DNS, filename=filename, testcase=TESTCASE, plot='diff', \
                        Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

            it = it+1


        # print final residuals
        lr = lr_schedule_DNS(it)
        tend = time.time()
        print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
            .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

    else:

        # find DNS and LES fields from random input
        if (k>=0):
            z0 = tf.random.uniform(shape=[BATCH_SIZE, 1 + 2*(G_LAYERS-M_LAYERS), LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)

        UVP_max = [INIT_SCA, INIT_SCA, INIT_SCA]
        UVP_DNS, UVP_LES, fUVP_DNS, _, predictions = find_predictions(fwl_synthesis, gfilter, z0, UVP_max)

        #-------------- preprare targets
        imgA   = tf.identity(UVP_DNS)
        fimgA  = tf.identity(UVP_LES) 
        LES_in = []
        for res in range(2,RES_LOG2-FIL+1):
            LES_in.append(predictions[res-2])



    #--------------------------------------------- check filtered quantities

    # find delx_LES and dely_LES
    delx_LES    = DELX*2**FIL
    dely_LES    = DELY*2**FIL
    if (k==0):
        print("Find LES quantities. Delx_LES is: ", delx_LES)


    #-------- get fields
    res = 2**(RES_LOG2-FIL)
    UVP_DNS, UVP_LES, fUVP_DNS, _, predictions = find_predictions(wl_synthesis, gfilter, [z0, LES_in], UVP_max)
      
    # DNS fields
    U_DNS = UVP_DNS[0,0,:,:]
    V_DNS = UVP_DNS[0,1,:,:]
    P_DNS = UVP_DNS[0,2,:,:]

    # LES fields
    U_LES = UVP_LES[0,0,:,:]
    V_LES = UVP_LES[0,1,:,:]
    P_LES = UVP_LES[0,2,:,:]
        
    # fDNS fields
    fU_DNS   = (gfilter(U_DNS[tf.newaxis,tf.newaxis,:,:]))
    fV_DNS   = (gfilter(V_DNS[tf.newaxis,tf.newaxis,:,:]))
    fP_DNS   = (gfilter(P_DNS[tf.newaxis,tf.newaxis,:,:]))
    fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], 1)  
    fU_DNS   = fUVP_DNS[0,0,:,:]
    fV_DNS   = fUVP_DNS[0,1,:,:]
    fP_DNS   = fUVP_DNS[0,2,:,:]   

    # plot fields
    filename = "results_latentSpace/plots/plots_LES_lat" + str(k) + "_res" + str(res) + ".png"
    print_fields_3(U_LES, V_LES, P_LES, N=res, filename=filename, testcase=TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

    filename = "results_latentSpace/plots/plots_fDNS_lat" + str(k) + "_res" + str(res) + ".png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, N=res, filename=filename, testcase=TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

    filename = "results_latentSpace/plots/plots_diffLES_lat" + str(k) + "_res" + str(res) + ".png"
    print_fields_3(U_LES-fU_DNS, V_LES-fV_DNS, P_LES-fP_DNS, N=res, filename=filename, testcase=TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, plot='diff')


    #-------- verify filter properties
    # find
    cf_DNS =                               (10.0*P_DNS)[tf.newaxis,tf.newaxis,:,:]  # conservation
    lf_DNS =                            (U_DNS + V_DNS)[tf.newaxis,tf.newaxis,:,:]  # linearity
    df_DNS = ((tr(P_DNS, 1, 0) - tr(P_DNS,-1, 0))/DELX)[tf.newaxis,tf.newaxis,:,:]  # commutative with derivatives
    
    cf_DNS = (gfilter(cf_DNS))[0,0,:,:]
    lf_DNS = (gfilter(lf_DNS))[0,0,:,:]
    df_DNS = (gfilter(df_DNS))[0,0,:,:]
    
    c_LES = 10.0*fP_DNS
    l_LES = fU_DNS + fV_DNS
    d_LES = (tr((gfilter(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:], 1, 0) \
           - tr((gfilter(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:],-1, 0))/delx_LES

    # plot
    filename = "results_latentSpace/plots/plots_conservation_lat" + str(k) + "_res" + str(res) + ".png"
    print_fields_3(cf_DNS, c_LES, cf_DNS-c_LES, N=res, filename=filename, testcase=TESTCASE, plot='diff')

    filename = "results_latentSpace/plots/plots_linearity_lat" + str(k) + "_res" + str(res) + ".png"
    print_fields_3(lf_DNS, l_LES, lf_DNS-l_LES, N=res, filename=filename, testcase=TESTCASE, plot='diff')

    filename = "results_latentSpace/plots/plots_derivatives_lat" + str(k) + "_res" + str(res) + ".png"
    print_fields_3(df_DNS, d_LES, df_DNS-d_LES, N=res, filename=filename, testcase=TESTCASE, plot='diff')


    # spectrum
    filename = "results_latentSpace/energy/energy_spectrum_LES_lat" + str(k) + "_res" + str(res) + ".png"
    if (TESTCASE=='HIT_2D'):
        plot_spectrum_2d_3v(fU_DNS, fV_DNS, L, filename, close=True, label='LES')
    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        gradV = np.sqrt(((cr(fV_DNS, 1, 0) - cr(fV_DNS, -1, 0))/(2.0*delx_LES))**2 \
                      + ((cr(fV_DNS, 0, 1) - cr(fV_DNS, 0, -1))/(2.0*dely_LES))**2)
        plot_spectrum_2d_3v(fU_DNS, gradV, L, filename, close=False, label='LES')

    filename = "results_latentSpace/energy/energy_spectrum_DNS_lat" + str(k) + "_res" + str(res) + ".png"
    if (TESTCASE=='HIT_2D'):
        plot_spectrum_2d_3v(U_DNS_org, V_DNS_org, L, filename, close=closePlot, label='DNS')
    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        gradV = np.sqrt(((cr(V_DNS_org, 1, 0) - cr(V_DNS_org, -1, 0))/(2.0*DELX))**2 \
                      + ((cr(V_DNS_org, 0, 1) - cr(V_DNS_org, 0, -1))/(2.0*DELY))**2)
        plot_spectrum_2d_3v(U_DNS_org, gradV, L, filename, close=True, label='DNS')




    #--------------------------------------------- find fields for each layer
    closePlot=False
    for kk in range(RES_LOG2-NFIL, RES_LOG2+1):
        UVP_DNS = predictions[kk-2]
        res = 2**kk
        
        delx = DELX*2**(RES_LOG2-kk)
        dely = DELY*2**(RES_LOG2-kk)

        U_DNS = UVP_DNS[0, 0, :, :].numpy()
        V_DNS = UVP_DNS[0, 1, :, :].numpy()
        P_DNS = UVP_DNS[0, 2, :, :].numpy()

        # find vorticity field
        if (TESTCASE=='HIT_2D'):
            cP_DNS = find_vorticity(U_DNS, V_DNS)
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            # V_DNS = sc.ndimage.gaussian_filter(V_DNS, 4, mode=['wrap','wrap'])
            # cP_DNS = (tr(V_DNS, 1, 0) - 2*V_DNS + tr(V_DNS, -1, 0))/(delx**2) \
            #          + (tr(V_DNS, 0, 1) - 2*V_DNS + tr(V_DNS, 0, -1))/(dely**2)
            cP_DNS = (-tr(V_DNS, 2, 0) + 16*tr(V_DNS, 1, 0) - 30*V_DNS + 16*tr(V_DNS,-1, 0) - tr(V_DNS,-2, 0))/(12*delx**2) \
                   + (-tr(V_DNS, 0, 2) + 16*tr(V_DNS, 0, 1) - 30*V_DNS + 16*tr(V_DNS, 0,-1) - tr(V_DNS, 0,-2))/(12*dely**2)
            # cP_DNS = sc.ndimage.gaussian_filter(cP_DNS.numpy(), 1, mode=['wrap','wrap'])
            # cP_DNS =tf.convert_to_tensor(cP_DNS, dtype=DTYPE)

        # plot fields        
        filename = "results_latentSpace/plots/plots_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(U_DNS, V_DNS, P_DNS, N=res, filename=filename, testcase=TESTCASE)

        if (TESTCASE=='HIT_2D'):
            filename = "results_latentSpace/plots/plots_VortDiff_" + str(k) + "_res" + str(res) + ".png"
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            filename = "results_latentSpace/plots/plots_PhiDiff_" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, N=res, filename=filename, testcase=TESTCASE, plot='diff')

        # filename = "results_latentSpace/plots/plot_1field_lat" + str(k) + "_res" + str(res) + ".png"
        # print_fields_1(P_DNS, filename, legend=False)

        # save fields
        filename = "results_latentSpace/fields/fields_lat"  + str(k) + "_res" + str(res) + ".npz"
        save_fields(0, U_DNS, V_DNS, P_DNS, filename=filename)
        
    print ("Done for latent " + str(k))
