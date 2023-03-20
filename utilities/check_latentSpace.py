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
NL          = 1       # number of different latent vectors randomly selected
TUNE_NOISE  = True
LOAD_FIELD  = True    # load field from DNS solver (via restart.npz file)
RESTART_WL  = False
CHKP_DIR_WL = "./checkpoints_wl"
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2_FIL
N2_DNS      = int(N_DNS/2)
N2_LES      = int(N_LES/2)
tollDNS     = 1.0e-4
lr_kDNS_It  = 100

if (TESTCASE=='HIT_2D'):
    FILE_REAL_PATH  = "../LES_Solvers/fields/"
    from HIT_2D import L, N
elif (TESTCASE=='HW'):
    FILE_REAL_PATH  = "../../../data/Fields/HW/fields_N256/"   # to do: make more general
    L = 50.176
elif (TESTCASE=='mHW'):
    FILE_REAL_PATH  = "../../../data/Fields/mHW/fields_N512/"   # to do: make more general
    L = 50.176

DELX = L/N_DNS
DELY = L/N_DNS




#------------------------------------------- initialization
zero_DNS = np.zeros([N_DNS, N_DNS], dtype=DTYPE)
zero_LES = np.zeros([N_LES, N_LES], dtype=DTYPE)

os.system("rm results_latentSpace/*.png")
os.system("rm -rf results_latentSpace/plots")
os.system("rm -rf results_latentSpace/fields")
os.system("rm -rf results_latentSpace/uvw")
os.system("rm -rf results_latentSpace/energy")
if (LOAD_FIELD):
    os.system("rm -rf results_latentSpace/plots_org")
    os.system("rm -rf results_latentSpace/fields_org")
    os.system("rm -rf results_latentSpace/uvw_org")
    os.system("rm -rf results_latentSpace/energy_org")
os.system("rm -rf logs")

os.system("mkdir -p results_latentSpace/plots")
os.system("mkdir -p results_latentSpace/fields")
os.system("mkdir -p results_latentSpace/uvw")
os.system("mkdir -p results_latentSpace/energy")
os.system("mkdir -p results_latentSpace/plots_org/")
os.system("mkdir -p results_latentSpace/fields_org")
os.system("mkdir -p results_latentSpace/uvw_org")
os.system("mkdir -p results_latentSpace/energy_org")

dir_log = 'logs/'
train_summary_writer = tf.summary.create_file_writer(dir_log)
tf.random.set_seed(SEED_RESTART)


# loading StyleGAN checkpoint
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=1)
checkpoint.restore(managerCheckpoint.latest_checkpoint)

if managerCheckpoint.latest_checkpoint:
    print("StyleGAN restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing StyleGAN from scratch.")

time.sleep(3)


# create variable synthesis model
layer_k = layer_klatent()
layer_m = layer_mlatent_DNS()

zlatents     = tf.keras.Input(shape=([LATENT_SIZE]),           dtype=DTYPE)
w2           = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)
zlatents_DNS = layer_k(zlatents)
w1           = mapping(zlatents_DNS)
w            = layer_m(w1, w2)
outputs      = synthesis(w, training=False)
wl_synthesis = tf.keras.Model(inputs=[zlatents, w2], outputs=outputs)


# define optimizer for kDNS search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_kDNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_kDNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_k = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_kDNS)


# define optimizer for mDNS search
if (lr_LES_POLICY=="EXPONENTIAL"):
    lr_schedule_mDNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_LES,
        decay_steps=lr_LES_STEP,
        decay_rate=lr_LES_RATE,
        staircase=lr_LES_EXP_ST)
elif (lr_LES_POLICY=="PIECEWISE"):
    lr_schedule_mDNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_LES_BOUNDS, lr_LES_VALUES)
opt_m = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_mDNS)


# define checkpoints wl_synthesis and filter
checkpoint_wl = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# set trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []

for variable in layer_k.trainable_variables:
    ltv_DNS.append(variable)

for variable in layer_m.trainable_variables:
    ltv_DNS.append(variable)
    
print("\n ltv_DNS variables:")
for variable in ltv_DNS:
    print(variable.name, variable.shape)

time.sleep(3)


# re-load checkpoint and set latent space z
if (RESTART_WL):
    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")
    data = np.load("results_latentSpace/zlatents.npz")
    zlatents = data["zlatents"]
else:
    zlatents = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)




#------------------------------------------- loop over different latent spaces
for k in range(NL):
    
    # load initial flow
    tf.random.set_seed(k)
    if (LOAD_FIELD):

        # load initial flow
        if (TESTCASE=='HIT_2D'):
            tail = str(int(k*100+100))
            FILE_REAL = FILE_REAL_PATH + "fields_run0_it" + tail + ".npz"

        if (TESTCASE=='HW'):
            tail = str(int(k+200))
            FILE_REAL = FILE_REAL_PATH + "fields_run0_time" + tail + ".npz"

        if (TESTCASE=='mHW'):
            tail = str(int(k+200))
            FILE_REAL = FILE_REAL_PATH + "fields_run1000_time" + tail + ".npz"


        # load original field
        if (FILE_REAL.endswith('.npz')):

            # load numpy array
            U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_REAL)
            U_DNS_org = np.cast[DTYPE](U_DNS)
            V_DNS_org = np.cast[DTYPE](V_DNS)
            P_DNS_org = np.cast[DTYPE](P_DNS)

        elif (FILE_REAL.endswith('.png')):

            # load image
            orig = Image.open(FILE_REAL).convert('RGB')

            # convert to black and white, if needed
            if (NUM_CHANNELS==1):
                orig = orig.convert("L")

            # remove white spaces
            #orig = trim(orig)

            # resize images
            orig = orig.resize((OUTPUT_DIM,OUTPUT_DIM))

            # convert to numpy array
            orig = np.asarray(orig, dtype=DTYPE)
            orig = orig/255.0

            U_DNS_org = orig[:,:,0]
            V_DNS_org = orig[:,:,1]
            P_DNS_org = orig[:,:,2]


        # find vorticity
        if (TESTCASE=='HIT_2D'):
            P_DNS_org  = find_vorticity(U_DNS_org, V_DNS_org)
            cP_DNS_org = find_vorticity(U_DNS_org, V_DNS_org)
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            # cP_DNS_org = (tr(V_DNS_org, 1, 0) - 2*V_DNS_org + tr(V_DNS_org, -1, 0))/(DELX**2) \
            #            + (tr(V_DNS_org, 0, 1) - 2*V_DNS_org + tr(V_DNS_org, 0, -1))/(DELY**2)
            cP_DNS_org = (-tr(V_DNS_org, 2, 0) + 16*tr(V_DNS_org, 1, 0) - 30*V_DNS_org + 16*tr(V_DNS_org,-1, 0) - tr(V_DNS_org,-2, 0))/(12*DELX**2) \
                       + (-tr(V_DNS_org, 0, 2) + 16*tr(V_DNS_org, 0, 1) - 30*V_DNS_org + 16*tr(V_DNS_org, 0,-1) - tr(V_DNS_org, 0,-2))/(12*DELY**2)

        # normalize
        U_min = np.min(U_DNS_org)
        U_max = np.max(U_DNS_org)
        V_min = np.min(V_DNS_org)
        V_max = np.max(V_DNS_org)
        P_min = np.min(P_DNS_org)
        P_max = np.max(P_DNS_org)

        U_DNS_org = 2.0*(U_DNS_org - U_min)/(U_max - U_min) - 1.0
        V_DNS_org = 2.0*(V_DNS_org - V_min)/(V_max - V_min) - 1.0
        P_DNS_org = 2.0*(P_DNS_org - P_min)/(P_max - P_min) - 1.0


        # print plots
        filename = "results_latentSpace/plots_org/Plots_DNS_org_" + tail +".png"
        print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N_DNS, filename, \
                       Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

        filename = "results_latentSpace/plots_org/Plots_DNS_org_diffVort_" + tail +".png"
        print_fields_3(P_DNS_org, cP_DNS_org, P_DNS_org-cP_DNS_org, N_DNS, filename, \
                       Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

        filename = "results_latentSpace/energy_org/energy_spectrum_org_" + str(k) + ".png"
        plot_spectrum(U_DNS_org, V_DNS_org, L, filename, label='DNS')
        
        if (TESTCASE=='HW' or TESTCASE=='mHW'):
            print("Total and calculated total vorticity for original image are: ")
            print(np.sum(P_DNS_org), np.sum(cP_DNS_org))

        filename = "results_latentSpace/fields_org/fields_lat_" + str(k) + "_res_" + str(N_DNS) + ".npz"
        save_fields(0, U_DNS_org, V_DNS_org, P_DNS_org, zero_DNS, zero_DNS, zero_DNS, filename)


        #-------------- preprare targets
        U_DNS = U_DNS_org[np.newaxis,np.newaxis,:,:]
        V_DNS = V_DNS_org[np.newaxis,np.newaxis,:,:]
        P_DNS = P_DNS_org[np.newaxis,np.newaxis,:,:]

        U_DNS = tf.convert_to_tensor(U_DNS)
        V_DNS = tf.convert_to_tensor(V_DNS)
        P_DNS = tf.convert_to_tensor(P_DNS)

        # concatenate
        imgA  = tf.concat([U_DNS, V_DNS, P_DNS], 1)

        # filter
        fU_DNS = filter(U_DNS)
        fV_DNS = filter(V_DNS)
        fP_DNS = filter(P_DNS)                

        fimgA  = tf.concat([fU_DNS, fV_DNS, fP_DNS], 1)

        
        #-------------- start research on the latent space
        it = 0
        resREC = large
        wlatents = mapping(zlatents, training=False)
        tstart = time.time()
        while (resREC>tollDNS and it<lr_DNS_maxIt):

            # iterate on zlatent 1
            if (it<lr_kDNS_It):
                resREC, resDNS, resLES, UVP_DNS = step_find_latents_kDNS(wl_synthesis, opt_k, zlatents, wlatents, imgA, fimgA, ltv_DNS)

            if (it==lr_kDNS_It):
                # find z1,w1
                k_new = layer_k.trainable_variables[0].numpy()
                z1 = k_new*zlatents
                layer_k.trainable_variables[0].assign(tf.fill((LATENT_SIZE), 1.0))
                w1 = mapping(z1, training=False)
                zlatents = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
                wlatents = mapping(zlatents, training=False)

            # iterate on zlatent 2
            if (it>=lr_kDNS_It and it<2*lr_kDNS_It):
                resREC, resDNS, resLES, UVP_DNS = step_find_latents_kDNS(wl_synthesis, opt_k, zlatents, wlatents, imgA, fimgA, ltv_DNS)

            if (it==2*lr_kDNS_It):
                # find z2,w2
                k_new = layer_k.trainable_variables[0].numpy()
                z2 = k_new*zlatents
                layer_k.trainable_variables[0].assign(tf.fill((LATENT_SIZE), 1.0))
                w2 = mapping(z2, training=False)

            # find new M
            if (it>=2*lr_kDNS_It):            
                resREC, resDNS, resLES, UVP_DNS = step_find_latents_mDNS(wl_synthesis, opt_m, z2, w1, imgA, fimgA, ltv_DNS)

            # print residuals and fields
            if (it%100==0):

                if (it<2*lr_kDNS_It):
                    lr = lr_schedule_kDNS(it)
                else:
                    lr = lr_schedule_mDNS(it)
            
                # print residuals
                tend = time.time()
                print("DNS iterations:  time {0:3e}   step {1:4d}  it {2:6d}  resREC {3:3e} resDNS {4:3e}  resLES {5:3e} lr {6:3e} " \
                    .format(tend-tstart, k, it, resREC.numpy(), resDNS.numpy(), resLES.numpy(), lr))

                # write losses to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('resREC',   resREC,   step=it)
                    tf.summary.scalar('resDNS',   resDNS,   step=it)
                    tf.summary.scalar('resLES',   resLES,   step=it)                    
                    tf.summary.scalar('lr',       lr,       step=it)

                if (it%1000==0):

                    # separate DNS fields from GAN
                    U_DNS = UVP_DNS[0, 0, :, :].numpy()
                    V_DNS = UVP_DNS[0, 1, :, :].numpy()
                    P_DNS = UVP_DNS[0, 2, :, :].numpy()

                    # filename = "results_latentSpace/plots/Plots_DNS_fromGAN_" + str(it).zfill(5) + ".png"
                    filename = "results_latentSpace/plots/Plots_DNS_fromGAN.png"
                    print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename)
                    # print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename, \
                    #         Umin=-1, Umax=1, Vmin=-1, Vmax=1, Pmin=-0.25, Pmax=0.25)

            it = it+1


        # print final residuals
        lr = lr_schedule_mDNS(it)
        tend = time.time()
        print("DNS iterations:  time {0:3e}   step {1:4d}  it {2:6d}  resREC {3:3e} resDNS {4:3e}  resLES {5:3e} lr {6:3e} " \
            .format(tend-tstart, k, it, resREC.numpy(), resDNS.numpy(), resLES.numpy(), lr))

    else:

        # find DNS and LES fields from random input
        if (k>=0):
            z2 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=k)
            w1 = mapping(z2, training=False)



    #-------------- save checkpoint for wl_synthesis and zlatents
    managerCheckpoint_wl.save()
    if (tf.is_tensor(zlatents)):
        np.savez("results_latentSpace/zlatents.npz", zlatents=zlatents.numpy())
    else:
        np.savez("results_latentSpace/zlatents.npz", zlatents=zlatents)



    #-------------- check filtered quantities
    res         = 2**(RES_LOG2-FIL)
    predictions = wl_synthesis([z2, w1], training=False)
    UVP_DNS     = predictions[RES_LOG2-2]
    UVP_LES     = predictions[RES_LOG2-FIL-2]
    delx_LES    = DELX*2**FIL
    dely_LES    = DELY*2**FIL

    if (k==0):
        print("Find LES quantities. Delx_LES is: ", delx_LES)
        

    # DNS fields
    U_DNS = UVP_DNS[0,0,:,:]
    V_DNS = UVP_DNS[0,1,:,:]
    P_DNS = UVP_DNS[0,2,:,:]

    if (TESTCASE=='HW' or TESTCASE=='mHW'):
        P_DNS = P_DNS - tf.reduce_mean(P_DNS)


    # LES fields
    U_LES = UVP_LES[0,0,:,:]
    V_LES = UVP_LES[0,1,:,:]
    P_LES = UVP_LES[0,2,:,:]
    
    if (TESTCASE=='HW' or TESTCASE=='mHW'):    
        P_LES = P_LES - tf.reduce_mean(P_LES)
    

    # filtered DNS fields
    fU_DNS = (filter(U_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
    fV_DNS = (filter(V_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]
    fP_DNS = (filter(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:]

    
    # verify filter properties
    cf_DNS =                               (10.0*U_DNS)[tf.newaxis,tf.newaxis,:,:]  # conservation
    lf_DNS =                            (U_DNS + V_DNS)[tf.newaxis,tf.newaxis,:,:]  # linearity
    df_DNS = ((tr(P_DNS, 1, 0) - tr(P_DNS,-1, 0))/DELX)[tf.newaxis,tf.newaxis,:,:]  # commutative with derivatives
    
    cf_DNS = (filter(cf_DNS))[0,0,:,:]
    lf_DNS = (filter(lf_DNS))[0,0,:,:]
    df_DNS = (filter(df_DNS))[0,0,:,:]
    
    c_LES = 10*U_LES
    l_LES = U_LES + V_LES
    d_LES = (tr((filter(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:], 1, 0) \
           - tr((filter(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:],-1, 0))/delx_LES
    
    
    # print fields
    filename = "results_latentSpace/plots/plots_LES_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(U_LES, V_LES, P_LES, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

    filename = "results_latentSpace/plots/plots_fDNS_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

    filename = "results_latentSpace/plots/plots_diffLES_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(U_LES-fU_DNS, V_LES-fV_DNS, P_LES-fP_DNS, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)


    # print filter properties
    filename = "results_latentSpace/plots/plots_conservation_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(cf_DNS, c_LES, cf_DNS-c_LES, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

    filename = "results_latentSpace/plots/plots_linearity_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(lf_DNS, l_LES, lf_DNS-l_LES, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

    filename = "results_latentSpace/plots/plots_derivatives_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(df_DNS, d_LES, df_DNS-d_LES, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)
    

    # save fields
    # filename = "results_latentSpace/fields/fields_fil_lat_" + str(k) + "_res_" + str(res) + ".npz"
    # save_fields(0, U_LES, V_LES, P_LES, zero_LES, zero_LES, zero_LES, filename)


    # plot spectrum
    filename = "results_latentSpace/energy/energy_spectrum_LES_lat_" + str(k) + "_res_" + str(res) + ".png"
    if (TESTCASE=='HIT_2D'):
        plot_spectrum(U_LES, V_LES, L, filename, label='LES')
    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        gradV_LES = np.sqrt(((cr(V_LES, 1, 0) - cr(V_LES, -1, 0))/(2.0*delx_LES))**2 + ((cr(V_LES, 0, 1) - cr(V_LES, 0, -1))/(2.0*dely_LES))**2)
        plot_spectrum(U_LES, gradV_LES, L, filename, label='LES')




    #-------------- find fields for each layer
    closePlot=False
    for kk in range(RES_LOG2-FIL, RES_LOG2+1):
        UVP_DNS = predictions[kk-2]
        res = 2**kk
        
        delx = DELX*2**(RES_LOG2-kk)
        dely = DELY*2**(RES_LOG2-kk)

        U_DNS_t = UVP_DNS[0, 0, :, :].numpy()
        V_DNS_t = UVP_DNS[0, 1, :, :].numpy()
        P_DNS_t = UVP_DNS[0, 2, :, :].numpy()
        
        if (TESTCASE=='HW' or TESTCASE=='mHW'):
            P_DNS_t = P_DNS_t - tf.reduce_mean(P_DNS_t)

        # find vorticity field
        if (TESTCASE=='HIT_2D'):
            cP_DNS_t = find_vorticity(U_DNS_t, V_DNS_t)
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            # V_DNS_t = sc.ndimage.gaussian_filter(V_DNS_t, 4, mode=['wrap','wrap'])
            # cP_DNS_t = (tr(V_DNS_t, 1, 0) - 2*V_DNS_t + tr(V_DNS_t, -1, 0))/(delx**2) \
            #          + (tr(V_DNS_t, 0, 1) - 2*V_DNS_t + tr(V_DNS_t, 0, -1))/(dely**2)
            cP_DNS_t = (-tr(V_DNS_t, 2, 0) + 16*tr(V_DNS_t, 1, 0) - 30*V_DNS_t + 16*tr(V_DNS_t,-1, 0) - tr(V_DNS_t,-2, 0))/(12*delx**2) \
                     + (-tr(V_DNS_t, 0, 2) + 16*tr(V_DNS_t, 0, 1) - 30*V_DNS_t + 16*tr(V_DNS_t, 0,-1) - tr(V_DNS_t, 0,-2))/(12*dely**2)
            # cP_DNS_t = sc.ndimage.gaussian_filter(cP_DNS_t.numpy(), 1, mode=['wrap','wrap'])
            # cP_DNS_t = tf.convert_to_tensor(cP_DNS_t)
            # cP_DNS_t = cP_DNS_t - tf.reduce_mean(cP_DNS_t)
        
        if (TESTCASE=='HW' or TESTCASE=='mHW'):
            totVorticity = np.sum(P_DNS_t)
            print ("Total vorticity for latent " + str(k) + " is: " + str(totVorticity))


        # print fields        
        filename = "results_latentSpace/plots/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
        print_fields_3(U_DNS_t, V_DNS_t, P_DNS_t, res, filename, TESTCASE, \
                       Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

        filename = "results_latentSpace/plots/plots_VortDiff_" + str(k) + "_res_" + str(res) + ".png"
        print_fields_3(P_DNS_t, cP_DNS_t, P_DNS_t-cP_DNS_t, res, filename, TESTCASE, \
                       Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

        # filename = "results_latentSpace/plots/plot_1field_lat_" + str(k) + "_res_" + str(res) + ".png"
        # print_fields_1(P_DNS_t, filename, legend=False)


        # save fields
        filename = "results_latentSpace/fields/fields_lat_" + str(k) + "_res_" + str(res) + ".npz"
        save_fields(0, U_DNS_t, V_DNS_t, P_DNS_t, zero_DNS, zero_DNS, zero_DNS, filename)


        # find spectrum
        filename = "results_latentSpace/energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".png"

        if (not LOAD_FIELD and kk==RES_LOG2):
            closePlot=True

        if (TESTCASE=='HIT_2D'):
            plot_spectrum(U_DNS_t, V_DNS_t, L, filename, close=closePlot, label=str(res))
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            gradV_DNS = np.sqrt(((cr(V_DNS_t, 1, 0) - cr(V_DNS_t, -1, 0))/(2.0*delx))**2 + ((cr(V_DNS_t, 0, 1) - cr(V_DNS_t, 0, -1))/(2.0*dely))**2)
            plot_spectrum(U_DNS_t, gradV_DNS, L, filename, close=closePlot, label=str(res))
            

    if (LOAD_FIELD):
        filename = "results_latentSpace/energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".png"
        plot_spectrum(U_DNS_org, V_DNS_org, L, filename, label='DNS')
        
    print ("Done for latent " + str(k))
