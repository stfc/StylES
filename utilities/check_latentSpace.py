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
NL          = 1       # number of different latent vectors randomly selected. Set to 1 for LOAD_FIELD=True
TUNE_NOISE  = True    # tune noise
LOAD_FIELD  = False   # load field from DNS solver (via restart.npz file)
INIT_SCAL   = 10      # initial scaling if a field is not loaded 
RESTART_WL  = False
CHKP_DIR_WL = "./checkpoints_wl"
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2_FIL
N2_DNS      = int(N_DNS/2)
N2_LES      = int(N_LES/2)
tollDNS     = 1.0e-3

if (TESTCASE=='HIT_2D'):
    FILE_PATH  = "../LES_Solvers/fields/"
    from HIT_2D import L, N
elif (TESTCASE=='HW'):
    FILE_PATH  = "../bout_interfaces/results_bout/fields/"   # to do: make more general
    L = 50.176
elif (TESTCASE=='mHW'):
    FILE_PATH  = "../../../data/Fields/mHW/fields_N512/"   # to do: make more general
    L = 50.176

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
os.system("rm -rf logs")

os.system("mkdir -p results_latentSpace/plots")
os.system("mkdir -p results_latentSpace/fields")
os.system("mkdir -p results_latentSpace/energy")
os.system("mkdir -p results_latentSpace/plots_org/")
os.system("mkdir -p results_latentSpace/fields_org")
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



# create variable synthesis model
layer_mDNS = layer_latent_mDNS()
layer_mLES = layer_latent_mLES()

z_in         = tf.keras.Input(shape=([LATENT_SIZE]), dtype=DTYPE)
w_in         = mapping(z_in)
w_DNS        = layer_mDNS(w_in)
w_LES        = layer_mLES(w_DNS)
outputs      = synthesis(w_LES, training=False)
wl_synthesis = tf.keras.Model(inputs=z_in, outputs=outputs)


# define optimizer for mDNS search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_mDNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_mDNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_mDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_mDNS)


# define checkpoints wl_synthesis and filter
checkpoint_wl = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# set trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []

ltv_mDNS = ltv_DNS
for variable in layer_mDNS.trainable_variables:
    ltv_mDNS.append(variable)

print("\n ltv_mDNS variables:")
for variable in ltv_mDNS:
    print(variable.name, variable.shape)




#------------------------------------------- loop over different latent spaces
for k in range(NL):
    
    # load initial flow
    if (LOAD_FIELD):

        # load initial flow
        if (TESTCASE=='HIT_2D'):
            tail = str(int(k*100+100))
            FILE_REAL = FILE_PATH + "fields_run0_it" + tail + ".npz"

        if (TESTCASE=='HW'):
            tail = str(int(k)).zfill(3)
            FILE_REAL = FILE_PATH + "fields_time" + tail + ".npz"

        if (TESTCASE=='mHW'):
            tail = str(int(k+200))
            FILE_REAL = FILE_PATH + "fields_run1000_time" + tail + ".npz"


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


        z0 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)
        w0 = mapping(z0, training=False)
        
    else:

        if (RESTART_WL):

            # loading wl_synthesis checkpoint and zlatents
            if managerCheckpoint_wl.latest_checkpoint:
                print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
            else:
                print("Initializing wl_synthesis from scratch.")
            data      = np.load("results_latentSpace/z0.npz")
            z0        = data["z0"]
            mDNS      = data["mDNS"]
            noise_DNS = data["noise_DNS"]

            # convert to TensorFlow tensors            
            z0        = tf.convert_to_tensor(z0)
            mDNS      = tf.convert_to_tensor(mDNS)
            noise_DNS = tf.convert_to_tensor(noise_DNS)

            # assign mDNS
            layer_mDNS.trainable_variables[0].assign(mDNS)

            # assign variable noise
            it=0
            for layer in synthesis.layers:
                if "layer_noise_constants" in layer.name:
                    layer.trainable_variables[0].assign(noise_DNS[it])
                    it=it+1
        else:
            # find DNS and LES fields from random input
            z0 = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)        

        predictions = wl_synthesis(z0, training=False)
        UVP_DNS     = predictions[RES_LOG2-2]*INIT_SCAL
        U_DNS_org   = UVP_DNS[0,0,:,:]
        V_DNS_org   = UVP_DNS[0,1,:,:]
        P_DNS_org   = UVP_DNS[0,2,:,:]
        tail        = str(k).zfill(3)




    # find vorticity
    if (TESTCASE=='HIT_2D'):
        P_DNS_org  = find_vorticity(U_DNS_org, V_DNS_org)
        cP_DNS_org = find_vorticity(U_DNS_org, V_DNS_org)
    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        # cP_DNS_org = (tr(V_DNS_org, 1, 0) - 2*V_DNS_org + tr(V_DNS_org, -1, 0))/(DELX**2) \
        #            + (tr(V_DNS_org, 0, 1) - 2*V_DNS_org + tr(V_DNS_org, 0, -1))/(DELY**2)
        cP_DNS_org = (-tr(V_DNS_org, 2, 0) + 16*tr(V_DNS_org, 1, 0) - 30*V_DNS_org + 16*tr(V_DNS_org,-1, 0) - tr(V_DNS_org,-2, 0))/(12*DELX**2) \
                    + (-tr(V_DNS_org, 0, 2) + 16*tr(V_DNS_org, 0, 1) - 30*V_DNS_org + 16*tr(V_DNS_org, 0,-1) - tr(V_DNS_org, 0,-2))/(12*DELY**2)


    # plots fields
    filename = "results_latentSpace/plots_org/Plots_DNS_org_" + tail +".png"
    print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N_DNS, filename, \
                    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

    filename = "results_latentSpace/plots_org/Plots_DNS_org_diffVort_" + tail +".png"
    print_fields_3(P_DNS_org, cP_DNS_org, P_DNS_org-cP_DNS_org, N_DNS, filename, \
                    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

    filename = "results_latentSpace/energy_org/energy_spectrum_org_" + str(k) + ".png"
    plot_spectrum(U_DNS_org, V_DNS_org, L, filename, label='DNS')

    filename = "results_latentSpace/fields_org/fields_lat" + str(k) + "_res" + str(N_DNS) + ".npz"
    save_fields(0, U_DNS_org, V_DNS_org, P_DNS_org, filename=filename)


    # normalize
    U_min = np.min(U_DNS_org)
    U_max = np.max(U_DNS_org)
    V_min = np.min(V_DNS_org)
    V_max = np.max(V_DNS_org)
    P_min = np.min(P_DNS_org)
    P_max = np.max(P_DNS_org)

    UVP_minmax = np.asarray([U_min, U_max, V_min, V_max, P_min, P_max])
    UVP_minmax = tf.convert_to_tensor(UVP_minmax)        

    U_DNS = 2.0*(U_DNS_org - U_min)/(U_max - U_min) - 1.0
    V_DNS = 2.0*(V_DNS_org - V_min)/(V_max - V_min) - 1.0
    P_DNS = 2.0*(P_DNS_org - P_min)/(P_max - P_min) - 1.0
    


    #-------------- preprare targets
    U_DNS = U_DNS[np.newaxis,np.newaxis,:,:]
    V_DNS = V_DNS[np.newaxis,np.newaxis,:,:]
    P_DNS = P_DNS[np.newaxis,np.newaxis,:,:]

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
    resLES = large
    tstart = time.time()
    while (resLES>tollDNS and it<lr_DNS_maxIt):

        resLES, loss_fil, UVP_DNS, UVP_LES, fUVP_DNS = step_find_latents_mDNS(wl_synthesis, filter, opt_mDNS, z0, ltv_mDNS)

        # print residuals and fields
        if (it%100==0):

            lr = lr_schedule_mDNS(it)
        
            # print residuals
            tend = time.time()
            print("DNS iterations:  time {0:3e}   step {1:4d}  it {2:6d}  resLES {3:3e} lr {4:3e} " \
                .format(tend-tstart, k, it, resLES.numpy(), lr))

            # write losses to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('resLES',   resLES,   step=it)                    
                tf.summary.scalar('lr',       lr,       step=it)

            if (it%1000==0):

                # separate DNS fields from GAN
                UVP_DNS = rescale(UVP_DNS, UVP_minmax)

                U_DNS = UVP_DNS[0, 0, :, :].numpy()
                V_DNS = UVP_DNS[0, 1, :, :].numpy()
                P_DNS = UVP_DNS[0, 2, :, :].numpy()

                filename = "results_latentSpace/plots/Plots_DNS_fromGAN_" + str(it).zfill(5) + ".png"
                # filename = "results_latentSpace/plots/Plots_DNS_fromGAN.png"
                print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename)
                # print_fields_3(U_DNS, V_DNS, P_DNS, N_DNS, filename, \
                #         Umin=-1, Umax=1, Vmin=-1, Vmax=1, Pmin=-0.25, Pmax=0.25)

        it = it+1


    # print final residuals
    lr = lr_schedule_mDNS(it)
    tend = time.time()
    print("DNS iterations:  time {0:3e}   step {1:4d}  it {2:6d}  resLES {3:3e} lr {4:3e} " \
        .format(tend-tstart, k, it, resLES.numpy(), lr))


    # save checkpoint for wl_synthesis and zlatents
    if (not RESTART_WL):
        managerCheckpoint_wl.save()

        # find z0
        z0   = z0.numpy()

        # find mDNS
        mDNS = layer_mDNS.trainable_variables[0].numpy()

        # find noise_DNS
        it=0
        noise_DNS=[]
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                noise_DNS.append(layer.trainable_variables[0].numpy())
                        
        np.savez("results_latentSpace/z0.npz", z0=z0, mDNS=mDNS, noise_DNS=noise_DNS)


    #-------------- check filtered quantities
    res         = 2**(RES_LOG2-FIL)
    predictions = wl_synthesis(z0, training=False)
    UVP_DNS     = predictions[RES_LOG2-2]
    UVP_LES     = predictions[RES_LOG2-FIL-2]
    
    UVP_DNS     = rescale(UVP_DNS, UVP_minmax)
    UVP_LES     = rescale(UVP_LES, UVP_minmax)
    
    delx_LES    = DELX*2**FIL
    dely_LES    = DELY*2**FIL

    if (k==0):
        print("Find LES quantities. Delx_LES is: ", delx_LES)
        
    # DNS fields
    U_DNS = UVP_DNS[0,0,:,:]
    V_DNS = UVP_DNS[0,1,:,:]
    P_DNS = UVP_DNS[0,2,:,:]

    # LES fields
    U_LES = UVP_LES[0,0,:,:]
    V_LES = UVP_LES[0,1,:,:]
    P_LES = UVP_LES[0,2,:,:]
        
    #------------- filtered DNS fields
    # normalize
    U_DNS = normalize_sc(U_DNS)
    V_DNS = normalize_sc(V_DNS)
    P_DNS = normalize_sc(P_DNS)
    
    # filter
    fU_DNS = (filter(U_DNS[tf.newaxis,tf.newaxis,:,:]))
    fV_DNS = (filter(V_DNS[tf.newaxis,tf.newaxis,:,:]))
    fP_DNS = (filter(P_DNS[tf.newaxis,tf.newaxis,:,:]))

    # concatenate
    fUVP_DNS  = tf.concat([fU_DNS, fV_DNS, fP_DNS], 1)  
    
    # rescale
    fUVP_DNS = rescale(fUVP_DNS, UVP_minmax)
    
    # separate fields
    fU_DNS = fUVP_DNS[0,0,:,:]
    fV_DNS = fUVP_DNS[0,1,:,:]
    fP_DNS = fUVP_DNS[0,2,:,:]   


    #-------- verify filter properties
    cf_DNS =                               (10.0*U_DNS)[tf.newaxis,tf.newaxis,:,:]  # conservation
    lf_DNS =                            (U_DNS + V_DNS)[tf.newaxis,tf.newaxis,:,:]  # linearity
    df_DNS = ((tr(P_DNS, 1, 0) - tr(P_DNS,-1, 0))/DELX)[tf.newaxis,tf.newaxis,:,:]  # commutative with derivatives
    
    # normalize
    cf_DNS = normalize_sc(cf_DNS)
    lf_DNS = normalize_sc(lf_DNS)
    df_DNS = normalize_sc(df_DNS)
    
    cf_DNS = (filter(cf_DNS))[0,0,:,:]
    lf_DNS = (filter(lf_DNS))[0,0,:,:]
    df_DNS = (filter(df_DNS))[0,0,:,:]
    
    c_LES = 10*U_LES
    l_LES = U_LES + V_LES
    d_LES = (tr((filter(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:], 1, 0) \
           - tr((filter(P_DNS[tf.newaxis,tf.newaxis,:,:]))[0,0,:,:],-1, 0))/delx_LES
    
    # normalize
    c_LES = normalize_sc(c_LES)
    l_LES = normalize_sc(l_LES)
    d_LES = normalize_sc(d_LES)


    #-------- plots
    # fields
    filename = "results_latentSpace/plots/plots_LES_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(U_LES, V_LES, P_LES, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

    filename = "results_latentSpace/plots/plots_fDNS_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(fU_DNS, fV_DNS, fP_DNS, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

    filename = "results_latentSpace/plots/plots_diffLES_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(U_LES-fU_DNS, V_LES-fV_DNS, P_LES-fP_DNS, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

    # filter properties
    filename = "results_latentSpace/plots/plots_conservation_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(cf_DNS, c_LES, cf_DNS-c_LES, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

    filename = "results_latentSpace/plots/plots_linearity_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(lf_DNS, l_LES, lf_DNS-l_LES, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

    filename = "results_latentSpace/plots/plots_derivatives_lat_" + str(k) + "_res_" + str(res) + ".png"
    print_fields_3(df_DNS, d_LES, df_DNS-d_LES, res, filename, TESTCASE, \
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)


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

        UVP_DNS = rescale(UVP_DNS, UVP_minmax)
    
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
            # cP_DNS =tf.convert_to_tensor(cP_DNS)

        # plot fields        
        filename = "results_latentSpace/plots/plots_lat_" + str(k) + "_res_" + str(res) + ".png"
        print_fields_3(U_DNS, V_DNS, P_DNS, res, filename, TESTCASE, \
                       Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

        filename = "results_latentSpace/plots/plots_VortDiff_" + str(k) + "_res_" + str(res) + ".png"
        print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, res, filename, TESTCASE, \
                       Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

        # filename = "results_latentSpace/plots/plot_1field_lat_" + str(k) + "_res_" + str(res) + ".png"
        # print_fields_1(P_DNS, filename, legend=False)

        # save fields
        filename = "results_latentSpace/fields/fields_lat"  + str(k) + "_res" + str(res) + ".npz"
        save_fields(0, U_DNS, V_DNS, P_DNS, filename=filename)

        # find spectrum
        filename = "results_latentSpace/energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".png"
        if (TESTCASE=='HIT_2D'):
            plot_spectrum(U_DNS, V_DNS, L, filename, close=closePlot, label=str(res))
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            gradV_DNS = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*delx))**2 + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*dely))**2)
            plot_spectrum(U_DNS, gradV_DNS, L, filename, close=closePlot, label=str(res))

    closePlot=True
    filename = "results_latentSpace/energy/energy_spectrum_lat_" + str(k) + "_res_" + str(res) + ".png"
    plot_spectrum(U_DNS_org, V_DNS_org, L, filename, close=closePlot, label='DNS')
        
    print ("Done for latent " + str(k))
