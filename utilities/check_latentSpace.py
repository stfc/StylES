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
NL          = 1     # number of different latent vectors randomly selected
TUNE_NOISE  = False
LOAD_FIELD  = False       # load field from DNS solver (via restart.npz file)
NITEZ       = 0
CHECK_FIL   = True
RESTART_WL  = False
CHKP_DIR_WL = "./checkpoints_wl"
N_DNS       = 2**RES_LOG2
N_LES       = 2**RES_LOG2-FIL
N2_DNS      = int(N_DNS/2)
N2_LES      = int(N_LES/2)
tollDNS     = 6.0e-4
tollLES     = 1.0e-5
maxitLES    = 100

if (TESTCASE=='HIT_2D'):
    FILE_REAL_PATH  = "../LES_Solvers/fields/"
    from HIT_2D import L
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = 50.176
    FILE_REAL_PATH  = "../bout_interfaces/results_bout/fields/"

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


# set z
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
            FILE_REAL = FILE_REAL_PATH + "fields_time" + tail + ".npz"

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


        # save org fields
        filename = "results_latentSpace/fields_org/fields_lat" + str(k) + "_res" + str(N_DNS) + ".npz"
        save_fields(0, U_DNS_org, V_DNS_org, P_DNS_org, filename=filename)

        # print plots
        filename = "results_latentSpace/plots_org/Plots_DNS_org_" + tail +".png"
        print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N=N_DNS, filename=filename, \
                       Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

        # print spectrum
        filename = "results_latentSpace/energy_org/energy_spectrum_org_" + str(k) + ".png"
        closePlot=True
        if (TESTCASE=='HIT_2D'):
            plot_spectrum(U_DNS_org, V_DNS_org, L, filename, close=closePlot, label='DNS')
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            gradV_DNS_org = np.sqrt(((cr(V_DNS_org, 1, 0) - cr(V_DNS_org, -1, 0))/(2.0*DELX))**2 \
                                  + ((cr(V_DNS_org, 0, 1) - cr(V_DNS_org, 0, -1))/(2.0*DELY))**2)
            plot_spectrum(U_DNS_org, gradV_DNS_org, L, filename, close=closePlot, label='DNS')


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
        if (TESTCASE=='HIT_2D'):
            fP_DNS = tf_find_vorticity(fU_DNS[0,0,:,:], fV_DNS[0,0,:,:])
            fP_DNS = fP_DNS[tf.newaxis,tf.newaxis,:,:] 
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            fP_DNS = filter(P_DNS)
        
        fimgA = tf.concat([fU_DNS, fV_DNS, fP_DNS], 1)

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
                    step_find_zlatents_kDNS(wl_synthesis, filter, opt_kDNS, z, imgA, fimgA, ltv_DNS, INIT_SCAL, typeRes=0)

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
 
            # write losses to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('resREC',   resREC,   step=it)
                tf.summary.scalar('resDNS',   resDNS,   step=it)
                tf.summary.scalar('resLES',   resLES,   step=it)
                tf.summary.scalar('loss_fil', loss_fil, step=it)

            if (it%1000==0):

                # filename = "results_reconstruction/plots/Plots_DNS_fromGAN.png"
                filename = "results_reconstruction/plots/Plots_DNS_fromGAN_" + str(it) + ".png"
                print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename) #, \
                        #Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

                # filename = "results_reconstruction/plots/Plots_LES_fromGAN.png"
                filename = "results_reconstruction/plots/Plots_LES_fromGAN_" + str(it) + ".png"
                print_fields_3(U_LES, V_LES, P_LES, N=N_LES, filename=filename) #, \
                        #Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

                # filename = "results_reconstruction/plots/Plots_fDNS_fromGAN.png"
                filename = "results_reconstruction/plots/Plots_fDNS_fromGAN_" + str(it) + ".png"
                print_fields_3(fU_DNS, fV_DNS, fP_DNS, N=N_LES, filename=filename) #, \
                        #Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)


                # filename = "results_reconstruction/plots/Plots_diffLES_fromGAN.png"
                filename = "results_reconstruction/plots/Plots_diffRecLES_fromGAN_" + str(it) + ".png"
                print_fields_3(P_LES, fimgA[0,2,:,:], P_LES-fimgA[0,2,:,:], N=N_LES, filename=filename, diff=True) #, \
                        #Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

                # filename = "results_reconstruction/plots/Plots_diffLES_fromGAN.png"
                filename = "results_reconstruction/plots/Plots_diffRecDNS_fromGAN_" + str(it) + ".png"
                print_fields_3(fP_DNS, fimgA[0,2,:,:], fP_DNS-fimgA[0,2,:,:], N=N_LES, filename=filename, diff=True) #, \
                        #Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

                # filename = "results_reconstruction/plots/Plots_diffLES_fromGAN.png"
                filename = "results_reconstruction/plots/Plots_diffLES_fromGAN_" + str(it) + ".png"
                print_fields_3(P_LES, fP_DNS, P_LES-fP_DNS, N=N_LES, filename=filename, diff=True) #, \
                        #Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

            it = it+1


        # print final residuals
        tend = time.time()
        print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fil {6:3e} " \
            .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))

    else:

        # find DNS and LES fields from random input
        if (k>0):
            # set z
            z = tf.random.uniform([BATCH_SIZE, 4, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)



    #--------------------------------------------- check filtered quantities

    # find delx_LES and dely_LES
    delx_LES    = DELX*2**FIL
    dely_LES    = DELY*2**FIL
    if (k==0):
        print("Find LES quantities. Delx_LES is: ", delx_LES)


    #-------- get fields
    res = 2**(RES_LOG2-FIL)

    UVP_DNS, UVP_LES, fUVP_DNS, _, predictions = find_predictions(wl_synthesis, filter, z, INIT_SCAL)
    resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, UVP_DNS, UVP_LES, typeRes=0)
    
    if (CHECK_FIL):
        # DNS fields
        U_DNS = UVP_DNS[0,0,:,:]
        V_DNS = UVP_DNS[0,1,:,:]
        P_DNS = UVP_DNS[0,2,:,:]

        # LES fields
        U_LES = UVP_LES[0,0,:,:]
        V_LES = UVP_LES[0,1,:,:]
        P_LES = UVP_LES[0,2,:,:]
            
        # imgA fields
        fU_DNS   = (filter(U_DNS[tf.newaxis,tf.newaxis,:,:]))
        fV_DNS   = (filter(V_DNS[tf.newaxis,tf.newaxis,:,:]))
        fP_DNS   = (filter(P_DNS[tf.newaxis,tf.newaxis,:,:]))
        fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], 1)  
        fU_DNS   = fUVP_DNS[0,0,:,:]
        fV_DNS   = fUVP_DNS[0,1,:,:]
        fP_DNS   = fUVP_DNS[0,2,:,:]   


        #-------- verify filter properties
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
        

        #-------- plots
        # fields
        filename = "results_latentSpace/plots/plots_LES_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(U_LES, V_LES, P_LES, N=res, filename=filename, testcase=TESTCASE, \
                    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

        filename = "results_latentSpace/plots/plots_fDNS_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(fU_DNS, fV_DNS, fP_DNS, N=res, filename=filename, testcase=TESTCASE, \
                    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

        filename = "results_latentSpace/plots/plots_diffLES_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(U_LES-fU_DNS, V_LES-fV_DNS, P_LES-fP_DNS, N=res, filename=filename, testcase=TESTCASE, \
                    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)


        # filter properties
        filename = "results_latentSpace/plots/plots_conservation_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(cf_DNS, c_LES, cf_DNS-c_LES, N=res, filename=filename, testcase=TESTCASE, \
                    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

        filename = "results_latentSpace/plots/plots_linearity_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(lf_DNS, l_LES, lf_DNS-l_LES, N=res, filename=filename, testcase=TESTCASE, \
                    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

        filename = "results_latentSpace/plots/plots_derivatives_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(df_DNS, d_LES, df_DNS-d_LES, N=res, filename=filename, testcase=TESTCASE, \
                    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)


        # spectrum
        filename = "results_latentSpace/energy/energy_spectrum_LES_lat" + str(k) + "_res" + str(res) + ".png"
        if (TESTCASE=='HIT_2D'):
            plot_spectrum(U_LES, V_LES, L, filename, close=True, label='LES')
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            gradV_LES = np.sqrt(((cr(V_LES, 1, 0) - cr(V_LES, -1, 0))/(2.0*delx_LES))**2 + ((cr(V_LES, 0, 1) - cr(V_LES, 0, -1))/(2.0*dely_LES))**2)
            plot_spectrum(U_LES, gradV_LES, L, filename, close=True, label='LES')




    #--------------------------------------------- find fields for each layer
    closePlot=False
    for kk in range(RES_LOG2-FIL, RES_LOG2+1):
        UVP_DNS = predictions[kk-2]*INIT_SCAL
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
            # cP_DNS =tf.convert_to_tensor(cP_DNS)

        # plot fields        
        filename = "results_latentSpace/plots/plots_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(U_DNS, V_DNS, P_DNS, N=res, filename=filename, testcase=TESTCASE, \
                       Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)

        filename = "results_latentSpace/plots/plots_VortDiff_" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, N=res, filename=filename, testcase=TESTCASE, \
                       Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)

        # filename = "results_latentSpace/plots/plot_1field_lat" + str(k) + "_res" + str(res) + ".png"
        # print_fields_1(P_DNS, filename, legend=False)

        # save fields
        filename = "results_latentSpace/fields/fields_lat"  + str(k) + "_res" + str(res) + ".npz"
        save_fields(0, U_DNS, V_DNS, P_DNS, filename=filename)

        # find spectrum
        if (kk==RES_LOG2 and (not LOAD_FIELD)):
            closePlot=True
        filename = "results_latentSpace/energy/energy_spectrum_lat" + str(k) + "_res" + str(res) + ".png"
        if (TESTCASE=='HIT_2D'):
            plot_spectrum(U_DNS, V_DNS, L, filename, close=closePlot, label=str(res))
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            gradV_DNS = np.sqrt(((cr(V_DNS, 1, 0) - cr(V_DNS, -1, 0))/(2.0*delx))**2 + ((cr(V_DNS, 0, 1) - cr(V_DNS, 0, -1))/(2.0*dely))**2)
            plot_spectrum(U_DNS, gradV_DNS, L, filename, close=closePlot, label=str(res))


    # compare with target file
    if (LOAD_FIELD):
        closePlot=True
        filename = "results_latentSpace/energy/energy_spectrum_lat" + str(k) + "_res" + str(res) + ".png"
        if (TESTCASE=='HIT_2D'):
            plot_spectrum(U_DNS_org, V_DNS_org, L, filename, close=closePlot, label='DNS')
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            gradV_DNS_org = np.sqrt(((cr(V_DNS_org, 1, 0) - cr(V_DNS_org, -1, 0))/(2.0*DELX))**2 \
                                + ((cr(V_DNS_org, 0, 1) - cr(V_DNS_org, 0, -1))/(2.0*DELY))**2)
            plot_spectrum(U_DNS_org, gradV_DNS_org, L, filename, close=closePlot, label='DNS')
        
    print ("Done for latent " + str(k))
