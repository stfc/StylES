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
RESTART_WL  = True
CHKP_DIR_WL = "./checkpoints_wl"
N_DNS       = 2**RES_LOG2
N_LES       = 2**(RES_LOG2-FIL)
N2L         = int(N_LES/2)
RS          = int(2**FIL)
tollDNS     = 1.0e-3

if (TESTCASE=='HIT_2D'):
    FILE_REAL_PATH  = "../LES_Solvers/fields/"
    from HIT_2D import L
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    L = 50.176
    FILE_REAL_PATH  = "../bout_interfaces/results/fields/"

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


# loading StyleGAN checkpoint
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=1)
checkpoint.restore(managerCheckpoint.latest_checkpoint)

if managerCheckpoint.latest_checkpoint:
    print("StyleGAN restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing StyleGAN from scratch.")


# create variable synthesis model
layer_kDNS = layer_zlatent_kDNS()
z_in         = tf.keras.Input(shape=([2*(G_LAYERS-M_LAYERS)+1, LATENT_SIZE]), dtype=DTYPE)
img_in       = tf.keras.Input(shape=([NUM_CHANNELS, 2**(RES_LOG2-FIL), 2**(RES_LOG2-FIL)]), dtype=DTYPE)
w            = layer_kDNS(mapping, z_in)
outputs      = synthesis([w,img_in], training=False)
wl_synthesis = tf.keras.Model(inputs=[z_in, img_in], outputs=[outputs, w])


# create filter model
x_in    = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)
out     = gaussian_filter(x_in[0,0,:,:], rs=RS, rsca=RS)
gfilter = tf.keras.Model(inputs=x_in, outputs=out)


# define optimizer for DNS search
if (lr_DNS_POLICY=="EXPONENTIAL"):
    lr_schedule_DNS  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_DNS,
        decay_steps=lr_DNS_STEP,
        decay_rate=lr_DNS_RATE,
        staircase=lr_DNS_EXP_ST)
elif (lr_DNS_POLICY=="PIECEWISE"):
    lr_schedule_DNS = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_DNS_BOUNDS, lr_DNS_VALUES)
opt_kDNS = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_DNS)


# define checkpoints wl_synthesis and filter
checkpoint_wl = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)


# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []

for variable in layer_kDNS.trainable_variables:
    ltv_DNS.append(variable)

print("\n DNS variables:")
for variable in ltv_DNS:
    print(variable.name)

time.sleep(3)


# set z
if (RESTART_WL):
    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")

    if (TESTCASE=='HIT_2D'):
        data      = np.load("results_latentSpace/z0.npz")
    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        data      = np.load("../bout_interfaces/restart_fromGAN/z0.npz")

    z0        = data["z0"]
    kDNS      = data["kDNS"]
    noise_DNS = data["noise_DNS"]

    # convert to TensorFlow tensors            
    z0        = tf.convert_to_tensor(z0, dtype=DTYPE)
    kDNS      = layer_kDNS.trainable_variables[0].numpy()

    # assign kDNS
    layer_kDNS.trainable_variables[0].assign(kDNS)

    # assign variable noise
    if (TUNE_NOISE):
        noise_DNS = tf.convert_to_tensor(noise_DNS, dtype=DTYPE)
        it=0
        for layer in synthesis.layers:
            if "layer_noise_constants" in layer.name:
                layer.trainable_variables[0].assign(noise_DNS[it])
                it=it+1
else:
    # set z
    z0 = tf.random.uniform(shape=[BATCH_SIZE, 2*(G_LAYERS-M_LAYERS)+1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)


#-------------- load DNS field
# load numpy array
U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_DNS)
U_DNS = np.cast[DTYPE](U_DNS)
V_DNS = np.cast[DTYPE](V_DNS)
P_DNS = np.cast[DTYPE](P_DNS)

# convert to tensor
U_DNS = tf.convert_to_tensor(U_DNS, dtype=DTYPE)
V_DNS = tf.convert_to_tensor(V_DNS, dtype=DTYPE)
P_DNS = tf.convert_to_tensor(P_DNS, dtype=DTYPE)

# save original DNS
U_DNS_org = tf.identity(U_DNS)
V_DNS_org = tf.identity(V_DNS)
P_DNS_org = tf.identity(P_DNS)

U_DNS_org = U_DNS_org[tf.newaxis,tf.newaxis,:,:]
V_DNS_org = V_DNS_org[tf.newaxis,tf.newaxis,:,:]
P_DNS_org = P_DNS_org[tf.newaxis,tf.newaxis,:,:]

imgA = tf.concat([U_DNS_org, V_DNS_org, P_DNS_org], axis=1)




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

    



# find multiplier for DNS field
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


kUmax = ( fnV[N2L,N2L]*nfU[N2L,N2L])/(fnU[N2L,N2L]*nfV[N2L,N2L])*fU_amax*nV_amax/fV_amax
kVmax = ( fnU[N2L,N2L]*nfV[N2L,N2L])/(fnV[N2L,N2L]*nfU[N2L,N2L])*fV_amax*nU_amax/fU_amax
kPmax = ( fnU[N2L,N2L]*nfP[N2L,N2L])/(fnP[N2L,N2L]*nfU[N2L,N2L])*fP_amax*nU_amax/fU_amax

#verify multipliers
if (abs((kUmax-nU_amax) + (kVmax-nV_amax) + (kPmax-nP_amax))>1.e-4):
    print("Diff on kUmax =", kUmax - nP_amax)
    print("Diff on kUmax =", kVmax - nV_amax)
    print("Diff on kUmax =", kPmax - nP_amax)

    print("Mismatch in the filter properties!!!")
    exit(0)
else:
    print("Diff on kUmax =", kUmax - nU_amax)
    print("Diff on kUmax =", kVmax - nV_amax)
    print("Diff on kUmax =", kPmax - nP_amax)

    UVP_max = [kUmax, kVmax, kPmax]



#------------------------------------------- loop over different latent spaces
for k in range(NL):
    
    # load initial flow
    tf.random.set_seed(k)
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

        U_DNS = U_DNS/U_norm
        V_DNS = V_DNS/V_norm
        P_DNS = P_DNS/P_norm
        
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
        fU_DNS = gfilter(U_DNS)
        fV_DNS = gfilter(V_DNS)
        fP_DNS = gfilter(P_DNS)

        fimgA = tf.concat([fU_DNS, fV_DNS, fP_DNS], 1)
       
        # start research on the latent space
        it     = 0
        resREC = large
        tstart = time.time()
        while (resREC>tollDNS and it<lr_DNS_maxIt):

            lr = lr_schedule_DNS(it)
            UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, _, _ = \
                step_find_zlatents_kDNS(wl_synthesis, gfilter, opt_kDNS, z0, imgA, fimgA, ltv_DNS, typeRes=0)

            kDNS  = layer_kDNS.trainable_variables[0]
            kDNSn = tf.clip_by_value(kDNS, 0, 1)
            if (tf.reduce_any((kDNS-kDNSn)>0)):
                layer_kDNS.trainable_variables[0].assign(kDNSn)
                UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, gfilter, z0)
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
                    print_fields_3(U_DNS, U_DNS_org, U_DNS-U_DNS_org, N=N_DNS, filename=filename, testcase=TESTCASE, diff=True, \
                        Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)
                    
                    filename = "results_latentSpace/plots/Plots_DNS_VfromGAN_" + str(it).zfill(6) + ".png"
                    print_fields_3(V_DNS, V_DNS_org, V_DNS-V_DNS_org, N=N_DNS, filename=filename, testcase=TESTCASE, diff=True, \
                        Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)
                    
                    filename = "results_latentSpace/plots/Plots_DNS_PfromGAN_" + str(it).zfill(6) + ".png"
                    print_fields_3(P_DNS, P_DNS_org, P_DNS-P_DNS_org, N=N_DNS, filename=filename, testcase=TESTCASE, diff=True, \
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
            z0 = tf.random.uniform(shape=[BATCH_SIZE, 2*(G_LAYERS-M_LAYERS)+1, LATENT_SIZE], minval=MINVALRAN, maxval=MAXVALRAN, dtype=DTYPE, seed=SEED_RESTART)



    #--------------------------------------------- check filtered quantities

    # find delx_LES and dely_LES
    delx_LES    = DELX*2**FIL
    dely_LES    = DELY*2**FIL
    if (k==0):
        print("Find LES quantities. Delx_LES is: ", delx_LES)


    #-------- get fields
    res = 2**(RES_LOG2-FIL)
    UVP_DNS, UVP_LES, fUVP_DNS, _, predictions = find_predictions(wl_synthesis, gfilter, [z0, fimgA], UVP_max)
      
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
                  Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, diff=True)


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
    print_fields_3(cf_DNS, c_LES, cf_DNS-c_LES, N=res, filename=filename, testcase=TESTCASE, diff=True)

    filename = "results_latentSpace/plots/plots_linearity_lat" + str(k) + "_res" + str(res) + ".png"
    print_fields_3(lf_DNS, l_LES, lf_DNS-l_LES, N=res, filename=filename, testcase=TESTCASE, diff=True)

    filename = "results_latentSpace/plots/plots_derivatives_lat" + str(k) + "_res" + str(res) + ".png"
    print_fields_3(df_DNS, d_LES, df_DNS-d_LES, N=res, filename=filename, testcase=TESTCASE, diff=True)


    # spectrum
    filename = "results_latentSpace/energy/energy_spectrum_LES_lat" + str(k) + "_res" + str(res) + ".png"
    if (TESTCASE=='HIT_2D'):
        plot_spectrum(fU_DNS, fV_DNS, L, filename, close=True, label='LES')
    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        gradV_LES = np.sqrt(((cr(fV_DNS, 1, 0) - cr(fV_DNS, -1, 0))/(2.0*delx_LES))**2 \
                  + ((cr(fV_DNS, 0, 1) - cr(fV_DNS, 0, -1))/(2.0*dely_LES))**2)
        plot_spectrum(fU_DNS, gradV_LES, L, filename, close=False, label='LES')

    filename = "results_latentSpace/energy/energy_spectrum_DNS_lat" + str(k) + "_res" + str(res) + ".png"
    if (TESTCASE=='HIT_2D'):
        plot_spectrum(U_DNS_org[0,0,:,:], V_DNS_org[0,0,:,:], L, filename, close=closePlot, label='DNS')
    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        gradV_DNS_org = np.sqrt(((cr(V_DNS_org[0,0,:,:], 1, 0) - cr(V_DNS_org[0,0,:,:], -1, 0))/(2.0*DELX))**2 \
                              + ((cr(V_DNS_org[0,0,:,:], 0, 1) - cr(V_DNS_org[0,0,:,:], 0, -1))/(2.0*DELY))**2)
        plot_spectrum(U_DNS_org[0,0,:,:], gradV_DNS_org, L, filename, close=True, label='DNS')




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
            # cP_DNS =tf.convert_to_tensor(cP_DNS)

        # plot fields        
        filename = "results_latentSpace/plots/plots_lat" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(U_DNS, V_DNS, P_DNS, N=res, filename=filename, testcase=TESTCASE)

        if (TESTCASE=='HIT_2D'):
            filename = "results_latentSpace/plots/plots_VortDiff_" + str(k) + "_res" + str(res) + ".png"
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            filename = "results_latentSpace/plots/plots_PhiDiff_" + str(k) + "_res" + str(res) + ".png"
        print_fields_3(P_DNS, cP_DNS, P_DNS-cP_DNS, N=res, filename=filename, testcase=TESTCASE, diff=True)

        # filename = "results_latentSpace/plots/plot_1field_lat" + str(k) + "_res" + str(res) + ".png"
        # print_fields_1(P_DNS, filename, legend=False)

        # save fields
        filename = "results_latentSpace/fields/fields_lat"  + str(k) + "_res" + str(res) + ".npz"
        save_fields(0, U_DNS, V_DNS, P_DNS, filename=filename)
        
    print ("Done for latent " + str(k))
