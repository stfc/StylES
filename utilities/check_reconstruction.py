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
from HIT_2D import L
from math import floor, log10
from matplotlib.ticker import FuncFormatter

os.chdir('../')
from MSG_StyleGAN_tf2 import *
from IO_functions import StyleGAN_load_fields
from functions    import gaussian_kernel
os.chdir('./utilities')




#------------------------------------------- set local parameters
TUNE_NOISE    = False
NITEZ         = 0   # number of attempts to find a closer z. When restart from a GAN field, use NITEZ=0
RESTART_WL    = False
RELOAD_FREQ   = 100000
CHKP_DIR_WL   = "./checkpoints_wl"
N_DNS         = 2**RES_LOG2
N_LES         = 2**(RES_LOG2-FIL)
N2_DNS        = int(N_DNS/2)
N2_LES        = int(N_LES/2)
tollLESValues = [1.0e-3]
maxitLES      = 1000000

if (TESTCASE=='HIT_2D'):
    FILE_REAL_PATH = "../LES_Solvers/fields/"
    NL             = 101     # number of different latent vectors randomly selected
    t0_label       = 'step 0'
    tf_label       = 'step 10k'
    Z0_DIR_WL      = "../LES_Solvers/restart_fromGAN/"
    from HIT_2D import L
elif (TESTCASE=='HW' or TESTCASE=='mHW'):
    FILE_REAL_PATH  = "../bout_interfaces/results_bout/fields/"
    NL              = 1001     # number of different latent vectors randomly selected
    t0_label        = r'200 $\omega^{-1}_{ci}$'
    tf_label        = r'10000 $\omega^{-1}_{ci}$'
    L               = 50.176
    Z0_DIR_WL      = "../bout_interfaces/restart_fromGAN/"

DELX = L/N_DNS
DELY = L/N_DNS


#------------------------------------------- initialization
os.system("rm -rf results_reconstruction/plots")
os.system("rm -rf results_reconstruction/fields")
os.system("rm -rf results_reconstruction/energy")
os.system("rm -rf results_reconstruction/plots_org")
os.system("rm -rf results_reconstruction/fields_org")
os.system("rm -rf results_reconstruction/energy_org")
os.system("rm -rf logs")

os.system("mkdir -p results_reconstruction/plots")
os.system("mkdir -p results_reconstruction/fields")
os.system("mkdir -p results_reconstruction/energy")
os.system("mkdir -p results_reconstruction/plots_org/")
os.system("mkdir -p results_reconstruction/fields_org")
os.system("mkdir -p results_reconstruction/energy_org")


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





#---------------------------------------------- functions---------------------------------
def kilos(x, pos):
    'The two args are the value and tick position'
    return '%1dk' % (x*1e-3)



#------------------------------------------- loop over all tollerances
ltoll = len(tollLESValues)
totTime = np.zeros((NL), dtype=DTYPE)

velx_DNS = np.zeros((ltoll,2,NL), dtype=DTYPE)
vely_DNS = np.zeros((ltoll,2,NL), dtype=DTYPE)
vort_DNS = np.zeros((ltoll,2,NL), dtype=DTYPE)

velx_LES = np.zeros((ltoll,2,NL), dtype=DTYPE)
vely_LES = np.zeros((ltoll,2,NL), dtype=DTYPE)
vort_LES = np.zeros((ltoll,2,NL), dtype=DTYPE)

c_velx = np.zeros((2,2,N_DNS), dtype=DTYPE)
c_vely = np.zeros((2,2,N_DNS), dtype=DTYPE)
c_vort = np.zeros((2,2,N_DNS), dtype=DTYPE)

spectra = np.zeros((2,2,2,N_DNS), dtype=DTYPE)

fig, axs = plt.subplots(1, 3, figsize=(20,10))
fig.subplots_adjust(hspace=0.25)
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

colors = ['k','r','b','g','y']

formatter_kilos = FuncFormatter(kilos)


# start main loop
tstart = time.time()
for tv, tollLES in enumerate(tollLESValues):

    for k in range(NL):
        
        #-------------- load initial flow
        if (TESTCASE=='HIT_2D'):
            tail = str(int(k*100))
            totTime[k] = k*100
            FILE_REAL = FILE_REAL_PATH + "fields_run0_it" + tail + ".npz"

        if (TESTCASE=='HW' or TESTCASE=='mHW'):
            tail = str(int(k)).zfill(5)
            totTime[k] = k
            FILE_REAL = FILE_REAL_PATH + "fields_time" + tail + ".npz"


        #-------------- load original field
        if (FILE_REAL.endswith('.npz')):

            # load numpy array
            U_DNS, V_DNS, P_DNS, _ = load_fields(FILE_REAL)
            U_DNS = np.cast[DTYPE](U_DNS)
            V_DNS = np.cast[DTYPE](V_DNS)
            P_DNS = np.cast[DTYPE](P_DNS)

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

            U_DNS = orig[:,:,0]
            V_DNS = orig[:,:,1]
            P_DNS = orig[:,:,2]

        # find vorticity
        if (TESTCASE=='HIT_2D'):
            P_DNS = find_vorticity(U_DNS, V_DNS)
            cP_DNS = find_vorticity(U_DNS, V_DNS)
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            # cP_DNS = (tr(V_DNS, 1, 0) - 2*V_DNS + tr(V_DNS, -1, 0))/(DELX**2) \
            #            + (tr(V_DNS, 0, 1) - 2*V_DNS + tr(V_DNS, 0, -1))/(DELY**2)
            cP_DNS = (-tr(V_DNS, 2, 0) + 16*tr(V_DNS, 1, 0) - 30*V_DNS + 16*tr(V_DNS,-1, 0) - tr(V_DNS,-2, 0))/(12*DELX**2) \
                       + (-tr(V_DNS, 0, 2) + 16*tr(V_DNS, 0, 1) - 30*V_DNS + 16*tr(V_DNS, 0,-1) - tr(V_DNS, 0,-2))/(12*DELY**2)


        # rescale
        U_DNS_org = U_DNS/INIT_SCAL
        V_DNS_org = V_DNS/INIT_SCAL
        P_DNS_org = P_DNS/INIT_SCAL
            
        # print plots
        if (tv==-1):
            filename = "results_reconstruction/plots_org/Plots_DNS_org_" + str(k).zfill(4) +".png"
            print_fields_3(U_DNS_org, V_DNS_org, P_DNS_org, N=N_DNS, filename=filename, \
                Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None)
                # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)


        #-------------- save centerline for DNS values
        velx_DNS[tv,0,k] = U_DNS_org[N2_DNS, N2_DNS]
        vely_DNS[tv,0,k] = V_DNS_org[N2_DNS, N2_DNS]
        vort_DNS[tv,0,k] = P_DNS_org[N2_DNS, N2_DNS]

        # save centerline for initial and final values
        if ((tv==len(tollLESValues)-1) and (k==0)):
            c_velx[0,0,:] = U_DNS_org[N2_DNS, :]
            c_vely[0,0,:] = V_DNS_org[N2_DNS, :]
            c_vort[0,0,:] = P_DNS_org[N2_DNS, :]

            # find spectrum
            spectra[0,0,:,:] = plot_spectrum_noPlots(U_DNS_org, V_DNS_org, L)

        if ((tv==len(tollLESValues)-1) and (k==(NL-1))):
            c_velx[0,1,:] = U_DNS_org[N2_DNS, :]
            c_vely[0,1,:] = V_DNS_org[N2_DNS, :]
            c_vort[0,1,:] = P_DNS_org[N2_DNS, :]

            # find spectrum
            spectra[0,1,:,:] = plot_spectrum_noPlots(U_DNS_org, V_DNS_org, L)


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

        if (tv==0 and k==0):
            print("LES resolution is " + str(fimgA.shape[2]) + "x" + str(fimgA.shape[3]))


        #-------------- load coefficients
        if (k%RELOAD_FREQ==0):
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

        #-------------- start research on the latent space
        # save old w
        z0o = tf.identity(z[:,0,:])
        z2o = tf.identity(z[:,2,:])
        k0DNSo = layer_kDNS.trainable_variables[0][0,:]
        k1DNSo = layer_kDNS.trainable_variables[0][1,:]

        it = 0
        UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, filter, z, INIT_SCAL)
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=1)
        tollDNS = tollLES + (resREC - tollLES)/2.0
        while (resREC>tollLES and it<lr_LES_maxIt):

            if (resREC>tollDNS and it<lr_DNS_maxIt):

                resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
                    step_find_zlatents_kDNS(wl_synthesis, filter, opt_kDNS, z, imgA, fimgA, ltv_DNS, INIT_SCAL, typeRes=1)

                kDNSc = layer_kDNS.trainable_variables[0]
                kDNSc = tf.clip_by_value(kDNSc, 0.0, 1.0)
                layer_kDNS.trainable_variables[0].assign(kDNSc)

            else:

                resREC, resLES, resDNS, UVP_DNS, UVP_LES, fUVP_DNS, loss_fil = \
                    step_find_wlatents_mLES(wl_synthesis, filter, opt_mLES, z, imgA, fimgA, ltv_LES, INIT_SCAL, typeRes=1)

                mLESc = layer_mLES.trainable_variables[0]
                mLESc = tf.clip_by_value(mLESc, 0.0, 1.0)
                layer_mLES.trainable_variables[0].assign(mLESc)

            if (it!=0 and it%1000==0):
                # update z
                k0DNS = layer_kDNS.trainable_variables[0][0,:]
                k0DNSm = tf.reduce_mean(k0DNS) 
                zt = k0DNS*z[:,0,:] + (1.0-k0DNS)*z[:,1,:]
                if (k0DNSm>0.5):
                    z0 = z[:,0,:]
                    z1 = 2.0*zt - z0
                else:
                    z1 = z[:,1,:]
                    z0 = 2.0*zt - z1

                k1DNS = layer_kDNS.trainable_variables[0][1,:]
                k1DNSm = tf.reduce_mean(k1DNS) 
                zt = k1DNS*z[:,2,:] + (1.0-k1DNS)*z[:,3,:]
                if (k1DNSm>0.5):
                    z2 = z[:,2,:]
                    z3 = 2.0*zt - z2
                else:
                    z3 = z[:,3,:]
                    z2 = 2.0*zt - z3
                    
                z = tf.concat([z0,z1,z2,z3], axis=0)
                z = z[tf.newaxis,:,:]

                k0DNSo = tf.fill((LATENT_SIZE), 0.5)
                k0DNSo = tf.cast(k0DNSo, dtype=DTYPE)
                k1DNSo = tf.fill((LATENT_SIZE), 0.5)
                k1DNSo = tf.cast(k1DNSo, dtype=DTYPE)

                k0DNS = k0DNSo[tf.newaxis,:]
                k1DNS = k1DNSo[tf.newaxis,:]
                kDNSn = tf.concat([k0DNS, k1DNS], axis=0)
                layer_kDNS.trainable_variables[0].assign(kDNSn)


            # find correct inference
            UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, filter, z, INIT_SCAL)
            resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, imgA, fimgA, typeRes=1)


            # print residuals and fields
            #if ((it%100==0 and it!=0) or (it%100==0 and k==0)):
            if (it%10==0):

                # separate fields from GAN
                U_DNS = UVP_DNS[0, 0, :, :].numpy()
                V_DNS = UVP_DNS[0, 1, :, :].numpy()
                P_DNS = UVP_DNS[0, 2, :, :].numpy()

                U_LES = UVP_LES[0, 0, :, :].numpy()
                V_LES = UVP_LES[0, 1, :, :].numpy()
                P_LES = UVP_LES[0, 2, :, :].numpy()

                fU_DNS = fUVP_DNS[0, 0, :, :].numpy()
                fV_DNS = fUVP_DNS[0, 1, :, :].numpy()
                fP_DNS = fUVP_DNS[0, 2, :, :].numpy()

                # print residuals
                tend = time.time()
                print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fil {6:3e} " \
                    .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))

                # write losses to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('resREC',   resREC,   step=it)
                    tf.summary.scalar('resDNS',   resDNS,   step=it)
                    tf.summary.scalar('resLES',   resLES,   step=it)
                    tf.summary.scalar('loss_fil', loss_fil, step=it)

                if (it%1000==-1):

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
            .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil))


        # update z
        k0DNS = layer_kDNS.trainable_variables[0][0,:]
        k0DNSm = tf.reduce_mean(k0DNS) 
        zt = k0DNS*z[:,0,:] + (1.0-k0DNS)*z[:,1,:]
        if (k0DNSm>0.5):
            z0 = z[:,0,:]
            z1 = 2.0*zt - z0
        else:
            z1 = z[:,1,:]
            z0 = 2.0*zt - z1
            
        k1DNS = layer_kDNS.trainable_variables[0][1,:]
        k1DNSm = tf.reduce_mean(k1DNS) 
        zt = k1DNS*z[:,2,:] + (1.0-k1DNS)*z[:,3,:]
        if (k1DNSm>0.5):
            z2 = z[:,2,:]
            z3 = 2.0*zt - z2
        else:
            z3 = z[:,3,:]
            z2 = 2.0*zt - z3
            
        z = tf.concat([z0,z1,z2,z3], axis=0)
        z = z[tf.newaxis,:,:]

        k0DNSo = tf.fill((LATENT_SIZE), 0.5)
        k0DNSo = tf.cast(k0DNSo, dtype=DTYPE)
        k1DNSo = tf.fill((LATENT_SIZE), 0.5)
        k1DNSo = tf.cast(k1DNSo, dtype=DTYPE)

        k0DNS = k0DNSo[tf.newaxis,:]
        k1DNS = k1DNSo[tf.newaxis,:]
        kDNSn = tf.concat([k0DNS, k1DNS], axis=0)
        layer_kDNS.trainable_variables[0].assign(kDNSn)


        # separate DNS fields from GAN
        U_DNS = UVP_DNS[0, 0, :, :].numpy()
        V_DNS = UVP_DNS[0, 1, :, :].numpy()
        P_DNS = UVP_DNS[0, 2, :, :].numpy()

        # save fields
        if ((k==0) or k==(NL-1)):
            filename = "results_reconstruction/fields/fields_tv" + str(tv) + "_k" + str(k) + ".npz"
            save_fields(0, U_DNS, V_DNS, P_DNS, filename)


        #---------------------- save centerline DNS from GAN values
        velx_DNS[tv,1,k] = U_DNS[N2_DNS, N2_DNS]
        vely_DNS[tv,1,k] = V_DNS[N2_DNS, N2_DNS]
        vort_DNS[tv,1,k] = P_DNS[N2_DNS, N2_DNS]

        if ((tv==len(tollLESValues)-1) and (k==0)):
            c_velx[1,0,:] = U_DNS[N2_DNS, :]
            c_vely[1,0,:] = V_DNS[N2_DNS, :]
            c_vort[1,0,:] = P_DNS[N2_DNS, :]

            # find spectrum
            spectra[1,0,:,:] = plot_spectrum_noPlots(U_DNS, V_DNS, L)

        if ((tv==len(tollLESValues)-1) and (k==(NL-1))):
            c_velx[1,1,:] = U_DNS[N2_DNS, :]
            c_vely[1,1,:] = V_DNS[N2_DNS, :]
            c_vort[1,1,:] = P_DNS[N2_DNS, :]

            # find spectrum
            spectra[1,1,:,:] = plot_spectrum_noPlots(U_DNS, V_DNS, L)

        if (tv==len(tollLESValues)-1):

            # print fields
            # filename = "results_reconstruction/plots/Plots_DNS_fromGAN_" + tail + "_" + str(tv) + ".png"
            # print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename, \
            # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

            # filename = "results_reconstruction/plots/Plots_DNS_diffs_" + tail + "_" + str(tv) + ".png"
            # print_fields_3_diff(P_DNS_org, P_DNS, tf.math.abs(P_DNS_org-P_DNS), N=N_DNS, filename=filename, \
            # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=0.0, Pmax=1.0)

            # separate DNS fields from GAN
            UVP_DNS, UVP_LES, fUVP_DNS, _, _ = find_predictions(wl_synthesis, filter, z, INIT_SCAL)

            U_LES = UVP_LES[0, 0, :, :].numpy()
            V_LES = UVP_LES[0, 1, :, :].numpy()
            P_LES = UVP_LES[0, 2, :, :].numpy()

            # mvars = layer_LES.trainable_variables[0].numpy()
            # print("Final min and max k: ", np.min(mvars), np.max(mvars))

            # filename = "results_reconstruction/plots/Plots_LES_diffs_" + tail + "_" + str(tv) + ".png"
            # print_fields_2(P_LES, P_DNS, N_DNS, filename, Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0)

            filename = "results_reconstruction/plots/Plots_DNS_diffs_" + str(k).zfill(5) + "_tv" + str(tv) + ".png"
            print_fields_4_diff(P_DNS_org, P_LES, P_DNS, tf.math.abs(P_DNS_org-P_DNS), N_DNS, filename) #, \
                                # Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=0.0, Pmax=1.0)

            # save fields
            filename = "results_reconstruction/fields/fields_lat"  + str(k).zfill(5) + "_tv" + str(tv) + ".npz"
            save_fields(0, U_DNS, V_DNS, P_DNS, filename=filename)

    # plot values
    if (tv==0):
        lineColor = colors[tv]
        stollLES = "{:.1e}".format(tollLESValues[tv])

        if (TESTCASE=='HIT_2D'):
            ax1.plot(totTime[:], velx_DNS[tv,0,:], color=lineColor, label=r'DNS')
            ax2.plot(totTime[:], vely_DNS[tv,0,:], color=lineColor, label=r'DNS')
            ax3.plot(totTime[:], vort_DNS[tv,0,:], color=lineColor, label=r'DNS')
        elif (TESTCASE=='HW' or TESTCASE=='mHW'):
            ax1.plot(totTime[:], velx_DNS[tv,0,:], color=lineColor, label=r'DNS')
            ax2.plot(totTime[:], vely_DNS[tv,0,:], color=lineColor, label=r'DNS')
            ax3.plot(totTime[:], vort_DNS[tv,0,:], color=lineColor, label=r'DNS')

    lineColor = colors[tv+1]
    exponent = int(floor(log10(abs(tollLESValues[tv]))))
    stollLES = r"$10^{{{0:d}}}$".format(exponent, 1)

    velx_norm = np.sum(np.sqrt((velx_DNS[tv,0,:] - velx_DNS[tv,1,:])**2))
    vely_norm = np.sum(np.sqrt((vely_DNS[tv,0,:] - vely_DNS[tv,1,:])**2))
    vort_norm = np.sum(np.sqrt((vort_DNS[tv,0,:] - vort_DNS[tv,1,:])**2))

    ax1.plot(totTime[:], velx_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'$\epsilon_{REC}$=' + stollLES + ',  $L_2$ = {:.2f}'.format(velx_norm))
    ax2.plot(totTime[:], vely_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'$\epsilon_{REC}$=' + stollLES + ',  $L_2$ = {:.2f}'.format(vely_norm))
    ax3.plot(totTime[:], vort_DNS[tv,1,:], color=lineColor, linestyle='dashed', label=r'$\epsilon_{REC}$=' + stollLES + ',  $L_2$ = {:.2f}'.format(vort_norm))

    # save centerline values on file
    np.savez("results_reconstruction/uvw_vs_time.npz", totTime=totTime, \
        U_DNS=velx_DNS, V_DNS=vely_DNS, W_DNS=vort_DNS, \
        U_LES=velx_LES, V_LES=vely_LES, W_LES=vort_LES)

    ax1.legend(frameon=False, prop={'size': 14})
    ax2.legend(frameon=False, prop={'size': 14})
    ax3.legend(frameon=False, prop={'size': 14})

    ax1.set_xlabel(r'steps', fontsize=20)
    ax2.set_xlabel(r'steps', fontsize=20)
    ax3.set_xlabel(r'steps', fontsize=20)
    
    if (TESTCASE=='HIT_2D'):
        ax1.set_ylabel(r'$u$',      fontsize=20)
        ax2.set_ylabel(r'$v$',      fontsize=20)
        ax3.set_ylabel(r'$\omega$', fontsize=20)
    else:
        ax1.set_ylabel(r'$n$',      fontsize=20)
        ax2.set_ylabel(r'$\phi$',   fontsize=20)
        ax3.set_ylabel(r'$\omega$', fontsize=20)

    ax1.xaxis.set_major_formatter(formatter_kilos)
    ax2.xaxis.set_major_formatter(formatter_kilos)
    ax3.xaxis.set_major_formatter(formatter_kilos)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(16)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(16)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(16)
    
    fig.tight_layout(pad=5.0)
                                   
    plt.savefig("results_reconstruction/uvw_vs_time.png", bbox_inches='tight', pad_inches=0.05)

plt.close()


#------------------- plot centerline profiles
x = np.linspace(0, L, N_DNS)
c_fig, c_axs = plt.subplots(2, 3, figsize=(20,10))
c_fig.subplots_adjust(hspace=0.25)
cax1 = c_axs[0,0]
cax2 = c_axs[0,1]
cax3 = c_axs[0,2]
cax4 = c_axs[1,0]
cax5 = c_axs[1,1]
cax6 = c_axs[1,2]

# first step
k=0
lineColor = 'k'
cax1.plot(x, c_velx[0,0,:], color=lineColor, label=r'DNS $u$ at ' + t0_label)
cax2.plot(x, c_vely[0,0,:], color=lineColor, label=r'DNS $v$ at ' + t0_label)
cax3.plot(x, c_vort[0,0,:], color=lineColor, label=r'DNS $\omega$ ' + t0_label)

lineColor = 'r'
cax1.plot(x, c_velx[1,0,:], color=lineColor, label=r'StylES $u$ at ' + t0_label)
cax2.plot(x, c_vely[1,0,:], color=lineColor, label=r'StylES $v$ at ' + t0_label)
cax3.plot(x, c_vort[1,0,:], color=lineColor, label=r'StylES $\omega$ at ' + t0_label)

# final step
k = 1
lineColor = 'k'
cax4.plot(x, c_velx[0,1,:], color=lineColor, linestyle='dotted', label=r'DNS $u$ at ' + tf_label)
cax5.plot(x, c_vely[0,1,:], color=lineColor, linestyle='dotted', label=r'DNS $v$ at ' + tf_label)
cax6.plot(x, c_vort[0,1,:], color=lineColor, linestyle='dotted', label=r'DNS $\omega$ at ' + tf_label)

lineColor = 'r'
cax4.plot(x, c_velx[1,1,:], color=lineColor, linestyle='dotted', label=r'StylES $u$ at ' + tf_label)
cax5.plot(x, c_vely[1,1,:], color=lineColor, linestyle='dotted', label=r'StylES $v$ at ' + tf_label)
cax6.plot(x, c_vort[1,1,:], color=lineColor, linestyle='dotted', label=r'StylES $\omega$ at ' + tf_label)

cax1.set_xlabel("x", fontsize=20)
cax2.set_xlabel("x", fontsize=20)
cax3.set_xlabel("x", fontsize=20)

cax4.set_xlabel("x", fontsize=20)
cax5.set_xlabel("x", fontsize=20)
cax6.set_xlabel("x", fontsize=20)

for label in (cax1.get_xticklabels() + cax1.get_yticklabels()):
    label.set_fontsize(16)
for label in (cax2.get_xticklabels() + cax2.get_yticklabels()):
    label.set_fontsize(16)
for label in (cax3.get_xticklabels() + cax3.get_yticklabels()):
    label.set_fontsize(16)

for label in (cax4.get_xticklabels() + cax4.get_yticklabels()):
    label.set_fontsize(16)
for label in (cax5.get_xticklabels() + cax5.get_yticklabels()):
    label.set_fontsize(16)
for label in (cax6.get_xticklabels() + cax6.get_yticklabels()):
    label.set_fontsize(16)
        
cax1.legend(frameon=False, prop={'size': 14})
cax2.legend(frameon=False, prop={'size': 14})
cax3.legend(frameon=False, prop={'size': 14})

cax4.legend(frameon=False, prop={'size': 14})
cax5.legend(frameon=False, prop={'size': 14})
cax6.legend(frameon=False, prop={'size': 14})

plt.savefig("results_reconstruction/uvw_vs_x.png", bbox_inches='tight', pad_inches=0.05)
plt.close()

np.savez("results_reconstruction/uvw_vs_x.npz", totTime=totTime, U=c_velx, V=c_vely, W=c_vort)




#------------------- plot energy spectra
plt.plot(spectra[0,0,0,:], spectra[0,0,1,:], color='k',                     linewidth=0.5, label=r'DNS at ' + t0_label)
plt.plot(spectra[0,1,0,:], spectra[0,1,1,:], color='r',                     linewidth=0.5, label=r'DNS at ' + tf_label)
plt.plot(spectra[1,0,0,:], spectra[1,0,1,:], color='k', linestyle='dotted', linewidth=0.5, label=r'StylES at ' + t0_label)
plt.plot(spectra[1,1,0,:], spectra[1,1,1,:], color='r', linestyle='dotted', linewidth=0.5, label=r'StylES at ' + tf_label)

plt.legend(frameon=False, prop={'size': 14})

plt.xscale("log")
plt.yscale("log")
# plt.xlim(xLogLim)
# plt.ylim(yLogLim)
plt.xlabel("k")
plt.ylabel("E")

plt.savefig("results_reconstruction/Energy_spectrum.png", pad_inches=0.5)

np.savez("results_reconstruction/uvw_vs_time.npz", spectra=spectra)