#----------------------------------------------------------------------------------------------
#
#    Copyright (C): 2021 UKRI-STFC (Hartree Centre)
#
#    Author: Jony Castagna
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

from time import time
from PIL import Image
from math import sqrt

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

from LES_functions  import *
from LES_plot       import *
from LES_lAlg       import *

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, './testcases/HIT_2D/')

DTYPE_LES = DTYPE

os.chdir('../')
from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *
os.chdir('./LES_Solvers/')

from tensorflow.keras.applications.vgg16 import VGG16

DTYPE = DTYPE_LES  # this is only because the StyleGAN is trained with float32 usually






#---------------------------- local variables
NLES = 2**RES_LOG2_FIL
PROCEDURE = "DNS"

if PROCEDURE=="A1":
    FILTER       = "StyleGAN_layer"
    USE_DLATENTS = True
    INIT_BC      = 0
elif PROCEDURE=="A2":
    FILTER       = "Trained_filter"
    USE_DLATENTS = True
    INIT_BC      = 0
elif PROCEDURE=="B2":
    FILTER       = "StyleGAN_layer"
    USE_DLATENTS = True
    INIT_BC      = 0
    firstRetrain = True
elif PROCEDURE=="DNS":
    FILTER       = "Gaussian"
    USE_DLATENTS = False
    INIT_BC      = 3

Uo = nc.zeros([NLES,NLES], dtype=DTYPE)   # old x-velocity
Vo = nc.zeros([NLES,NLES], dtype=DTYPE)   # old y-velocity
Po = nc.zeros([NLES,NLES], dtype=DTYPE)   # old pressure field
Co = nc.zeros([NLES,NLES], dtype=DTYPE)   # old passive scalar

Uo_DNS = nc.zeros([N,N], dtype=DTYPE)     # old x-velocity DNS
Vo_DNS = nc.zeros([N,N], dtype=DTYPE)     # old y-velocity DNS
Po_DNS = nc.zeros([N,N], dtype=DTYPE)     # old pressure DNS

pc = nc.zeros([NLES,NLES], dtype=DTYPE)   # pressure correction
Z  = nc.zeros([NLES,NLES], dtype=DTYPE)   # zero array
C  = np.zeros([NLES,NLES], dtype=DTYPE)   # scalar
B  = np.zeros([NLES,NLES], dtype=DTYPE)   # body force
P  = np.zeros([NLES,NLES], dtype=DTYPE)   # body force

DNS_cv = np.zeros([totSteps+1, 4])
LES_cv = np.zeros([totSteps+1, 4])
LES_cv_fromDNS = np.zeros([totSteps+1, 4])

U_diff = np.zeros([N, N], dtype=DTYPE)
V_diff = np.zeros([N, N], dtype=DTYPE)
P_diff = np.zeros([N, N], dtype=DTYPE)
W_diff = np.zeros([N, N], dtype=DTYPE)






#---------------------------- clean up and prepare run
# clean up and declarations
#os.system("rm restart.npz")
os.system("rm DNS_fromGAN_center_values.txt")
os.system("rm LES_center_values.txt")
os.system("rm LES_fromGAN_center_values.txt")
os.system("rm Plots*")
os.system("rm Fields*")
os.system("rm Energy_spectrum*")
os.system("rm Uvp*")

os.system("rm -rf plots")
os.system("rm -rf fields")
os.system("rm -rf uvp")
os.system("rm -rf energy")
os.system("rm -rf logs")
os.system("rm -rf v_viol")

os.system("mkdir plots")
os.system("mkdir fields")
os.system("mkdir uvp")
os.system("mkdir energy")
os.system("mkdir v_viol")

DiffCoef = np.full([NLES, NLES], Dc)
NL_DNS   = np.zeros([1, 3, N, N])
NL       = np.zeros([1, 3, NLES, NLES])

if (len(te)>0):
    tail = "0te"
else:
    tail = "it0"

dir_train_log        = 'logs/DNS_solver/'
train_summary_writer = tf.summary.create_file_writer(dir_train_log)

tf.random.set_seed(1)



# define wl_synthesis models
with mirrored_strategy.scope():

    # define noise variances
    inputVar1 = tf.constant(1.0, shape=[BATCH_SIZE, G_LAYERS-2], dtype=DTYPE)
    inputVar2 = tf.constant(1.0, shape=[BATCH_SIZE, 2], dtype=DTYPE)
    inputVariances = tf.concat([inputVar1,inputVar2],1)


    # Download VGG16 model
    VGG_model         = VGG16(input_shape=(OUTPUT_DIM, OUTPUT_DIM, NUM_CHANNELS), include_top=False, weights='imagenet')
    VGG_features_list = [layer.output for layer in VGG_model.layers]
    VGG_extractor     = tf.keras.Model(inputs=VGG_model.input, outputs=VGG_features_list)


    # loading StyleGAN checkpoint and filter
    checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))


    # create variable synthesis model
    if (USE_DLATENTS):
        dlatents       = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
        wlatents       = layer_wlatent(dlatents)
        ndlatents      = wlatents(dlatents)
        outputs        = synthesis([ndlatents, inputVariances], training=False)
        wl_synthesis   = tf.keras.Model(dlatents, outputs)
    else:
        latents        = tf.keras.Input(shape=[LATENT_SIZE])
        wlatents       = layer_wlatent(latents)
        nlatents       = wlatents(latents)
        dlatents       = mapping(nlatents)
        outputs        = synthesis([dlatents, inputVariances], training=False)
        wl_synthesis   = tf.keras.Model(latents, outputs)


    # add latent space to trainable variables
    for variable in wlatents.trainable_variables:
        list_DNS_trainable_variables.append(variable)
        list_LES_trainable_variables.append(variable)


    # define learnin rate schedule and optimizer for DNS step
    if (lrDNS_POLICY=="EXPONENTIAL"):
        lr_schedule  = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lrDNS,
            decay_steps=lrDNS_STEP,
            decay_rate=lrDNS_RATE,
            staircase=lrDNS_EXP_ST)
    elif (lrDNS_POLICY=="PIECEWISE"):
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lrDNS_BOUNDS, lrDNS_VALUES)
    opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)


    # define learnin rate schedule and optimizer for LES step
    if (lrDNS_POLICY=="EXPONENTIAL"):
        lr_schedule_LES  = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lrDNS,
            decay_steps=lrDNS_STEP,
            decay_rate=lrDNS_RATE,
            staircase=lrDNS_EXP_ST)
    elif (lrDNS_POLICY=="PIECEWISE"):
        lr_schedule_LES = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lrDNS_BOUNDS, lrDNS_VALUES)
    opt_LES = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_LES)


    # define checkpoint
    wl_checkpoint = tf.train.Checkpoint(wl_synthesis=wl_synthesis,
                                        opt=opt)








#---------------------------- local functions
@tf.function
def find_latents_DNS(latents, imgA, list_trainable_variables=wl_synthesis.trainable_variables):
    with tf.GradientTape() as tape_DNS:
        predictions = wl_synthesis(latents, training=False)
        UVP_DNS     = predictions[RES_LOG2-2]

        resDNS = tf.math.reduce_mean(tf.math.squared_difference(imgA, UVP_DNS[0,:,:,:]))

        gradients_DNS = tape_DNS.gradient(resDNS, list_trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, list_trainable_variables))

        return resDNS, predictions, UVP_DNS



@tf.function
def find_latents_DNS_step(latents, imgA, list_trainable_variables):
    resDNS, predictions, UVP_DNS = mirrored_strategy.run(find_latents_DNS, args=(latents, imgA, list_trainable_variables))
    return resDNS, predictions, UVP_DNS



@tf.function
def find_latents_LES(latents, imgA, list_trainable_variables=wl_synthesis.trainable_variables):
    with tf.GradientTape() as tape_DNS:
        predictions = wl_synthesis(latents)

        if (FILTER=="Trained_filter"):
            UVP = filter(predictions[RES_LOG2-2], training=False)
        elif (FILTER=="StyleGAN_layer"):
            UVP = predictions[RES_LOG2_FIL-2]
        elif (FILTER=="Gaussian"):
            pass

        resDNS = tf.math.reduce_mean(tf.math.squared_difference(imgA[0:2,:,:], UVP[0,0:2,:,:]))  # match only the U and V values!

        gradients_DNS = tape_DNS.gradient(resDNS, list_trainable_variables)
        opt_LES.apply_gradients(zip(gradients_DNS, list_trainable_variables))

        return resDNS, predictions, UVP


@tf.function
def find_latents_LES_step(latents, imgA, list_trainable_variables):
    resDNS, predictions, UVP_DNS = mirrored_strategy.run(find_latents_LES, args=(latents, imgA, list_trainable_variables))
    return resDNS, predictions, UVP_DNS




#---------------------------- initialize flow
tstart = time.time()

if (INIT_BC==0):

    # find DNS and LES fields from a reference DNS field

    # load DNS reference fields
    U_DNS_org, V_DNS_org, P_DNS_org, C_DNS_org, B_DNS_org, totTime = load_fields()  #from restart.npz file

    W_DNS_org = find_vorticity(U_DNS_org, V_DNS_org)
    print_fields(U_DNS_org, V_DNS_org, P_DNS_org, W_DNS_org, N, filename="plots/plots_DNS_org.png")

    # find max/min values and normalize
    maxU = np.max(U_DNS_org)
    minU = np.min(U_DNS_org)
    U_DNS_org = two*(U_DNS_org - minU)/(maxU- minU) - one

    maxV = np.max(V_DNS_org)
    minV = np.min(V_DNS_org)
    V_DNS_org = two*(V_DNS_org - minV)/(maxV - minV) - one

    maxP = np.max(P_DNS_org)
    minP = np.min(P_DNS_org)
    P_DNS_org = two*(P_DNS_org - minP)/(maxP - minP) - one

    # create image for TensorFlow
    itDNS   = 0
    resDNS  = large
    tU_DNS = tf.convert_to_tensor(U_DNS_org, dtype=np.float32)
    tV_DNS = tf.convert_to_tensor(V_DNS_org, dtype=np.float32)
    tP_DNS = tf.convert_to_tensor(P_DNS_org, dtype=np.float32)

    tU_DNS = tU_DNS[np.newaxis,:,:]
    tV_DNS = tV_DNS[np.newaxis,:,:]
    tP_DNS = tP_DNS[np.newaxis,:,:]

    imgA_DNS = tf.concat([tU_DNS, tV_DNS, tP_DNS], 0)

    # prepare latent space
    if (USE_DLATENTS):
        zlatents = tf.random.uniform([1, LATENT_SIZE])
        latents  = mapping(zlatents, training=False)
    else:
        latents = tf.random.uniform([1, LATENT_SIZE])

    # find latent space
    while (resDNS>tollDNS and itDNS<maxItDNS):
        resDNS, predictions, UVP_DNS = find_latents_DNS_step(latents, imgA_DNS, list_DNS_trainable_variables)

        lr = lr_schedule(itDNS)
        if (itDNS%100 == 0):
            print("Search DNS iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))
            #U_DNS = UVP_DNS[0, 0, :, :].numpy()
            #V_DNS = UVP_DNS[0, 1, :, :].numpy()
            #P_DNS = UVP_DNS[0, 2, :, :].numpy()
            #W_DNS = find_vorticity(U_DNS, V_DNS)
            #print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="Plots_DNS_fromGAN.png")

            with train_summary_writer.as_default():
                tf.summary.scalar("residuals", resDNS, step=itDNS)
                tf.summary.scalar("lr", lr, step=itDNS)

        itDNS = itDNS+1

    # print final residuals
    print("Search DNS iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))
    
    with train_summary_writer.as_default():
        tf.summary.scalar("residuals", resDNS, step=itDNS)
        tf.summary.scalar("lr", lr, step=itDNS)

    # find numpy arrays from GAN inference
    U_DNS = UVP_DNS[0, 0, :, :].numpy()
    V_DNS = UVP_DNS[0, 1, :, :].numpy()
    P_DNS = UVP_DNS[0, 2, :, :].numpy()
    W_DNS = find_vorticity(U_DNS, V_DNS)

    # re-normalize
    U_DNS = (U_DNS+one)*(maxU-minU)/two + minU
    V_DNS = (V_DNS+one)*(maxV-minV)/two + minV
    P_DNS = (P_DNS+one)*(maxP-minP)/two + minP

    # find LES field
    if (FILTER=="Trained_filter"):
        UVP = filter(UVP_DNS, training=False)
        U = UVP[0, 0, :, :].numpy()
        V = UVP[0, 1, :, :].numpy()
        P = UVP[0, 2, :, :].numpy()
    elif (FILTER=="StyleGAN_layer"):
        U = predictions[RES_LOG2_FIL-2][0, 0, :, :].numpy()
        V = predictions[RES_LOG2_FIL-2][0, 1, :, :].numpy()
        P = predictions[RES_LOG2_FIL-2][0, 2, :, :].numpy()
    elif (FILTER=="Gaussian"):
        pass

    # re-normalize new DNS and LES fields
    U = (U+one)*(maxU-minU)/two + minU
    V = (V+one)*(maxV-minV)/two + minV
    P = (P+one)*(maxP-minP)/two + minP    


elif (INIT_BC==1):

    pass

    # # find DNS and LES fields from random input 
    # totTime = zero
    # B_DNS = nc.zeros([N,N], dtype=DTYPE)   # body force
    # C_DNS = nc.zeros([N,N], dtype=DTYPE)   # passive scalar

    # if (USE_DLATENTS):
    #     zlatents = tf.random.uniform([1, LATENT_SIZE])
    #     latents  = mapping(zlatents, training=False)
    # else:
    #     latents = tf.random.uniform([1, LATENT_SIZE])

    # predictions = wl_synthesis(latents, training=False)
    # UVP_DNS     = predictions[RES_LOG2-2]

    # # find DNS field
    # U_DNS = UVP_DNS[0, 0, :, :].numpy()
    # V_DNS = UVP_DNS[0, 1, :, :].numpy()
    # P_DNS = UVP_DNS[0, 2, :, :].numpy()

    # # set LES field: you must first find the pressure field!
    # # assemble fields U, V from generated DNS plus pressure from restart file
    # UVP_DNS = tf.convert_to_tensor(UVP_DNS)

    # # filter them
    # if (FILTER):
    #     UVP = filter(UVP_DNS, training=False)
    #     U = UVP[0, 0, :, :].numpy()
    #     V = UVP[0, 1, :, :].numpy()
    #     P = UVP[0, 2, :, :].numpy()
    # else:
    #     U = predictions[RES_LOG2_FIL-2][0, 0, :, :].numpy()
    #     V = predictions[RES_LOG2_FIL-2][0, 1, :, :].numpy()
    #     P = predictions[RES_LOG2_FIL-2][0, 2, :, :].numpy()

elif (INIT_BC==2):

    pass
    # # load latest StyLES checkpoint
    # wl_checkpoint.restore(tf.train.latest_checkpoint(CHKP_DIR))

    # totTime = zero
    # B_DNS = nc.zeros([N,N], dtype=DTYPE)   # body force
    # C_DNS = nc.zeros([N,N], dtype=DTYPE)   # passive scalar

    # if (USE_DLATENTS):
    #     zlatents     = tf.random.uniform([1, LATENT_SIZE])
    #     dlatents    = mapping(zlatents, training=False)
    #     predictions = wl_synthesis([dlatents, inputVariances], training=False)
    # else:
    #     latents      = tf.random.uniform([1, LATENT_SIZE])
    #     predictions  = wl_synthesis([latents, inputVariances], training=False)

    # UVP_DNS = predictions[RES_LOG2-2]

    # # set DNS field
    # U_DNS = UVP_DNS[0, 0, :, :].numpy()
    # V_DNS = UVP_DNS[0, 1, :, :].numpy()
    # P_DNS = UVP_DNS[0, 2, :, :].numpy()

    # # set LES fields
    # if (FILTER):
    #     UVP = filter(UVP_DNS, training=False)
    #     U = UVP[0, 0, :, :].numpy()
    #     V = UVP[0, 1, :, :].numpy()
    #     P = UVP[0, 2, :, :].numpy()
    # else:
    #     U = predictions[RES_LOG2_FIL-2][0, 0, :, :].numpy()
    #     V = predictions[RES_LOG2_FIL-2][0, 1, :, :].numpy()
    #     P = predictions[RES_LOG2_FIL-2][0, 2, :, :].numpy()
    

elif (INIT_BC==3):

    # find DNS and LES fields from a reference DNS time series

    # load DNS reference fields from restart.npz file
    U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, totTime = load_fields()

    # print original field
    W_DNS = find_vorticity(U_DNS, V_DNS)
    print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_org.png")

    # find LES field
    if (FILTER=="Trained_filter"):

        # find max/min values and normalize
        maxU = np.max(U_DNS)
        minU = np.min(U_DNS)
        U_DNS = two*(U_DNS - minU)/(maxU- minU) - one

        maxV = np.max(V_DNS)
        minV = np.min(V_DNS)
        V_DNS = two*(V_DNS - minV)/(maxV - minV) - one

        maxP = np.max(P_DNS)
        minP = np.min(P_DNS)
        P_DNS = two*(P_DNS - minP)/(maxP - minP) - one

        # create image for TensorFlow
        itDNS   = 0
        resDNS  = large
        tU_DNS = tf.convert_to_tensor(U_DNS, dtype=np.float32)
        tV_DNS = tf.convert_to_tensor(V_DNS, dtype=np.float32)
        tP_DNS = tf.convert_to_tensor(P_DNS, dtype=np.float32)

        tU_DNS = tU_DNS[np.newaxis,np.newaxis,:,:]
        tV_DNS = tV_DNS[np.newaxis,np.newaxis,:,:]
        tP_DNS = tP_DNS[np.newaxis,np.newaxis,:,:]

        UVP_DNS = tf.concat([tU_DNS, tV_DNS, tP_DNS], 1)

        # filter
        UVP = filter(UVP_DNS, training=False)
        U = UVP[0, 0, :, :].numpy()
        V = UVP[0, 1, :, :].numpy()
        P = UVP[0, 2, :, :].numpy()      

        # re-normalize new DNS and LES fields
        U_DNS = (U_DNS+one)*(maxU-minU)/two + minU
        V_DNS = (V_DNS+one)*(maxV-minV)/two + minV
        P_DNS = (P_DNS+one)*(maxP-minP)/two + minP

        U = (U+one)*(maxU-minU)/two + minU
        V = (V+one)*(maxV-minV)/two + minV
        P = (P+one)*(maxP-minP)/two + minP   

    elif (FILTER=="StyleGAN_layer"):

        pass

    elif (FILTER=="Gaussian"):
        rs = int(N/NLES)
        if (rs==1):
            U = U_DNS[::rs,::rs]
            V = V_DNS[::rs,::rs]  
            P = P_DNS[::rs,::rs]  
        else:
            fU = sc.ndimage.gaussian_filter(U_DNS, rs, mode='grid-wrap')
            fV = sc.ndimage.gaussian_filter(V_DNS, rs, mode='grid-wrap')
            fP = sc.ndimage.gaussian_filter(P_DNS, rs, mode='grid-wrap')
            
            U = fU[::rs,::rs]
            V = fV[::rs,::rs]  
            P = fP[::rs,::rs]  





#---------------------------- main time step loop

# init variables
tstep    = 0
resM_cpu = zero
resP_cpu = zero
resC_cpu = zero
res_cpu  = zero
its      = 0

# check divergence
div = rho*A*nc.sum(nc.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
div = div*iNN
div_cpu = convert(div)

# find new delt based on Courant number
cdelt = CNum*dl/(sqrt(nc.max(U_DNS)*nc.max(U_DNS) + nc.max(V_DNS)*nc.max(V_DNS))+small)
delt = convert(cdelt)
delt = min(delt, maxDelt)

# print values
tend = time.time()
if (tstep%print_res == 0):
    wtime = (tend-tstart)
    print("Wall time [s] {0:6.1f}  steps {1:3d}  time {2:5.2e}  delt {3:5.2e}  resM {4:5.2e}  "\
        "resP {5:5.2e}  resC {6:5.2e}  res {7:5.2e}  its {8:3d}  div {9:5.2e}"       \
    .format(wtime, tstep, totTime, delt, resM_cpu, resP_cpu, \
    resC_cpu, res_cpu, its, div_cpu))

# plot, save, find spectrum fields
if (len(te)>0):

    #loop for turnover times(te) and respective time in seconds(te_s)
    for s in range(len(te_s)):
        if (totTime<te_s[s]+hf*delt and totTime>te_s[s]-hf*delt):

            # find vorticity
            W_DNS = find_vorticity(U_DNS, V_DNS)
            W     = find_vorticity(U,V)

            # save fields
            save_fields(totTime, U, V, P, C, B, W, "fields/fields_" + str(te[s]) + "te.npz")

            # print fields
            print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N   ,"plots/plots_DNS_fromGAN_" + str(te[s]) + "te.png")
            print_fields(U,     V,     P,     W,     NLES,"plots/plots_LES_"         + str(te[s]) + "te.png")

            #print spectrum
            plot_spectrum(U_DNS, V_DNS, L, "energy/energy_DNS_fromGAN_" + str(te[s]) + "te.txt")
            plot_spectrum(U, V, L,         "energy/energy_LES_"         + str(te[s]) + "te.txt")
else:

    tail = "it{0:d}".format(tstep)

    # find vorticity
    W_DNS = find_vorticity(U_DNS, V_DNS)
    W     = find_vorticity(U,V)

    # save fields
    if (tstep%print_ckp == 0):
        save_fields(totTime, U, V, P, C, B, W, "fields/fields_" + tail + ".npz")

    # print fields
    if (tstep%print_img == 0):
        print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N,    "plots/plots_DNS_fromGAN_" + tail + ".png")
        print_fields(U,     V,     P,     W,     NLES, "plots/plots_LES_"         + tail + ".png")

    # print spectrum
    if (tstep%print_spe == 0):
        plot_spectrum(U_DNS, V_DNS, L, "energy/energy_spectrum_DNS_fromGAN_" + tail + ".txt")
        plot_spectrum(U,     V,     L, "energy/energy_spectrum_LES_"         + tail + ".txt")

# track center point velocities and pressure
DNS_cv[tstep,0] = totTime
DNS_cv[tstep,1] = U_DNS[N//2, N//2]
DNS_cv[tstep,2] = V_DNS[N//2, N//2]
DNS_cv[tstep,3] = P_DNS[N//2, N//2]

LES_cv[tstep,0] = totTime
LES_cv[tstep,1] = U[NLES//2, NLES//2]
LES_cv[tstep,2] = V[NLES//2, NLES//2]
LES_cv[tstep,3] = P[NLES//2, NLES//2]

LES_cv_fromDNS[tstep,0] = totTime
LES_cv_fromDNS[tstep,1] = U[NLES//2, NLES//2]
LES_cv_fromDNS[tstep,2] = V[NLES//2, NLES//2]
LES_cv_fromDNS[tstep,3] = P[NLES//2, NLES//2]




# start loop
while (tstep<totSteps and totTime<finalTime):


    # save old values of U, V and P
    Uo[:,:]     = U[:,:]
    Vo[:,:]     = V[:,:]
    Po[:,:]     = P[:,:]
    Uo_DNS[:,:] = U_DNS[:,:]
    Vo_DNS[:,:] = V_DNS[:,:]
    Po_DNS[:,:] = P_DNS[:,:]

    if (PASSIVE):
        Co[:,:] = C[:,:]


    # find non linear  terms
    NL_DNS[0,0,:,:] = Uo_DNS*Uo_DNS
    NL_DNS[0,1,:,:] = Uo_DNS*Vo_DNS
    NL_DNS[0,2,:,:] = Vo_DNS*Vo_DNS

    # filter them
    if (FILTER=="Gaussian"):

        rs = int(N/NLES)
        if (rs==1):
            fUU = NL_DNS[0,0,::rs,::rs]
            fUV = NL_DNS[0,1,::rs,::rs]
            fVV = NL_DNS[0,2,::rs,::rs]        
        else:
            UU = sc.ndimage.gaussian_filter(NL_DNS[0,0,:,:], rs, mode='grid-wrap')
            UV = sc.ndimage.gaussian_filter(NL_DNS[0,1,:,:], rs, mode='grid-wrap')
            VV = sc.ndimage.gaussian_filter(NL_DNS[0,2,:,:], rs, mode='grid-wrap')
            
            fUU = UU[::rs,::rs]
            fUV = UV[::rs,::rs]
            fVV = VV[::rs,::rs]        

    else:

        # find max/min values and normalize
        maxUU = np.max(NL_DNS[:,0,:,:])
        minUU = np.min(NL_DNS[:,0,:,:])
        NL_DNS[:,0,:,:] = two*(NL_DNS[:,0,:,:] - minUU)/(maxUU- minUU) - one

        maxUV = np.max(NL_DNS[:,1,:,:])
        minUV = np.min(NL_DNS[:,1,:,:])
        NL_DNS[:,1,:,:] = two*(NL_DNS[:,1,:,:] - minUV)/(maxUV - minUV) - one

        maxVV = np.max(NL_DNS[:,2,:,:])
        minVV = np.min(NL_DNS[:,2,:,:])
        NL_DNS[:,2,:,:] = two*(NL_DNS[:,2,:,:] - minVV)/(maxVV - minVV) - one

        tNL_DNS = tf.convert_to_tensor(NL_DNS)

        NL = filter(tNL_DNS, training=False)
        fUU = NL[0, 0, :, :].numpy()
        fUV = NL[0, 1, :, :].numpy()
        fVV = NL[0, 2, :, :].numpy()

        # re-normalize new DNS and LES fields
        fUU = (fUU+one)*(maxUU-minUU)/two + minUU
        fUV = (fUV+one)*(maxUV-minUV)/two + minUV
        fVV = (fVV+one)*(maxVV-minVV)/two + minVV



    # find Tau_SGS
    RsgsUU = fUU - Uo*Uo
    RsgsUV = fUV - Uo*Vo
    RsgsVV = fVV - Vo*Vo

    # print(np.max(RsgsUU), np.max(RsgsUV), np.max(RsgsVV))
    # print(np.min(RsgsUU), np.min(RsgsUV), np.min(RsgsVV))


    # start outer loop on SIMPLE convergence
    it = 0
    res = large
    while (res>toll and it<maxIt):


        #---------------------------- solve momentum equations
        # x-direction
        Fw = A*rho*hf*(Uo            + cr(Uo, -1, 0))
        Fe = A*rho*hf*(cr(Uo,  1, 0) + Uo           )
        Fs = A*rho*hf*(Vo            + cr(Vo, -1, 0))
        Fn = A*rho*hf*(cr(Vo,  0, 1) + cr(Vo, -1, 1))
            
        Aw = Dc + hf*Fw  # hf*(nc.abs(Fw) + Fw)
        Ae = Dc - hf*Fe  # hf*(nc.abs(Fe) - Fe)
        As = Dc + hf*Fs  # hf*(nc.abs(Fs) + Fs)
        An = Dc - hf*Fn  # hf*(nc.abs(Fn) - Fn)
        Ao = rho*A*dl/delt

        Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
        iApU = one/Ap
        sU = Ao*Uo -(P - cr(P, -1, 0))*A + hf*(B + cr(B, -1, 0))  \
            - rho*A*hf*(cr(RsgsUU, 1, 0) - cr(RsgsUU, -1, 0))     \
            - rho*A*hf*(cr(RsgsUV, 0, 1) - cr(RsgsUV, 0, -1))

        itM  = 0
        resM = large
        while (resM>tollM and itM<maxIt):

            dd = sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0)
            U = solver_TDMAcyclic(-As, Ap, -An, dd, NLES)
            U = (sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0) + As*cr(U, 0, -1) + An*cr(U, 0, 1))*iApU
            resM = nc.sum(nc.abs(Ap*U - sU - Aw*cr(U, -1, 0) - Ae*cr(U, 1, 0) - As*cr(U, 0, -1) - An*cr(U, 0, 1)))
            resM = resM*iNN
            resM_cpu = convert(resM)
            if ((itM+1)%100 == 0):
                print("x-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
            itM = itM+1


        # y-direction
        Fw = A*rho*hf*(Uo             + cr(Uo, 0, -1))
        Fe = A*rho*hf*(cr(Uo,  1,  0) + cr(Uo, 1, -1))
        Fs = A*rho*hf*(cr(Vo,  0, -1) + Vo           )
        Fn = A*rho*hf*(Vo             + cr(Vo, 0,  1))

        Aw = Dc + hf*Fw  # hf*(nc.abs(Fw) + Fw)
        Ae = Dc - hf*Fe  # hf*(nc.abs(Fe) - Fe)
        As = Dc + hf*Fs  # hf*(nc.abs(Fs) + Fs)
        An = Dc - hf*Fn  # hf*(nc.abs(Fn) - Fn)
        Ao = rho*A*dl/delt

        Ap  = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
        iApV = one/Ap
        sV = Ao*Vo -(P - cr(P, 0, -1))*A + hf*(B + cr(B, 0, -1))  \
            - rho*A*hf*(cr(RsgsUV, 1, 0) - cr(RsgsUV, -1, 0))     \
            - rho*A*hf*(cr(RsgsVV, 0, 1) - cr(RsgsVV, 0, -1))

        itM  = 0
        resM = one
        while (resM>tollM and itM<maxIt):

            dd = sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0)
            V = solver_TDMAcyclic(-As, Ap, -An, dd, NLES)
            V = (sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0) + As*cr(V, 0, -1) + An*cr(V, 0, 1))*iApV
            resM = nc.sum(nc.abs(Ap*V - sV - Aw*cr(V, -1, 0) - Ae*cr(V, 1, 0) - As*cr(V, 0, -1) - An*cr(V, 0, 1)))

            resM = resM*iNN
            resM_cpu = convert(resM)
            if ((itM+1)%100 == 0):
                print("y-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
            itM = itM+1


        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = large
        Aw = rho*A*A*cr(iApU, 0,  0)
        Ae = rho*A*A*cr(iApU, 1,  0)
        As = rho*A*A*cr(iApV, 0,  0)
        An = rho*A*A*cr(iApV, 0,  1)
        Ap = Aw+Ae+As+An
        iApP = one/Ap
        So = -rho*A*(cr(U, 1, 0) - U + cr(V, 0, 1) - V)

        pc[:,:] = Z[:,:]

        itP  = 0
        resP = large
        while (resP>tollP and itP<maxIt):

            dd = So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0)
            pc = solver_TDMAcyclic(-As, Ap, -An, dd, NLES)
            pc = (So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0) + As*cr(pc, 0, -1) + An*cr(pc, 0, 1))*iApP

            resP = nc.sum(nc.abs(Ap*pc - So - Aw*cr(pc, -1, 0) - Ae*cr(pc, 1, 0) - As*cr(pc, 0, -1) - An*cr(pc, 0, 1)))
            resP = resP*iNN

            resP_cpu = convert(resP)
            if ((itP+1)%100 == 0):
                print("Pressure correction:  it {0:3d}  residuals {1:3e}".format(itP, resP_cpu))
            itP = itP+1




        #---------------------------- update values using under relaxation factors
        deltpX1 = cr(pc, -1, 0) - pc
        deltpY1 = cr(pc, 0, -1) - pc

        P  = P + alphaP*pc
        U  = U + A*iApU*deltpX1
        V  = V + A*iApV*deltpY1

        res = nc.sum(nc.abs(So))
        res = res*iNN
        res_cpu = convert(res)
        if ((it+1)%100 == 0):
            print("SIMPLE iterations:  it {0:3d}  residuals {1:3e}".format(it, res_cpu))

        it = it+1



    #---------------------------- find DNS field from GAN

    if (PROCEDURE=="DNS"):

        # find DNS and LES fields from a reference DNS time series

        # load DNS reference fields
        filename = "./results/DNS_N256/fields/fields_run0_it" + str(tstep+1) + ".npz"

        # load DNS reference fields from restart.npz file
        U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, newtotTime = load_fields(filename)

        # find LES field
        itDNS   = 0
        resDNS  = large
        if (FILTER=="Trained_filter"):

            # find max/min values and normalize
            maxU = np.max(U_DNS)
            minU = np.min(U_DNS)
            U_DNS = two*(U_DNS - minU)/(maxU- minU) - one

            maxV = np.max(V_DNS)
            minV = np.min(V_DNS)
            V_DNS = two*(V_DNS - minV)/(maxV - minV) - one

            maxP = np.max(P_DNS)
            minP = np.min(P_DNS)
            P_DNS = two*(P_DNS - minP)/(maxP - minP) - one

            # create image for TensorFlow
            itDNS   = 0
            resDNS  = large
            tU_DNS = tf.convert_to_tensor(U_DNS, dtype=np.float32)
            tV_DNS = tf.convert_to_tensor(V_DNS, dtype=np.float32)
            tP_DNS = tf.convert_to_tensor(P_DNS, dtype=np.float32)

            tU_DNS = tU_DNS[np.newaxis,np.newaxis,:,:]
            tV_DNS = tV_DNS[np.newaxis,np.newaxis,:,:]
            tP_DNS = tP_DNS[np.newaxis,np.newaxis,:,:]

            UVP_DNS = tf.concat([tU_DNS, tV_DNS, tP_DNS], 1)

            # filter
            UVP = filter(UVP_DNS, training=False)
            newU = UVP[0, 0, :, :].numpy()
            newV = UVP[0, 1, :, :].numpy()
            newP = UVP[0, 2, :, :].numpy()      

            # re-normalize new DNS and LES fields
            U_DNS = (U_DNS+one)*(maxU-minU)/two + minU
            V_DNS = (V_DNS+one)*(maxV-minV)/two + minV
            P_DNS = (P_DNS+one)*(maxP-minP)/two + minP

            newU = (newU+one)*(maxU-minU)/two + minU
            newV = (newV+one)*(maxV-minV)/two + minV
            newP = (newP+one)*(maxP-minP)/two + minP   

        elif (FILTER=="StyleGAN_layer"):

            pass

        elif (FILTER=="Gaussian"):
            rs = int(N/NLES)
            if (rs==1):
                newU = U_DNS[::rs,::rs]
                newV = V_DNS[::rs,::rs]  
                newP = P_DNS[::rs,::rs]  
            else:
                fU = sc.ndimage.gaussian_filter(U_DNS, rs, mode='grid-wrap')
                fV = sc.ndimage.gaussian_filter(V_DNS, rs, mode='grid-wrap')
                fP = sc.ndimage.gaussian_filter(P_DNS, rs, mode='grid-wrap')
                
                newU = fU[::rs,::rs]
                newV = fV[::rs,::rs]  
                newP = fP[::rs,::rs]

    else:

        # find new max/min values and normalize LES field
        maxU = np.max(U)
        minU = np.min(U)
        maxV = np.max(V)
        minV = np.min(V)
        maxP = np.max(P)
        minP = np.min(P)

        U = two*(U - minU)/(maxU - minU) - one
        V = two*(V - minV)/(maxV - minV) - one
        P = two*(P - minP)/(maxP - minP) - one

        tU = tf.convert_to_tensor(U, dtype=np.float32)
        tV = tf.convert_to_tensor(V, dtype=np.float32)
        tP = tf.convert_to_tensor(P, dtype=np.float32)

        tU = tU[np.newaxis,:,:]
        tV = tV[np.newaxis,:,:]
        tP = tP[np.newaxis,:,:]

        imgA = tf.concat([tU, tV, tP], 0)

        itDNS  = 0
        resDNS = large
        while (resDNS>tollDNS and itDNS<maxItDNS):
            if (PROCEDURE=="A1"):
                resDNS, predictions, UVP = find_latents_LES_step(latents, imgA, list_LES_trainable_variables)
            elif (PROCEDURE=="A2"):
                resDNS, predictions, UVP = find_latents_LES_step(latents, imgA, list_LES_trainable_variables)
            elif (PROCEDURE=="B1"):
                resDNS, predictions, UVP = find_latents_LES_step(latents, imgA, list_LES_trainable_variables)

            if (itDNS%100 == 0):
                lr = lr_schedule_LES(itDNS)
                print("Search LES iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))
                U_LES = UVP[0, 0, :, :].numpy()
                V_LES = UVP[0, 1, :, :].numpy()
                P_LES = UVP[0, 2, :, :].numpy()
                #W_LES = find_vorticity(U_LES, V_LES)
                #print_fields(U_LES, V_LES, P_LES, W_LES, NLES, filename="plots/plots_LES_fromGAN.png")
                #print_fields(U, V, P, W,                 NLES, filename="plots/plots_LES.png")

            itDNS = itDNS+1

        lr = lr_schedule(itDNS)
        print("Search LES iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))

        # find new DNS fields from GAN
        U_DNS = predictions[RES_LOG2-2].numpy()[0,0,:,:]
        V_DNS = predictions[RES_LOG2-2].numpy()[0,1,:,:]
        P_DNS = predictions[RES_LOG2-2].numpy()[0,2,:,:]

        # re-normalize new DNS and LES fields
        U_DNS = (U_DNS+one)*(maxU-minU)/two + minU
        V_DNS = (V_DNS+one)*(maxV-minV)/two + minV
        P_DNS = (P_DNS+one)*(maxP-minP)/two + minP    

        U = (U+one)*(maxU-minU)/two + minU
        V = (V+one)*(maxV-minV)/two + minV
        P = (P+one)*(maxP-minP)/two + minP 



        # #---------------------------- retrain GAN
        # if PROCEDURE=="B1":

        #     # Create noise for sample images
        #     if (firstRetrain):
        #         tf.random.set_seed(1)
        #         input_latent = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
        #         inputVariances = tf.constant(1.0, shape=(1, G_LAYERS), dtype=DTYPE)
        #         lr = LR
        #         mtr = np.zeros([5], dtype=DTYPE)
        #     firstRetrain = False

        #     # reaload checkpoint
        #     checkpoint.restore(tf.train.latest_checkpoint(CHKP_DIR))

        #     # starts retrain
        #     tstart = time.time()
        #     tint   = tstart
        #     for it in range(TOT_ITERATIONS):
            
        #         # take next batch
        #         input_batch = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
        #         image_batch = next(iter(dataset))
        #         mtr = distributed_train_step(input_batch, inputVariances, image_batch)

        #         # print losses
        #         if it % PRINT_EVERY == 0:
        #             tend = time.time()
        #             lr = lr_schedule(it)
        #             print ('Total time {0:3.2f} h, Iteration {1:8d}, Time Step {2:6.2f} s, ' \
        #                 'ld {3:6.2e}, ' \
        #                 'lg {4:6.2e}, ' \
        #                 'lf {5:6.2e}, ' \
        #                 'r1 {6:6.2e}, ' \
        #                 'sr {7:6.2e}, ' \
        #                 'sf {8:6.2e}, ' \
        #                 'lr {9:6.2e}, ' \
        #                 .format((tend-tstart)/3600, it, tend-tint, \
        #                 mtr[0], \
        #                 mtr[1], \
        #                 mtr[2], \
        #                 mtr[3], \
        #                 mtr[4], \
        #                 mtr[5], \
        #                 lr))
        #             tint = tend

        #             # write losses to tensorboard
        #             with train_summary_writer.as_default():
        #                 tf.summary.scalar('a/loss_disc',   mtr[0], step=it)
        #                 tf.summary.scalar('a/loss_gen',    mtr[1], step=it)
        #                 tf.summary.scalar('a/loss_filter', mtr[2], step=it)
        #                 tf.summary.scalar('a/r1_penalty',  mtr[3], step=it)
        #                 tf.summary.scalar('a/score_real',  mtr[4], step=it)
        #                 tf.summary.scalar('a/score_fake',  mtr[5], step=it)                                
        #                 tf.summary.scalar('a/lr',              lr, step=it)

        #     #save the model
        #     checkpoint.save(file_prefix = CHKP_PREFIX)






    #---------------------------- solve transport equation for passive scalar
    if (PASSIVE):

        # solve iteratively
        Fw = A*rho*cr(U, -1, 0)
        Fe = A*rho*U
        Fs = A*rho*cr(V, 0, -1)
        Fn = A*rho*V

        Aw = DiffCoef + hf*(nc.abs(Fw) + Fw)
        Ae = DiffCoef + hf*(nc.abs(Fe) - Fe)
        As = DiffCoef + hf*(nc.abs(Fs) + Fs)
        An = DiffCoef + hf*(nc.abs(Fn) - Fn)
        Ao = rho*A*dl/delt

        Ap = Ao + (Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs))
        iApC = one/Ap

        itC  = 0
        resC = large
        while (resC>tollC and itC<maxIt):
            dd = Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0)
            C = solver_TDMAcyclic(-As, Ap, -An, dd, NLES)
            C = (Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0) + As*cr(C, 0, -1) + An*cr(C, 0, 1))*iApC

            resC = nc.sum(nc.abs(Ap*C - Ao*Co - Aw*cr(C, -1, 0) - Ae*cr(C, 1, 0) - As*cr(C, 0, -1) - An*cr(C, 0, 1)))
            resC = resC*iNN
            resC_cpu = convert(resC)
            if ((itC+1)%100 == 0):
                print("Passive scalar:  it {0:3d}  residuals {1:3e}".format(itC, resC_cpu))
            itC = itC+1

        # find integral of passive scalar
        totSca = convert(nc.sum(C))
        maxSca = convert(nc.max(C))
        print("Tot scalar {0:.8e}  max scalar {1:3e}".format(totSca, maxSca))




    #---------------------------- print update and save fields
    if (it==maxIt):
        print("Attention: SIMPLE solver not converged!!!")
        exit()

    else:

        if (PROCEDURE=="DNS"):
            # find new delt based on the DNS list
            delt = newtotTime -  totTime
        else:
            # find new delt based on Courant number
            cdelt = CNum*dl/(sqrt(nc.max(U_DNS)*nc.max(U_DNS) + nc.max(V_DNS)*nc.max(V_DNS))+small)
            delt = convert(cdelt)
            delt = min(delt, maxDelt)

        totTime = totTime + delt
        tstep = tstep+1
        its = it

        # check divergence
        div = rho*A*nc.sum(nc.abs(cr(U_DNS, 1, 0) - U_DNS + cr(V_DNS, 0, 1) - V_DNS))
        div = div*iNN
        div_cpu = convert(div)  

        # print values
        tend = time.time()
        if (tstep%print_res == 0):
            wtime = (tend-tstart)
            print("Wall time [s] {0:6.1f}  steps {1:3d}  time {2:5.2e}  delt {3:5.2e}  resM {4:5.2e}  "\
                "resP {5:5.2e}  resC {6:5.2e}  res {7:5.2e}  its {8:3d}  div {9:5.2e}"       \
            .format(wtime, tstep, totTime, delt, resM_cpu, resP_cpu, \
            resC_cpu, res_cpu, its, div_cpu))


        # track center point velocities and pressure
        DNS_cv[tstep,0] = totTime
        DNS_cv[tstep,1] = U_DNS[N//2, N//2]
        DNS_cv[tstep,2] = V_DNS[N//2, N//2]
        DNS_cv[tstep,3] = P_DNS[N//2, N//2]

        LES_cv[tstep,0] = totTime
        LES_cv[tstep,1] = U[NLES//2, NLES//2]
        LES_cv[tstep,2] = V[NLES//2, NLES//2]
        LES_cv[tstep,3] = P[NLES//2, NLES//2]

        if (PROCEDURE=="DNS"):
            LES_cv_fromDNS[tstep,0] = totTime
            LES_cv_fromDNS[tstep,1] = newU[NLES//2, NLES//2]
            LES_cv_fromDNS[tstep,2] = newV[NLES//2, NLES//2]
            LES_cv_fromDNS[tstep,3] = newP[NLES//2, NLES//2]
        else:
            LES_cv_fromDNS[tstep,0] = totTime
            LES_cv_fromDNS[tstep,1] = U[NLES//2, NLES//2]
            LES_cv_fromDNS[tstep,2] = V[NLES//2, NLES//2]
            LES_cv_fromDNS[tstep,3] = P[NLES//2, NLES//2]


        # check min and max values
        u_max = nc.max(nc.abs(U_DNS))
        v_max = nc.max(nc.abs(V_DNS))
        u_max_cpu = convert(u_max)
        v_max_cpu = convert(v_max)
        uv_max = [u_max_cpu, v_max_cpu]
        if uv_max[0] > uRef or uv_max[1] > uRef:
            save_vel_violations("v_viol/v_viol.txt", uv_max, tstep)


        # plot, save, find spectrum fields
        if (len(te)>0):

            #loop for turnover times(te) and respective time in seconds(te_s)
            for s in range(len(te_s)):
                if (totTime<te_s[s]+hf*delt and totTime>te_s[s]-hf*delt):

                    # find vorticity
                    W_DNS = find_vorticity(U_DNS, V_DNS)
                    W     = find_vorticity(U,V)

                    # save fields
                    save_fields(totTime, U, V, P, C, B, W, "fields/fields_" + str(te[s]) + "te.npz")

                    # print fields
                    print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N   ,"plots/plots_DNS_fromGAN_" + str(te[s]) + "te.png")
                    print_fields(U,     V,     P,     W,     NLES,"plots/plots_LES_"         + str(te[s]) + "te.png")

                    #print spectrum
                    plot_spectrum(U_DNS, V_DNS, L, "energy/energy_DNS_fromGAN_" + str(te[s]) + "te.txt")
                    plot_spectrum(U, V, L,         "energy/energy_LES_"         + str(te[s]) + "te.txt")
        else:
    
            tail = "it{0:d}".format(tstep)

            # find vorticity
            W_DNS = find_vorticity(U_DNS, V_DNS)
            W     = find_vorticity(U,V)

            # save fields
            if (tstep%print_ckp == 0):
                save_fields(totTime, U, V, P, C, B, W, "fields/fields_" + tail + ".npz")

            # print fields
            if (tstep%print_img == 0):
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N,    "plots/plots_DNS_fromGAN_" + tail + ".png")
                print_fields(U,     V,     P,     W,     NLES, "plots/plots_LES_"         + tail + ".png")

            # print spectrum
            if (tstep%print_spe == 0):
                plot_spectrum(U_DNS, V_DNS, L, "energy/energy_spectrum_DNS_fromGAN_" + tail + ".txt")
                plot_spectrum(U,     V,     L, "energy/energy_spectrum_LES_"         + tail + ".txt")


# end of the simulation


# plot, save, find spectrum fields
if (len(te)==0):
    tail = "it{0:d}".format(tstep)

    # save images
    W_DNS = find_vorticity(U_DNS, V_DNS)
    W     = find_vorticity(U, V)
    print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N,    "plots/plots_DNS_fromGAN_" + tail + ".png")
    print_fields(U,     V,     P,     W,     NLES, "plots/plots_LES_"         + tail + ".png")

    # write checkpoint
    W = find_vorticity(U, V)
    save_fields(totTime, U, V, P, C, B, W, "fields/fields_" + tail + ".npz")

    # print spectrum
    plot_spectrum(U_DNS, V_DNS, L, "energy/energy_spectrum_DNS_fromGAN_" + tail + ".txt")
    plot_spectrum(U,     V,     L, "energy/energy_spectrum_LES_"         + tail + ".txt")

# save center values
filename = "DNS_fromGAN_center_values.txt"
np.savetxt(filename, np.c_[DNS_cv[0:tstep+1,0], DNS_cv[0:tstep+1,1], DNS_cv[0:tstep+1,2], DNS_cv[0:tstep+1,3]], fmt='%1.4e')

filename = "LES_center_values.txt"
np.savetxt(filename, np.c_[LES_cv[0:tstep+1,0], LES_cv[0:tstep+1,1], LES_cv[0:tstep+1,2], LES_cv[0:tstep+1,3]], fmt='%1.4e')

filename = "LES_fromGAN_center_values.txt"
np.savetxt(filename, np.c_[LES_cv_fromDNS[0:tstep+1,0], LES_cv_fromDNS[0:tstep+1,1], LES_cv_fromDNS[0:tstep+1,2], LES_cv_fromDNS[0:tstep+1,3]], fmt='%1.4e')


print("Simulation successfully completed!")
