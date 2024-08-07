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

DTYPE_LES = DTYPEN_LES

os.chdir('../')
from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *
os.chdir('./LES_Solvers/')

DTYPE = DTYPE_LES  # this is only because the StyleGAN is trained with float32 usually






#---------------------------- local variables
N_LES          = 2**(RES_LOG2-FIL)
iNN_LES        = one/(N_LES*N_LES)
SIG            = 8*int(N/N_LES)  # Gaussian (tf and np) filter sigma
DW             = int(N/N_LES)      # downscaling factor
PROCEDURE      = "A1"
WL_CHKP_DIR    = "../utilities/wl_checkpoints/"
WL_CHKP_PREFIX = os.path.join(WL_CHKP_DIR, "ckpt")
FILE_REAL      = "../../../results/decayIsoTurb_2D/paper_results/tollm7/DNS_N256/fields/fields_run0_it100.npz"

if PROCEDURE=="A1":
    FILTER       = "Gaussian_tf"
    USE_DLATENTS = True
    INIT_BC      = 0
    WL_IRESTART  = False
elif PROCEDURE=="A2":
    FILTER       = "StyleGAN_layer"
    USE_DLATENTS = True
    INIT_BC      = 0
    WL_IRESTART  = True
elif PROCEDURE=="B2":
    FILTER       = "StyleGAN_layer"
    USE_DLATENTS = True
    INIT_BC      = 0
    firstRetrain = True
    WL_IRESTART  = True
elif PROCEDURE=="DNS":
    FILTER       = "Gaussian_np"
    USE_DLATENTS = False
    INIT_BC      = 1
    WL_IRESTART  = False

print("Procedure ", PROCEDURE)

Uo = nc.zeros([N_LES,N_LES], dtype=DTYPE)   # old x-velocity
Vo = nc.zeros([N_LES,N_LES], dtype=DTYPE)   # old y-velocity
Po = nc.zeros([N_LES,N_LES], dtype=DTYPE)   # old pressure field
Co = nc.zeros([N_LES,N_LES], dtype=DTYPE)   # old passive scalar

Uo_DNS = nc.zeros([N,N], dtype=DTYPE)     # old x-velocity DNS
Vo_DNS = nc.zeros([N,N], dtype=DTYPE)     # old y-velocity DNS
Po_DNS = nc.zeros([N,N], dtype=DTYPE)     # old pressure DNS

pc = nc.zeros([N_LES,N_LES], dtype=DTYPE)   # pressure correction
Z  = nc.zeros([N_LES,N_LES], dtype=DTYPE)   # zero array
C  = np.zeros([N_LES,N_LES], dtype=DTYPE)   # scalar
B  = np.zeros([N_LES,N_LES], dtype=DTYPE)   # body force
P  = np.zeros([N_LES,N_LES], dtype=DTYPE)   # body force

DNS_cv = np.zeros([totSteps+1, 4])
LES_cv = np.zeros([totSteps+1, 4])
LES_cv_fromDNS = np.zeros([totSteps+1, 4])

U_diff = np.zeros([N, N], dtype=DTYPE)
V_diff = np.zeros([N, N], dtype=DTYPE)
P_diff = np.zeros([N, N], dtype=DTYPE)
W_diff = np.zeros([N, N], dtype=DTYPE)

minMaxUVP = np.zeros((1,6), dtype="float32")







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

DiffCoef = np.full([N_LES, N_LES], Dc)
NL_DNS   = np.zeros([16, N, N])
NL       = np.zeros([16, N_LES, N_LES])

if (len(te)>0):
    tail = "0te"
else:
    tail = "it0"

dir_train_log        = 'logs/DNS_solver/'
train_summary_writer = tf.summary.create_file_writer(dir_train_log)



# loading StyleGAN checkpoint
managerCheckpoint = tf.train.CheckpointManager(checkpoint, '../' + CHKP_DIR, max_to_keep=1)
checkpoint.restore(managerCheckpoint.latest_checkpoint)

if managerCheckpoint.latest_checkpoint:
    print("StyleGAN restored from {}".format(managerCheckpoint.latest_checkpoint, max_to_keep=1))
else:
    print("Initializing StyleGAN from scratch.")

time.sleep(3)



# create variable synthesis model
layer_LES = layer_wlatent_LES()

zlatents     = tf.keras.Input(shape=([LATENT_SIZE]), dtype=DTYPE)
wlatents     = mapping(zlatents)
wlatents_LES = layer_LES(wlatents)
outputs      = synthesis(wlatents_LES, training=False)
wl_synthesis = tf.keras.Model(inputs=zlatents, outputs=outputs)


# define optimizer for LES search
if (lr_LES_POLICY=="EXPONENTIAL"):
    lr_schedule_LES  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_LES,
        decay_steps=lr_LES_STEP,
        decay_rate=lr_LES_RATE,
        staircase=lr_LES_EXP_ST)
elif (lr_LES_POLICY=="PIECEWISE"):
    lr_schedule_LES = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_LES_BOUNDS, lr_LES_VALUES)
opt_LES = tf.keras.optimizers.Adamax(learning_rate=lr_schedule_LES)




# define checkpoints wl_synthesis and filter
checkpoint_wl = tf.train.Checkpoint(wl_synthesis=wl_synthesis)
managerCheckpoint_wl = tf.train.CheckpointManager(checkpoint_wl, CHKP_DIR_WL, max_to_keep=1)



# add latent space to trainable variables
if (not TUNE_NOISE):
    ltv_DNS = []
    ltv_LES = []

for variable in layer_LES.trainable_variables:
    ltv_DNS.append(variable)
    ltv_LES.append(variable)


print("\n DNS variables:")
for variable in ltv_DNS:
    print(variable.name)

print("\n LES variables:")
for variable in ltv_LES:
    print(variable.name)

time.sleep(3)



#---------------------------- local functions
@tf.function
def step_find_latents_LES(latents, fimgA, ltv):
    with tf.GradientTape() as tape_LES:

        # find predictions
        predictions = wl_synthesis(latents, training=False)
        UVP_DNS = predictions[RES_LOG2-2]
        UVP_LES = predictions[RES_LOG2-FIL-2]

        # normalize
        U_DNS = UVP_DNS[0, 0, :, :]
        V_DNS = UVP_DNS[0, 1, :, :]
        P_DNS = UVP_DNS[0, 2, :, :]

        U_DNS = 2.0*(U_DNS - tf.math.reduce_min(U_DNS))/(tf.math.reduce_max(U_DNS) - tf.math.reduce_min(U_DNS)) - 1.0
        V_DNS = 2.0*(V_DNS - tf.math.reduce_min(V_DNS))/(tf.math.reduce_max(V_DNS) - tf.math.reduce_min(V_DNS)) - 1.0
        P_DNS = 2.0*(P_DNS - tf.math.reduce_min(P_DNS))/(tf.math.reduce_max(P_DNS) - tf.math.reduce_min(P_DNS)) - 1.0

        # convert back to 1 tensor
        U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
        V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
        P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]

        UVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], 1)

        # filter        
        fUVP_DNS = filters[IFIL](UVP_DNS, training=False)

        # find residuals
        resDNS = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, fimgA))
        resLES = tf.math.reduce_mean(tf.math.squared_difference(UVP_LES, fimgA))

        resREC = resDNS + resLES

        # aply gradients
        gradients_LES = tape_LES.gradient(resREC, ltv)
        opt_LES.apply_gradients(zip(gradients_LES, ltv))

        # find filter loss
        loss_fil    = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, UVP_LES))

    return resREC, resLES, resDNS, UVP_DNS, loss_fil




#---------------------------- initialize flow
tstart = time.time()


# load DNS reference fields
U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, totTime = load_fields(FILE_REAL, DNSrun=True)

W_DNS = find_vorticity(U_DNS, V_DNS)
print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_org.png")

# find max/min values and normalize
minMaxUVP[0,0] = np.max(U_DNS)
minMaxUVP[0,1] = np.min(U_DNS)
minMaxUVP[0,2] = np.max(V_DNS)
minMaxUVP[0,3] = np.min(V_DNS)
minMaxUVP[0,4] = np.max(P_DNS)
minMaxUVP[0,5] = np.min(P_DNS)
tminMaxUVP = tf.convert_to_tensor(minMaxUVP, dtype="float32")

# create image for TensorFlow
tU_DNS = tf.convert_to_tensor(U_DNS, dtype=np.float32)
tV_DNS = tf.convert_to_tensor(V_DNS, dtype=np.float32)
tP_DNS = tf.convert_to_tensor(P_DNS, dtype=np.float32)

tU_DNS = tU_DNS[tf.newaxis,:,:]
tV_DNS = tV_DNS[tf.newaxis,:,:]
tP_DNS = tP_DNS[tf.newaxis,:,:]

imgA_DNS = tf.concat([tU_DNS, tV_DNS, tP_DNS], 0)

# prepare latent space
if (RESTART_WL):
    # loading wl_synthesis checkpoint and zlatents
    if managerCheckpoint_wl.latest_checkpoint:
        print("wl_synthesis restored from {}".format(managerCheckpoint_wl.latest_checkpoint, max_to_keep=1))
    else:
        print("Initializing wl_synthesis from scratch.")
    data = np.load("results_reconstruction/zlatents.npz")
    zlatents = data["zlatents"]
else:
    zlatents = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN, seed=SEED_RESTART)

# save checkpoint for wl_synthesis
managerCheckpoint_wl.save()

# start research on the latent space
it = 0
resREC = large
opt_LES.initial_learning_rate = lr_LES      # reload initial learning rate
while (resREC>tollLES and it<lr_LES_maxIt):                
    lr = lr_schedule_LES(it)
    resREC, resLES, resDNS, UVP_DNS, loss_fil = step_find_latents_LES(zlatents, fimgA, ltv_LES)


    # print residuals and fields
    if ((it%100==0 and it!=0) or (it%100==0 and k==0)):

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
            tf.summary.scalar('resREC',   resREC,   step=it)
            tf.summary.scalar('resDNS',   resDNS,   step=it)
            tf.summary.scalar('resLES',   resLES,   step=it)
            tf.summary.scalar('loss_fil', loss_fil, step=it)
            tf.summary.scalar('lr',       lr,       step=it)

        if (it%1000==-1):

            filename = "results_reconstruction/plots/Plots_DNS_fromGAN.png"
            # filename = "results_reconstruction/plots/Plots_DNS_fromGAN_" + str(it) + ".png"
            print_fields_3(U_DNS, V_DNS, P_DNS, N=N_DNS, filename=filename, \
                    Umin=-1.0, Umax=1.0, Vmin=-1.0, Vmax=1.0, Pmin=-1.0, Pmax=1.0)

    it = it+1


# print final residuals
lr = lr_schedule_LES(it)
tend = time.time()
print("LES iterations:  time {0:3e}   step {1:4d}  it {2:6d}  residuals {3:3e} resLES {4:3e}  resDNS {5:3e} loss_fill {6:3e}  lr {7:3e} " \
    .format(tend-tstart, k, it, resREC.numpy(), resLES.numpy(), resDNS.numpy(), loss_fil, lr))


# find numpy arrays from GAN inference
U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
# P_DNS = UVP_DNS[0, 2, :, :].numpy()  # Note: this is important, We only want U and V from the GAN!
W_DNS = find_vorticity(U_DNS, V_DNS)



# find LES field
predictions = wl_synthesis(zlatents, training=False)
UVP_LES = predictions[RES_LOG2-FIL-2]

U_LES = UVP_LES[0, 0, :, :].numpy()
V_LES = UVP_LES[0, 1, :, :].numpy()
P_LES = np.zeros([N_LES, N_LES])


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
div = div*iNN_LES
div_cpu = convert(div)

# find new delt based on Courant number
cdelt = CNum*dl/(sqrt(nc.max(U_DNS)*nc.max(U_DNS) + nc.max(V_DNS)*nc.max(V_DNS))+small)
delt = convert(cdelt)
delt = min(delt, maxDelt)


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


    # calculate coefficients x-direction
    Fw = A*rho*hf*(Uo_DNS            + cr(Uo_DNS, -1, 0))
    Fe = A*rho*hf*(cr(Uo_DNS,  1, 0) + Uo_DNS           )
    Fs = A*rho*hf*(Vo_DNS            + cr(Vo_DNS, -1, 0))
    Fn = A*rho*hf*(cr(Vo_DNS,  0, 1) + cr(Vo_DNS, -1, 1))

    # find non linear terms in x-direction
    NL_DNS[0,:,:] = Fe
    NL_DNS[1,:,:] = Fw
    NL_DNS[2,:,:] = Fn
    NL_DNS[3,:,:] = Fs
    NL_DNS[4,:,:] = Fe*hf*(cr(Uo_DNS, 1, 0) + Uo_DNS            )
    NL_DNS[5,:,:] = Fw*hf*(Uo_DNS           + cr(Uo_DNS, -1,  0))
    NL_DNS[6,:,:] = Fn*hf*(cr(Uo_DNS, 0, 1) + Uo_DNS            ) 
    NL_DNS[7,:,:] = Fs*hf*(Uo_DNS           + cr(Uo_DNS,  0, -1))


    # calculate coefficients y-direction
    Fw = A*rho*hf*(Uo_DNS             + cr(Uo_DNS, 0, -1))
    Fe = A*rho*hf*(cr(Uo_DNS,  1,  0) + cr(Uo_DNS, 1, -1))
    Fs = A*rho*hf*(cr(Vo_DNS,  0, -1) + Vo_DNS           )
    Fn = A*rho*hf*(Vo_DNS             + cr(Vo_DNS, 0,  1))

    # find non linear terms in y-direction
    NL_DNS[ 8,:,:] = Fe
    NL_DNS[ 9,:,:] = Fw
    NL_DNS[10,:,:] = Fn
    NL_DNS[11,:,:] = Fs
    NL_DNS[12,:,:] = Fe*hf*(cr(Vo_DNS, 1, 0) + Vo_DNS            ) 
    NL_DNS[13,:,:] = Fw*hf*(Vo_DNS           + cr(Vo_DNS, -1,  0))
    NL_DNS[14,:,:] = Fn*hf*(cr(Vo_DNS, 0, 1) + Vo_DNS            )
    NL_DNS[15,:,:] = Fs*hf*(Vo_DNS           + cr(Vo_DNS,  0, -1))



    # filter them
    tNL_DNS = tf.convert_to_tensor(NL_DNS[:,:,:,tf.newaxis])
    NL = filters[IFIL](tNL_DNS, training=False)
    fUU = NL[0, 0, :, :].numpy()
    fUV = NL[0, 1, :, :].numpy()
    fVV = NL[0, 2, :, :].numpy()

 
    # find Tau_SGS
    dRsgsUU_dx = NL[ 4,:,:] - NL[ 5,:,:] - NL[ 0,:,:]*hf*(cr(Uo, 1, 0) + Uo) + NL[ 1,:,:]*hf*(Uo + cr(Uo, -1,  0)) 
    dRsgsUV_dy = NL[ 6,:,:] - NL[ 7,:,:] - NL[ 2,:,:]*hf*(cr(Uo, 0, 1) + Uo) + NL[ 3,:,:]*hf*(Uo + cr(Uo,  0, -1))
    dRsgsVU_dx = NL[12,:,:] - NL[13,:,:] - NL[ 8,:,:]*hf*(cr(Vo, 1, 0) + Vo) + NL[ 9,:,:]*hf*(Vo + cr(Vo, -1,  0))
    dRsgsVV_dy = NL[14,:,:] - NL[15,:,:] - NL[10,:,:]*hf*(cr(Vo, 0, 1) + Vo) + NL[11,:,:]*hf*(Vo + cr(Vo,  0, -1))

    # print(np.max(dRsgsUU_dx), np.max(dRsgsUV_dy), np.max(dRsgsVU_dx), np.max(dRsgsVV_dy))
    # print(np.min(dRsgsUU_dx), np.min(dRsgsUV_dy), np.min(dRsgsVU_dx), np.min(dRsgsVV_dy))



    # start outer loop on SIMPLE convergence
    it = 0
    res = large
    while (res>toll and it<maxItDNS):


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
        sU = Ao*Uo -(P - cr(P, -1, 0))*A + hf*(B + cr(B, -1, 0)) - dRsgsUU_dx - dRsgsUV_dy

        itM  = 0
        resM = large
        while (resM>tollM and itM<maxItDNS):

            dd = sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0)
            U = solver_TDMAcyclic(-As, Ap, -An, dd, N_LES)
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
        sV = Ao*Vo -(P - cr(P, 0, -1))*A + hf*(B + cr(B, 0, -1)) - dRsgsVU_dx - dRsgsVV_dy

        itM  = 0
        resM = one
        while (resM>tollM and itM<maxItDNS):

            dd = sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0)
            V = solver_TDMAcyclic(-As, Ap, -An, dd, N_LES)
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
        while (resP>tollP and itP<maxItDNS):

            dd = So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0)
            pc = solver_TDMAcyclic(-As, Ap, -An, dd, N_LES)
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
        filename = "./paper_results/tollm5/DNS_N256_1000it/fields/fields_run0_it" + str(tstep+1) + ".npz"

        # load DNS reference fields from restart.npz file
        U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, newtotTime = load_fields(filename, DNSrun=True)

        # find max/min values and normalize
        maxU_DNS = np.max(U_DNS)
        minU_DNS = np.min(U_DNS)
        maxV_DNS = np.max(V_DNS)
        minV_DNS = np.min(V_DNS)
        maxP_DNS = np.max(P_DNS)
        minP_DNS = np.min(P_DNS)

        # U_DNS = two*(U_DNS - minU_DNS)/(maxU_DNS - minU_DNS) - one
        # V_DNS = two*(V_DNS - minV_DNS)/(maxV_DNS - minV_DNS) - one
        # P_DNS = two*(P_DNS - minP_DNS)/(maxP_DNS - minP_DNS) - one

        # create tensor
        tU_DNS = tf.convert_to_tensor(U_DNS, dtype=np.float32)
        tV_DNS = tf.convert_to_tensor(V_DNS, dtype=np.float32)
        tP_DNS = tf.convert_to_tensor(P_DNS, dtype=np.float32)

        tU_DNS = tU_DNS[tf.newaxis,tf.newaxis,:,:]
        tV_DNS = tV_DNS[tf.newaxis,tf.newaxis,:,:]
        tP_DNS = tP_DNS[tf.newaxis,tf.newaxis,:,:]

        UVP_DNS = tf.concat([tU_DNS, tV_DNS, tP_DNS], 1)

        # # rescale DNS field
        # U_DNS = (U_DNS+one)*(maxU_DNS-minU_DNS)/two + minU_DNS
        # V_DNS = (V_DNS+one)*(maxV_DNS-minV_DNS)/two + minV_DNS
        # P_DNS = (P_DNS+one)*(maxP_DNS-minP_DNS)/two + minP_DNS

        # find LES field
        if (FILTER=="Trained_filter"):
            UVP = filter(UVP_DNS, training=False)
            newU = UVP[0, 0, :, :].numpy()
            newV = UVP[0, 1, :, :].numpy()
            newP = UVP[0, 2, :, :].numpy()

        elif (FILTER=="StyleGAN_layer"):
            newU = predictions[RES_LOG2-FIL-2][0, 0, :, :].numpy()
            newV = predictions[RES_LOG2-FIL-2][0, 1, :, :].numpy()
            newP = predictions[RES_LOG2-FIL-2][0, 2, :, :].numpy()

        elif (FILTER=="Gaussian_tf"):

            # separate DNS fields
            rs = SIG
            U_DNS_t = tf.convert_to_tensor(U_DNS[tf.newaxis,:,:,tf.newaxis])
            V_DNS_t = tf.convert_to_tensor(V_DNS[tf.newaxis,:,:,tf.newaxis])
            P_DNS_t = tf.convert_to_tensor(P_DNS[tf.newaxis,:,:,tf.newaxis])

            # preprare Gaussian Kernel
            gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            gauss_kernel = tf.cast(gauss_kernel, dtype=U_DNS_t.dtype)

            # add padding
            pleft   = 4*rs
            pright  = 4*rs
            ptop    = 4*rs
            pbottom = 4*rs

            U_DNS_t = periodic_padding_flexible(U_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
            V_DNS_t = periodic_padding_flexible(V_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))
            P_DNS_t = periodic_padding_flexible(P_DNS_t, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

            # convolve
            fU = tf.nn.conv2d(U_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
            fV = tf.nn.conv2d(V_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
            fP = tf.nn.conv2d(P_DNS_t, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

            # downscale
            newU = fU[0,::DW,::DW,0].numpy()
            newV = fV[0,::DW,::DW,0].numpy()
            newP = fP[0,::DW,::DW,0].numpy()

        elif (FILTER=="Gaussian_np"):

            rs = SIG
            if (rs==1):
                newU = U_DNS[:,:]
                newV = V_DNS[:,:]
                newP = P_DNS[:,:]
            else:
                fU = sc.ndimage.gaussian_filter(U_DNS, rs, mode='wrap')
                fV = sc.ndimage.gaussian_filter(V_DNS, rs, mode='wrap')
                fP = sc.ndimage.gaussian_filter(P_DNS, rs, mode='wrap')
                
                newU = fU[::DW,::DW]
                newV = fV[::DW,::DW]
                newP = fP[::DW,::DW]



            # #-----------------------------------------fully implicit Reynolds stress term: remember to tab on the left the lines above
            # # calculate coefficients x-direction
            # Fw = A*rho*hf*(Uo_DNS            + cr(Uo_DNS, -1, 0))
            # Fe = A*rho*hf*(cr(Uo_DNS,  1, 0) + Uo_DNS           )
            # Fs = A*rho*hf*(Vo_DNS            + cr(Vo_DNS, -1, 0))
            # Fn = A*rho*hf*(cr(Vo_DNS,  0, 1) + cr(Vo_DNS, -1, 1))

            # # find non linear terms in x-direction
            # NL_DNS[0,:,:] = Fe
            # NL_DNS[1,:,:] = Fw
            # NL_DNS[2,:,:] = Fn
            # NL_DNS[3,:,:] = Fs
            # NL_DNS[4,:,:] = Fe*hf*(cr(U_DNS, 1, 0) + U_DNS            )
            # NL_DNS[5,:,:] = Fw*hf*(U_DNS           + cr(U_DNS, -1,  0))
            # NL_DNS[6,:,:] = Fn*hf*(cr(U_DNS, 0, 1) + U_DNS            ) 
            # NL_DNS[7,:,:] = Fs*hf*(U_DNS           + cr(U_DNS,  0, -1))


            # # calculate coefficients y-direction
            # Fw = A*rho*hf*(Uo_DNS             + cr(Uo_DNS, 0, -1))
            # Fe = A*rho*hf*(cr(Uo_DNS,  1,  0) + cr(Uo_DNS, 1, -1))
            # Fs = A*rho*hf*(cr(Vo_DNS,  0, -1) + Vo_DNS           )
            # Fn = A*rho*hf*(Vo_DNS             + cr(Vo_DNS, 0,  1))

            # # find non linear terms in y-direction
            # NL_DNS[ 8,:,:] = Fe
            # NL_DNS[ 9,:,:] = Fw
            # NL_DNS[10,:,:] = Fn
            # NL_DNS[11,:,:] = Fs
            # NL_DNS[12,:,:] = Fe*hf*(cr(V_DNS, 1, 0) + V_DNS            ) 
            # NL_DNS[13,:,:] = Fw*hf*(V_DNS           + cr(V_DNS, -1,  0))
            # NL_DNS[14,:,:] = Fn*hf*(cr(V_DNS, 0, 1) + V_DNS            )
            # NL_DNS[15,:,:] = Fs*hf*(V_DNS           + cr(V_DNS,  0, -1))




            # # filter them
            # if (FILTER=="Trained_filter"):

            #     tNL_DNS = tf.convert_to_tensor(NL_DNS[:,:,:,tf.newaxis])
            #     NL = filter(tNL_DNS, training=False)
            #     fUU = NL[0, 0, :, :].numpy()
            #     fUV = NL[0, 1, :, :].numpy()
            #     fVV = NL[0, 2, :, :].numpy()

            # elif (FILTER=="StyleGAN_layer"):

            #     # prepare fields
            #     rs = SIG
            #     pad = 4*rs
            #     tNL_DNS = tf.convert_to_tensor(NL_DNS[:,:,:,tf.newaxis])

            #     # preprare Gaussian Kernel
            #     gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
            #     gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            #     gauss_kernel = tf.cast(gauss_kernel, dtype=tNL_DNS.dtype)

            #     # add padding, convolve and downscale
            #     for i in range(16):
            #         tNL_DNS[i,:,:,:] = periodic_padding_flexible(tNL_DNS[i,:,:,:], axis=(1,2), padding=([pad, pad], [pad, pad]))
            #         tNL_DNS[i,:,:,:] = tf.nn.conv2d(tNL_DNS[i,:,:,:][tf.newaxis,:,:,:], gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
            #         NL[i,:,:] = tNL_DNS[i,::DW,::DW,0].numpy()

            # elif (FILTER=="Gaussian_tf"):

            #     # prepare fields
            #     rs = SIG
            #     pad = 4*rs
            #     tNL_DNS = tf.convert_to_tensor(NL_DNS[:,:,:,tf.newaxis])

            #     # preprare Gaussian Kernel
            #     gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
            #     gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            #     gauss_kernel = tf.cast(gauss_kernel, dtype=tNL_DNS.dtype)

            #     # add padding, convolve and downscale
            #     for i in range(16):
            #         tNL_DNS[i,:,:,:] = periodic_padding_flexible(tNL_DNS[i,:,:,:], axis=(1,2), padding=([pad, pad], [pad, pad]))
            #         tNL_DNS[i,:,:,:] = tf.nn.conv2d(tNL_DNS[i,:,:,:][tf.newaxis,:,:,:], gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
            #         NL[i,:,:] = tNL_DNS[i,::DW,::DW,0].numpy()

            # elif (FILTER=="Gaussian_np"):

            #     # prepare fields
            #     rs = SIG
            #     for i in range(16):
            #         NL_DNS[i,:,:] = sc.ndimage.gaussian_filter(NL_DNS[i,:,:], rs, mode='wrap')
            #         NL[i,:,:] = NL_DNS[i,::DW,::DW]
        
            # # find Tau_SGS
            # dRsgsUU_dx = NL[ 4,:,:] - NL[ 5,:,:] - NL[ 0,:,:]*hf*(cr(U, 1, 0) + U) + NL[ 1,:,:]*hf*(U + cr(U, -1,  0)) 
            # dRsgsUV_dy = NL[ 6,:,:] - NL[ 7,:,:] - NL[ 2,:,:]*hf*(cr(U, 0, 1) + U) + NL[ 3,:,:]*hf*(U + cr(U,  0, -1))
            # dRsgsVU_dx = NL[12,:,:] - NL[13,:,:] - NL[ 8,:,:]*hf*(cr(V, 1, 0) + V) + NL[ 9,:,:]*hf*(V + cr(V, -1,  0))
            # dRsgsVV_dy = NL[14,:,:] - NL[15,:,:] - NL[10,:,:]*hf*(cr(V, 0, 1) + V) + NL[11,:,:]*hf*(V + cr(V,  0, -1))




    else:

        # find new max/min values and normalize LES field
        minMaxUVP[0,0] = np.max(U_DNS)
        minMaxUVP[0,1] = np.min(U_DNS)
        minMaxUVP[0,2] = np.max(V_DNS)
        minMaxUVP[0,3] = np.min(V_DNS)
        minMaxUVP[0,4] = np.max(P_DNS)
        minMaxUVP[0,5] = np.min(P_DNS)
        tminMaxUVP = tf.convert_to_tensor(minMaxUVP)

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

        tU = tU[tf.newaxis,:,:]
        tV = tV[tf.newaxis,:,:]
        tP = tP[tf.newaxis,:,:]

        imgA = tf.concat([tU, tV, tP], 0)

        itDNS  = 0
        resDNS = large
        while ((resDNS>tollDNS or div_DNS>toll or div_LES>toll) and itDNS<maxItREC):
            resDNS, predictions, UVP_DNS, UVP, div_DNS, div_LES = find_latents_LES(latents, tminMaxUVP, imgA, list_LES_trainable_variables)

            if (itDNS%100 == 0):
                lr = lr_schedule_LES(itDNS)
                print("Search LES iterations:  it {0:3d}  residuals {1:3e}  div_DNS {2:3e}  div_LES {3:3e}  lr {4:3e} ".format(itDNS, resDNS, div_DNS, div_LES, lr))
                # U_LES = UVP[0, 0, :, :].numpy()
                # V_LES = UVP[0, 1, :, :].numpy()
                # P_LES = UVP[0, 2, :, :].numpy()
                # W_LES = find_vorticity(U_LES, V_LES)
                # print_fields(U_LES, V_LES, P_LES, W_LES, N_LES, filename="plots/plots_LES_fromGAN.png")
                # print_fields(U, V, P, W,                 N_LES, filename="plots/plots_LES.png")

            itDNS = itDNS+1

        lr = lr_schedule(itDNS)
        print("Search LES iterations:  it {0:3d}  residuals {1:3e}  div_DNS {2:3e}  div_LES {3:3e}  lr {4:3e} ".format(itDNS, resDNS, div_DNS, div_LES, lr))


        # # find rescaling factors
        # itDNS  = 0
        # div_DNS = large
        # loss_DNS = large
        # while (div_DNS>toll and itDNS<maxItREC):
        #     if (PROCEDURE=="A1"):
        #         loss_DNS, div_DNS, pdiff_DNS, predictions, UVP_DNS, UVP = find_scaling_step(latents, tminMaxUVP, imgA, list_rescaling_variables)
        #     elif (PROCEDURE=="A2"):
        #         loss_DNS, div_DNS, pdiff_DNS, predictions, UVP_DNS, UVP = find_scaling_step(latents, tminMaxUVP, imgA, list_rescaling_variables)
        #     elif (PROCEDURE=="B1"):
        #         loss_DNS, div_DNS, pdiff_DNS, predictions, UVP_DNS, UVP = find_scaling_step(latents, tminMaxUVP, imgA, list_rescaling_variables)

        #     if (itDNS%100 == 0):
        #         lr = lr_schedule_LES(itDNS)
        #         print("Search scaling iterations:  it {0:3d}  loss_DNS {1:3e}  div_DNS {2:3e}  pdiff_DNS {3:3e}  lr {4:3e} ".format(itDNS, loss_DNS, div_DNS, pdiff_DNS, lr))
        #         # U_LES = UVP[0, 0, :, :].numpy()
        #         # V_LES = UVP[0, 1, :, :].numpy()
        #         # P_LES = UVP[0, 2, :, :].numpy()
        #         # W_LES = find_vorticity(U_LES, V_LES)
        #         # print_fields(U_LES, V_LES, P_LES, W_LES, N_LES, filename="plots/plots_LES_fromGAN_scaling.png")
        #         # print_fields(U, V, P, W,                 N_LES, filename="plots/plots_LES.png")

        #     itDNS = itDNS+1

        # lr = lr_schedule(itDNS)
        # print("Search scaling iterations:  it {0:3d}  loss_DNS {1:3e}  div_DNS {2:3e}  pdiff_DNS {3:3e}  lr {4:3e} ".format(itDNS, loss_DNS, div_DNS, pdiff_DNS, lr))

        # find new DNS fields from GAN
        U_DNS = UVP_DNS.numpy()[0,0,:,:]
        V_DNS = UVP_DNS.numpy()[0,1,:,:]
        P_DNS = UVP_DNS.numpy()[0,2,:,:]

        # rescale LES field
        U = (U+one)*(maxU-minU)/two + minU
        V = (V+one)*(maxV-minV)/two + minV
        P = (P+one)*(maxP-minP)/two + minP



        # #---------------------------- retrain GAN
        #  create list of retrainable variables
        # synthe
        # for variable in synthesis.variables():

        # if PROCEDURE=="B1":

        #     # Create noise for sample images
        #     if (firstRetrain):
        #         tf.random.set_seed(SEED)        
        #         input_latent = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
        #         inputVariances = tf.constant(1.0, shape=(1, G_LAYERS), dtype=DTYPE)
        #         lr = LR
        #         mtr = np.zeros([5], dtype=DTYPE)
        #     firstRetrain = False

        #     # reaload checkpoint
        #     checkpoint.restore(managerCheckpoint.latest_checkpoint)

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
        while (resC>tollC and itC<maxItDNS):
            dd = Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0)
            C = solver_TDMAcyclic(-As, Ap, -An, dd, N_LES)
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
    if (it==maxItDNS):
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
        div = rho*A*nc.sum(nc.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
        div = div*iNN_LES
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
        LES_cv[tstep,1] = U[N_LES//2, N_LES//2]
        LES_cv[tstep,2] = V[N_LES//2, N_LES//2]
        LES_cv[tstep,3] = P[N_LES//2, N_LES//2]

        if (PROCEDURE=="DNS"):
            LES_cv_fromDNS[tstep,0] = totTime
            LES_cv_fromDNS[tstep,1] = newU[N_LES//2, N_LES//2]
            LES_cv_fromDNS[tstep,2] = newV[N_LES//2, N_LES//2]
            LES_cv_fromDNS[tstep,3] = newP[N_LES//2, N_LES//2]
        else:
            LES_cv_fromDNS[tstep,0] = totTime
            LES_cv_fromDNS[tstep,1] = U[N_LES//2, N_LES//2]
            LES_cv_fromDNS[tstep,2] = V[N_LES//2, N_LES//2]
            LES_cv_fromDNS[tstep,3] = P[N_LES//2, N_LES//2]


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
                    print_fields(U,     V,     P,     W,     N_LES,"plots/plots_LES_"         + str(te[s]) + "te.png")

                    #print spectrum
                    plot_spectrum_2d_3v(U_DNS, V_DNS, L, "energy/energy_DNS_fromGAN_" + str(te[s]) + "te.txt")
                    plot_spectrum_2d_3v(U, V, L,         "energy/energy_LES_"         + str(te[s]) + "te.txt")
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
                print_fields(U,     V,     P,     W,     N_LES, "plots/plots_LES_"         + tail + ".png")

            # print spectrum
            if (tstep%print_spe == 0):
                plot_spectrum_2d_3v(U_DNS, V_DNS, L, "energy/energy_spectrum_DNS_fromGAN_" + tail + ".txt")
                plot_spectrum_2d_3v(U,     V,     L, "energy/energy_spectrum_LES_"         + tail + ".txt")


# end of the simulation


# plot, save, find spectrum fields
if (len(te)==0):
    tail = "it{0:d}".format(tstep)

    # save images
    W_DNS = find_vorticity(U_DNS, V_DNS)
    W     = find_vorticity(U, V)
    print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N,    "plots/plots_DNS_fromGAN_" + tail + ".png")
    print_fields(U,     V,     P,     W,     N_LES, "plots/plots_LES_"         + tail + ".png")

    # write checkpoint
    W = find_vorticity(U, V)
    save_fields(totTime, U, V, P, C, B, W, "fields/fields_" + tail + ".npz")

    # print spectrum
    plot_spectrum_2d_3v(U_DNS, V_DNS, L, "energy/energy_spectrum_DNS_fromGAN_" + tail + ".txt")
    plot_spectrum_2d_3v(U,     V,     L, "energy/energy_spectrum_LES_"         + tail + ".txt")

# save center values
filename = "DNS_fromGAN_center_values.txt"
np.savetxt(filename, np.c_[DNS_cv[0:tstep+1,0], DNS_cv[0:tstep+1,1], DNS_cv[0:tstep+1,2], DNS_cv[0:tstep+1,3]], fmt='%1.4e')

filename = "LES_center_values.txt"
np.savetxt(filename, np.c_[LES_cv[0:tstep+1,0], LES_cv[0:tstep+1,1], LES_cv[0:tstep+1,2], LES_cv[0:tstep+1,3]], fmt='%1.4e')

filename = "LES_fromGAN_center_values.txt"
np.savetxt(filename, np.c_[LES_cv_fromDNS[0:tstep+1,0], LES_cv_fromDNS[0:tstep+1,1], LES_cv_fromDNS[0:tstep+1,2], LES_cv_fromDNS[0:tstep+1,3]], fmt='%1.4e')


print("Simulation successfully completed!")
