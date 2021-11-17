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

sys.path.insert(0, '../')
sys.path.insert(0, './testcases/HIT_2D/')

DTYPE_LES = DTYPE

os.chdir('../')
from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *
os.chdir('./LES_Solvers/')

DTYPE = DTYPE_LES  # this is only because the StyleGAN is trained with float32 usually



#---------------------------- local variables
USE_FILTER = True
USE_DLATENTS = True
NLES = int(N/2)

Uo = nc.zeros([NLES,NLES], dtype=DTYPE)   # old x-velocity
Vo = nc.zeros([NLES,NLES], dtype=DTYPE)   # old y-velocity
Po = nc.zeros([NLES,NLES], dtype=DTYPE)   # old pressure field
Co = nc.zeros([NLES,NLES], dtype=DTYPE)   # old passive scalar
pc = nc.zeros([NLES,NLES], dtype=DTYPE)   # pressure correction
Z  = nc.zeros([NLES,NLES], dtype=DTYPE)   # zero array
C  = np.zeros([NLES,NLES], dtype=DTYPE)   # scalar
B  = np.zeros([NLES,NLES], dtype=DTYPE)   # body force
P  = np.zeros([NLES,NLES], dtype=DTYPE)   # body force

DNS_cv = np.zeros([totSteps+1, 4])
LES_cv = np.zeros([totSteps+1, 4])
U_diff = np.zeros([N, N], dtype=DTYPE)
V_diff = np.zeros([N, N], dtype=DTYPE)
W_diff = np.zeros([N, N], dtype=DTYPE)






#---------------------------- clean up and prepare run
# clean up and declarations
#os.system("rm restart.npz")
os.system("rm DNSfromLES_center_values.txt")
os.system("rm LES_center_values.txt")
os.system("rm Plots*")
os.system("rm Fields*")
os.system("rm Energy_spectrum*")

os.system("rm -rf plots")
os.system("rm -rf fields")
os.system("rm -rf uvw")
os.system("rm -rf energy")
os.system("rm -rf logs")

os.system("mkdir plots")
os.system("mkdir fields")
os.system("mkdir uvw")
os.system("mkdir energy")

res      = 2**(RES_LOG2-1)
ires2    = one/(2*res*res)  #2 because we sum U and V residuals
DiffCoef = np.full([res, res], Dc)
NL_DNS   = np.zeros([1, NUM_CHANNELS, N, N])
NL       = np.zeros([1, NUM_CHANNELS, NLES, NLES])

dir_train_log        = 'logs/DNS_solver/'
train_summary_writer = tf.summary.create_file_writer(dir_train_log)

tf.random.set_seed(1)






#----------------------------  loading StyleGAN checkpoint
checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))
mapping_ave.trainable = False
synthesis_ave.trainable = False






#----------------------------  create variable synthesis model
if (USE_DLATENTS):
    latents      = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
    wlatents     = layer_wlatent(latents)
    dlatents     = wlatents(latents) 
    outputs      = synthesis_ave(dlatents, training=False)
    wl_synthesis = tf.keras.Model(latents, outputs)
else:
    latents      = tf.keras.Input(shape=[LATENT_SIZE])
    wlatents     = layer_wlatent(latents)
    nlatents     = wlatents(latents) 
    dlatents     = mapping_ave(nlatents, training=False)
    outputs      = synthesis_ave(dlatents, training=False)
    wl_synthesis = tf.keras.Model(latents, outputs)






#----------------------------   define learning rate schedule, optimizer and checkpoint synthesis
if (lrDNS_POLICY=="EXPONENTIAL"):
    lr_schedule  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lrDNS,
        decay_steps=lrDNS_STEP,
        decay_rate=lrDNS_RATE,
        staircase=lrDNS_EXP_ST)

elif (lrDNS_POLICY=="PIECEWISE"):
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lrDNS_BOUNDS, lrDNS_VALUES)

opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)

wl_checkpoint = tf.train.Checkpoint(wl_synthesis=synthesis,
                                    opt=opt)






#---------------------------- local functions
@tf.function
def step_latent_DNS(U, V, latents):
    with tf.GradientTape() as tape_DNS:
        predictions   = wl_synthesis(latents)
        UVW_DNS       = predictions[RES_LOG2-2]*uRef
        resDNS        =          tf.reduce_mean(tf.math.squared_difference(UVW_DNS[0,0,:,:], U))
        resDNS        = resDNS + tf.reduce_mean(tf.math.squared_difference(UVW_DNS[0,1,:,:], V))
        gradients_DNS = tape_DNS.gradient(resDNS, wl_synthesis.trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, wl_synthesis.trainable_variables))
        resDNS        = resDNS*ires2

        return resDNS, predictions, UVW_DNS


@tf.function
def step_latent_LES(U, V, latents):
    with tf.GradientTape() as tape_DNS:
        predictions   = wl_synthesis(latents)
        if (USE_FILTER):
            UVW_DNS = predictions[RES_LOG2-2]*uRef
            UVW = filter(UVW_DNS, training=False)
        else:
            UVW = predictions[RES_LOG2-3]*uRef
        resDNS        =          tf.reduce_mean(tf.math.squared_difference(UVW[0,0,:,:], U))
        resDNS        = resDNS + tf.reduce_mean(tf.math.squared_difference(UVW[0,1,:,:], V))
        gradients_DNS = tape_DNS.gradient(resDNS, wl_synthesis.trainable_variables)
        opt.apply_gradients(zip(gradients_DNS, wl_synthesis.trainable_variables))
        resDNS        = resDNS*ires2

        return resDNS, predictions, UVW






#---------------------------- initialize flow
tstart = time.time()

if (INIT_BC==0):

    # find DNS and LES fields from given field
    U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, totTime = load_fields()  #from restart.npz file
    W_DNS = find_vorticity(U_DNS, V_DNS)
    print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="Plots_DNS_org.png")

    itDNS   = 0
    resDNS  = large
    tU_DNS = tf.convert_to_tensor(U_DNS, dtype=np.float32)
    tV_DNS = tf.convert_to_tensor(V_DNS, dtype=np.float32)

    random_inputs = tf.random.uniform([1, LATENT_SIZE])
    if (USE_DLATENTS):
        latents = mapping_ave(random_inputs, training=False)
    else:
        latents = random_inputs

    while (resDNS>tollDNS and itDNS<maxItDNS):
        resDNS, predictions, UVW_DNS = step_latent_DNS(tU_DNS, tV_DNS, latents)

        lr = lr_schedule(itDNS)
        if (itDNS%1000 == 0):
            print("DNS iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))
            U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
            V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
            W_DNS_t = UVW_DNS[0, 2, :, :].numpy()
            print_fields(U_DNS_t, V_DNS_t, P_DNS, W_DNS_t, N, filename="Plots_DNS_fromGAN.png")

            with train_summary_writer.as_default():
                tf.summary.scalar("residuals", resDNS, step=itDNS)
                tf.summary.scalar("lr", lr, step=itDNS)

        itDNS = itDNS+1

    # save difference between initial DNS and generated DNS fields
    U_diff[:,:] = U_DNS[:,:] - UVW_DNS[0, 0, :, :].numpy()
    V_diff[:,:] = V_DNS[:,:] - UVW_DNS[0, 1, :, :].numpy()
    W_diff[:,:] = W_DNS[:,:] - UVW_DNS[0, 2, :, :].numpy()
    print_fields(U_diff, V_diff, P_DNS, W_diff, N, filename="Plots_diff_DNS.png")

    # set DNS fields equal to the generated ones 
    U_DNS = UVW_DNS[0, 0, :, :].numpy()
    V_DNS = UVW_DNS[0, 1, :, :].numpy()
    W_DNS = UVW_DNS[0, 2, :, :].numpy()

    # set LES field: you must first find the pressure field!
    # assemble fields U, V from generated DNS plus pressure from restart file
    n_DNS = np.concatenate([U_DNS, V_DNS, P_DNS], axis=2)
    UVW_DNS = tf.convert_to_tensor(n_DNS)

    # filter them
    UVW = filter(UVW_DNS, training=False)
    P = UVW[0, 2, :, :].numpy()  #Note as the pressure is always obtained from the filter

    if (USE_FILTER):
        U = UVW[0, 0, :, :].numpy()
        V = UVW[0, 1, :, :].numpy()
    else:
        U = predictions[RES_LOG2-3][0, 0, :, :].numpy()*uRef
        V = predictions[RES_LOG2-3][0, 1, :, :].numpy()*uRef

elif (INIT_BC==1):

    # find DNS and LES fields from random input 
    totTime = zero
    P_DNS = nc.zeros([N,N], dtype=DTYPE)   # pressure
    B_DNS = nc.zeros([N,N], dtype=DTYPE)   # body force
    C_DNS = nc.zeros([N,N], dtype=DTYPE)   # passive scalar

    random_inputs = tf.random.uniform([1, LATENT_SIZE])
    if (USE_DLATENTS):
        latents = mapping_ave(random_inputs, training=False)
    else:
        latents = random_inputs

    predictions = wl_synthesis(latents, training=False)
    UVW_DNS     = predictions[RES_LOG2-2]*uRef

    # find DNS field
    U_DNS = UVW_DNS[0, 0, :, :].numpy()
    V_DNS = UVW_DNS[0, 1, :, :].numpy()

    # set LES field: you must first find the pressure field!
    # assemble fields U, V from generated DNS plus pressure from restart file
    n_DNS = np.concatenate([UVW_DNS[:, 0:2, :, :].numpy(), P_DNS[np.newaxis,np.newaxis,:,:]], axis=1)
    UVW_DNS = tf.convert_to_tensor(n_DNS)

    # filter them
    UVW = filter(UVW_DNS, training=False)
    P = UVW[0, 2, :, :].numpy()

    if (USE_FILTER):
        U = UVW[0, 0, :, :].numpy()
        V = UVW[0, 1, :, :].numpy()
    else:
        U = predictions[RES_LOG2-3][0, 0, :, :].numpy()*uRef
        V = predictions[RES_LOG2-3][0, 1, :, :].numpy()*uRef

elif (INIT_BC==2):

    # load latest StyLES checkpoint
    wl_checkpoint.restore(tf.train.latest_checkpoint(CHKP_DIR))

    totTime = zero
    P_DNS = nc.zeros([N,N], dtype=DTYPE)   # pressure
    B_DNS = nc.zeros([N,N], dtype=DTYPE)   # body force
    C_DNS = nc.zeros([N,N], dtype=DTYPE)   # passive scalar

    random_inputs = tf.random.uniform([1, LATENT_SIZE])
    if (USE_DLATENTS):
        latents = mapping_ave(random_inputs, training=False)
    else:
        latents = random_inputs

    predictions = wl_synthesis(latents, training=False)
    UVW_DNS     = predictions[RES_LOG2-2]*uRef

    # set DNS field (P_DNS is zero)
    U_DNS = UVW_DNS[0, 0, :, :].numpy()
    V_DNS = UVW_DNS[0, 1, :, :].numpy()

    # set LES fields
    U, V, P, C, B, totTime = load_fields()






#----------------------------  print fields
if (TEST_CASE == "HIT_2D_L&D"):
    tail = "0te"
else:
    tail = "it0"


W_DNS = find_vorticity(U_DNS, V_DNS)
print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_" + tail + ".png")

W = find_vorticity(U, V)
print_fields(U, V, P, W, NLES, filename="plots/plots_LES_" + tail + ".png")






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
cdelt = CNum*dl/(sqrt(nc.max(U)*nc.max(U) + nc.max(V)*nc.max(V))+small)
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

# plot spectrum
plot_spectrum(U, V, L, filename="energy/energy_spectrum_it" + str(tstep) + ".txt")

# track center point velocities and pressure
DNS_cv[tstep,0] = totTime
DNS_cv[tstep,1] = U_DNS[N//2, N//2]
DNS_cv[tstep,2] = V_DNS[N//2, N//2]
DNS_cv[tstep,3] = P_DNS[N//2, N//2]

LES_cv[tstep,0] = totTime
LES_cv[tstep,1] = U[NLES//2, NLES//2]
LES_cv[tstep,2] = V[NLES//2, NLES//2]
LES_cv[tstep,3] = P[NLES//2, NLES//2]

# start loop
while (tstep<totSteps and totTime<finalTime):


    # save old values of U, V and P
    Uo[:,:] = U[:,:]
    Vo[:,:] = V[:,:]
    Po[:,:] = P[:,:]
    if (PASSIVE):
        Co[:,:] = C[:,:]

    # start outer loop on SIMPLE convergence
    it = 0
    res = large
    while (res>toll and it<maxIt):

        # find non linear  terms
        NL_DNS[0, 0, :, :] = U_DNS*U_DNS
        NL_DNS[0, 1, :, :] = U_DNS*V_DNS
        NL_DNS[0, 2, :, :] = V_DNS*V_DNS

        NL = filter(NL_DNS, training=False)
        UU = NL[0, 0, :, :].numpy()
        UV = NL[0, 1, :, :].numpy()
        VV = NL[0, 2, :, :].numpy()

        # RsgsUU = UU - U*Uo
        # RsgsUV = UV - U*Vo
        # RsgsVU = UV - V*Uo
        # RsgsVV = VV - V*Vo

        RsgsUU = UU
        RsgsUV = UV
        RsgsVU = UV
        RsgsVV = VV


        #---------------------------- solve momentum equations
        # # x-direction
        # Fw = A*rho*hf*(Uo            + cr(Uo, -1, 0))
        # Fe = A*rho*hf*(cr(Uo,  1, 0) + Uo           )
        # Fs = A*rho*hf*(Vo            + cr(Vo, -1, 0))
        # Fn = A*rho*hf*(cr(Vo,  0, 1) + cr(Vo, -1, 1))

        Fw = zero
        Fe = zero
        Fs = zero
        Fn = zero

        Aw = DiffCoef + hf*(nc.abs(Fw) + Fw)
        Ae = DiffCoef + hf*(nc.abs(Fe) - Fe)
        As = DiffCoef + hf*(nc.abs(Fs) + Fs)
        An = DiffCoef + hf*(nc.abs(Fn) - Fn)
        Ao = rho*A*dl/delt

        Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
        iApU = one/Ap
        sU  = Ao*Uo -(P - cr(P, -1, 0))*A + hf*(B + cr(B, -1, 0)) \
            - rho*A*(cr(RsgsUU, 1, 0) - cr(RsgsUU, -1, 0))        \
            - rho*A*(cr(RsgsUU, 1, 0) - cr(RsgsUV, -1, 0))

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


        # # y-direction
        # Fw = A*rho*hf*(Uo             + cr(Uo, 0, -1))
        # Fe = A*rho*hf*(cr(Uo,  1,  0) + cr(Uo, 1, -1))
        # Fs = A*rho*hf*(cr(Vo,  0, -1) + Vo           )
        # Fn = A*rho*hf*(Vo             + cr(Vo, 0,  1))

        Fw = zero
        Fe = zero
        Fs = zero
        Fn = zero

        Aw = DiffCoef + hf*(nc.abs(Fw) + Fw)
        Ae = DiffCoef + hf*(nc.abs(Fe) - Fe)
        As = DiffCoef + hf*(nc.abs(Fs) + Fs)
        An = DiffCoef + hf*(nc.abs(Fn) - Fn)
        Ao = rho*A*dl/delt

        Ap  = Ao + Aw + Ae + As + An
        iApV = one/Ap
        sV  = Ao*Vo -(P - cr(P, 0, -1))*A + hf*(B + cr(B, 0, -1))  \
            - rho*A*(cr(RsgsVU, 1, 0) - cr(RsgsVU, -1, 0))         \
            - rho*A*(cr(RsgsVV, 1, 0) - cr(RsgsVV, -1, 0))

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



    #---------------------------- find DNS field
    itDNS  = 0
    resDNS = large
    tU = tf.convert_to_tensor(U, dtype=np.float32)
    tV = tf.convert_to_tensor(V, dtype=np.float32)
    while (resDNS>tollDNS and itDNS<maxItDNS):
        resDNS, predictions, UVW = step_latent_LES(tU, tV, latents)

        if ((itDNS+1)%100 == 0):
            lr = lr_schedule(itDNS)
            print("DNS iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))
            U_LES = UVW[0, 0, :, :].numpy()
            V_LES = UVW[0, 1, :, :].numpy()
            W_LES = UVW[0, 2, :, :].numpy()
            print_fields(U_LES, V_LES, P, W_LES, NLES, filename="Plots_LES_fromGAN.png")
            print_fields(U, V, P, W,             NLES, filename="Plots_LES.png")

        itDNS = itDNS+1

    #lr = lr_schedule(itDNS)
    #print("DNS iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))



    # save new DNS field
    U_DNS = predictions[RES_LOG2-2].numpy()[0,0,:,:]*uRef
    V_DNS = predictions[RES_LOG2-2].numpy()[0,1,:,:]*uRef
    W_DNS = predictions[RES_LOG2-2].numpy()[0,2,:,:]*uRef



    #---------------------------- solve transport equation for passive scalar
    if (PASSIVE):

        # solve iteratively
        Fw = A*rho*cr(U, -1, 0)
        Fe = A*rho*U
        Fs = A*rho*cr(V, 0, -1)
        Fn = A*rho*V

        Aw = Dc + hf*(nc.abs(Fw) + Fw)
        Ae = Dc + hf*(nc.abs(Fe) - Fe)
        As = Dc + hf*(nc.abs(Fs) + Fs)
        An = Dc + hf*(nc.abs(Fn) - Fn)
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
        # find new delt based on Courant number
        cdelt = CNum*dl/(sqrt(nc.max(U)*nc.max(U) + nc.max(V)*nc.max(V))+small)
        delt = convert(cdelt)
        delt = min(delt, maxDelt)
        totTime = totTime + delt
        tstep = tstep+1
        its = it

        # check divergence
        div = rho*A*nc.sum(nc.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
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

        # track center values of DNS and LES fields
        DNS_cv[tstep,0] = totTime
        DNS_cv[tstep,1] = U_DNS[N//2, N//2]
        DNS_cv[tstep,2] = V_DNS[N//2, N//2]
        DNS_cv[tstep,3] = P_DNS[N//2, N//2]

        LES_cv[tstep,0] = totTime
        LES_cv[tstep,1] = U[NLES//2, NLES//2]
        LES_cv[tstep,2] = V[NLES//2, NLES//2]
        LES_cv[tstep,3] = P[NLES//2, NLES//2]

        # print fields and spectrum
        if (TEST_CASE == "HIT_2D_L&D"):
            if (totTime<0.010396104+hf*delt and totTime>0.010396104-hf*delt):
                W_DNS = find_vorticity(U_DNS, V_DNS)
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_9te.png")
                W = find_vorticity(U, V)
                print_fields(U, V, P, W, NLES, filename="plots/plots_LES_9te.png")
                plot_spectrum(U_DNS, V_DNS, L, filename="energy/energy_spectrum_9te.txt")

            if (totTime<0.027722944+hf*delt and totTime>0.027722944-hf*delt):
                W_DNS = find_vorticity(U_DNS, V_DNS)
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_24te.png")
                W = find_vorticity(U, V)
                print_fields(U, V, P, W, NLES, filename="plots/plots_LES_24te.png")
                plot_spectrum(U_DNS, V_DNS, L, filename="energy/energy_spectrum_24te.txt")

            if (totTime<0.112046897+hf*delt and totTime>0.112046897-hf*delt):
                W_DNS = find_vorticity(U_DNS, V_DNS)
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_97te.png")
                W = find_vorticity(U, V)
                print_fields(U, V, P, W, NLES, filename="plots/plots_LES_97te.png")
                plot_spectrum(U_DNS, V_DNS, L, filename="energy/energy_spectrum_97te.txt")

            if (totTime<0.152751599+hf*delt and totTime>0.152751599-hf*delt):
                W_DNS = find_vorticity(U_DNS, V_DNS)
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_134te.png")
                W = find_vorticity(U, V)
                print_fields(U, V, P, W, NLES, filename="plots/plots_LES_134te.png")
                plot_spectrum(U_DNS, V_DNS, L, filename="energy/energy_spectrum_134te.txt")

        else:

            # save images
            if (tstep%print_img == 0):
                #W_DNS = find_vorticity(U_DNS, V_DNS)
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_it" + str(tstep) + ".png")
                W = find_vorticity(U, V)
                print_fields(U, V, P, W, NLES, filename="plots/plots_LES_it" + str(tstep) + ".png")

            # write checkpoint
            if (tstep%print_ckp == 0):
                W = find_vorticity(U, V)
                save_fields(totTime, U, V, P, C, B, W, filename="fields/fields_it" + str(tstep) + ".npz")

            # print spectrum
            if (tstep%print_spe == 0):
                plot_spectrum(U_DNS, V_DNS, L, filename="energy/energy_spectrum_it" + str(tstep) + ".txt")




#---------------------------- end of the simulation

# save images
if (TEST_CASE != "HIT_2D_L&D"):
    print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, filename="plots/plots_DNS_it" + str(tstep) + ".png")
    W = find_vorticity(U, V)
    print_fields(U, V, P, C, NLES, filename="plots/plots_LES_it" + str(tstep) + ".png")

    # write checkpoint
    save_fields(totTime, U, V, P, C, B, W, filename="fields/fields_it" + str(tstep) + ".npz")

    # print spectrum
    plot_spectrum(U_DNS, V_DNS, L, filename="energy/energy_spectrum_it" + str(tstep) + ".txt")

    # save center values
    filename = "DNSfromLES_center_values.txt"
    np.savetxt(filename, np.c_[DNS_cv[0:tstep+1,0], DNS_cv[0:tstep+1,1], DNS_cv[0:tstep+1,2], DNS_cv[0:tstep+1,3]], fmt='%1.4e')

    filename = "LES_center_values.txt"
    np.savetxt(filename, np.c_[LES_cv[0:tstep+1,0], LES_cv[0:tstep+1,1], LES_cv[0:tstep+1,2], LES_cv[0:tstep+1,3]], fmt='%1.4e')

    # save checkpoint for wl_synthetis network
    wl_checkpoint.save(file_prefix = CHKP_PREFIX)


print("Simulation succesfully completed!")
