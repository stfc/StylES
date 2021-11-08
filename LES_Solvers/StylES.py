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

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from train import *
from PIL import Image


# start timing
tstart = time.time()



#---------------------------- define arrays
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
U_diff = np.zeros([OUTPUT_DIM, OUTPUT_DIM], dtype=DTYPE)
V_diff = np.zeros([OUTPUT_DIM, OUTPUT_DIM], dtype=DTYPE)



#---------------------------- initialize fields

# clean up and declarations
os.system("rm Energy_spectrum.png")
#os.system("rm restart.npz")
os.system("rm Energy_spectrum_it*")
os.system("rm fields_it*")
os.system("rm plots_it*")
os.system("rm uvw_it*")

res      = 2**(RES_LOG2-1)
ires2    = one/(2*res*res)  #2 because we sum U and V residuals
iOUTDIM22 = one/(2*OUTPUT_DIM*OUTPUT_DIM)  #2 because we sum U and V residuals  
DiffCoef  = np.full([res, res], Dc)
NL_DNS    = np.zeros([1, NUM_CHANNELS, OUTPUT_DIM, OUTPUT_DIM])
NL        = np.zeros([1, NUM_CHANNELS, OUTPUT_DIM, OUTPUT_DIM])

tf.random.set_seed(1)


# loading StyleGAN checkpoint and filter
checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))
mapping_ave.trainable = False
synthesis_ave.trainable = False


# create variable synthesis model
latents      = tf.keras.Input(shape=[G_LAYERS, LATENT_SIZE])
wlatents     = layer_wlatent(latents)
nlatents     = wlatents(latents) 
outputs      = synthesis_ave(nlatents, training=False)
wl_synthesis = tf.keras.Model(latents, outputs)

# latents       = tf.keras.Input(shape=[G_LAYERS-2, LATENT_SIZE])
# wlatents      = layer_wlatent(latents)
# nlatents      = wlatents(latents) 
# latents_const = tf.ones(shape=[2, LATENT_SIZE])
# flatents      = tf.concat(nlatents, latents_const)
# outputs       = synthesis_ave(flatents, training=False)
# wl_synthesis  = tf.keras.Model(latents, outputs)

# define learnin rate schedule and optimizer
if (lrDNS_POLICY=="EXPONENTIAL"):
    lr_schedule  = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lrDNS,
        decay_steps=lrDNS_STEP,
        decay_rate=lrDNS_RATE,
        staircase=lrDNS_EXP_ST)

elif (lrDNS_POLICY=="PIECEWISE"):
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lrDNS_BOUNDS, lrDNS_VALUES)

opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)

# log file for TensorBoard
dir_train_log        = 'logs/DNS_solver/'
train_summary_writer = tf.summary.create_file_writer(dir_train_log)


# initial flow
if (RESTART):

    # find DNS and LES fields from given field
    U_DNS, V_DNS, P_DNS, C_DNS, B_DNS, totTime = load_fields()
    W_DNS = find_vorticity(U, V)
    print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, name="DNS_org.png")

    itDNS        = 0
    resDNS       = large
    input_random = tf.random.uniform([1, LATENT_SIZE])
    dlatents     = mapping_ave(input_random, training=False)
    while (resDNS>tollDNS and itDNS<maxItDNS):
        with tf.GradientTape() as tape_DNS:
            predictions   = wl_synthesis(dlatents, training=False)
            UVW_DNS       = predictions[RES_LOG2-2]
            resDNS        =          tf.reduce_mean(tf.math.squared_difference(UVW_DNS[0,0,:,:], U_DNS))
            resDNS        = resDNS + tf.reduce_mean(tf.math.squared_difference(UVW_DNS[0,1,:,:], V_DNS))
            resDNS        = resDNS*iOUTDIM22
            gradients_DNS = tape_DNS.gradient(resDNS, wl_synthesis.trainable_variables)
            opt.apply_gradients(zip(gradients_DNS, wl_synthesis.trainable_variables))

        if (itDNS%100 == 0):
            lr = lr_schedule(itDNS)
            print("DNS iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))
            U_DNS_t = UVW_DNS[0, 0, :, :].numpy()
            V_DNS_t = UVW_DNS[0, 1, :, :].numpy()
            W_DNS_t = UVW_DNS[0, 2, :, :].numpy()
            print_fields(U_DNS_t, V_DNS_t, P_DNS, W_DNS_t, N, name="DNSfromDNS.png")

        with train_summary_writer.as_default():
            tf.summary.scalar("residuals", resDNS, step=itDNS)
            tf.summary.scalar("lr", lr, step=itDNS)

        itDNS = itDNS+1

else:

    # find DNS and LES fields from random input 
    totTime = zero
    P_DNS = nc.zeros([N,N], dtype=DTYPE)   # pressure
    B_DNS = nc.zeros([N,N], dtype=DTYPE)   # body force
    C_DNS = nc.zeros([N,N], dtype=DTYPE)   # passive scalar

    input_random = tf.random.uniform([1, LATENT_SIZE])
    dlatents     = mapping_ave(input_random, training=False)
    predictions  = synthesis_ave(dlatents, training=False)
    UVW_DNS      = predictions[RES_LOG2-2]


# save difference between initial DNS and provided DNS fields
U_diff[:,:] = U_DNS[:,:] - UVW_DNS[0, 0, :, :].numpy()
V_diff[:,:] = V_DNS[:,:] - UVW_DNS[0, 1, :, :].numpy()
W_diff[:,:] = W_DNS[:,:] - UVW_DNS[0, 2, :, :].numpy()

print_fields(U_diff, V_diff, P_DNS, W_diff, N, name="diff_DNS.png")


# find DNS field
U_DNS = UVW_DNS[0, 0, :, :].numpy()
V_DNS = UVW_DNS[0, 1, :, :].numpy()

n_DNS = np.zeros([1, 3, OUTPUT_DIM, OUTPUT_DIM])
n_DNS[0,0,:,:] = U_DNS[:,:]
n_DNS[0,1,:,:] = V_DNS[:,:]
n_DNS[0,2,:,:] = P_DNS[:,:]

UVW_DNS = tf.convert_to_tensor(n_DNS)


# find LES field
UVW     = filter(UVW_DNS, training=False)

U = UVW[0, 0, :, :].numpy()
V = UVW[0, 1, :, :].numpy()
P = UVW[0, 2, :, :].numpy()
W = find_vorticity(U, V)

# print fields
print_fields(U_DNS, V_DNS, P_DNS, W_DNS, N, name="DNS.png")
print_fields(U, V, P, W, NLES, name="LES.png")



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
plot_spectrum(U, V, L, tstep)

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
        # x-direction
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
            if ((itM+1)%101 == 0):
                print("x-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
            itM = itM+1


        # y-direction
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
            if ((itM+1)%101 == 0):
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
            if ((itP+1)%101 == 0):
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
        if ((it+1)%101 == 0):
            print("SIMPLE iterations:  it {0:3d}  residuals {1:3e}".format(it, res_cpu))

        it = it+1



    #---------------------------- find DNS field
    itDNS  = 0
    resDNS = large
    while (resDNS>tollDNS and itDNS<maxItDNS):
        with tf.GradientTape() as tape_DNS:
            predictions = wl_synthesis(dlatents, training=True)
            UVW    = predictions[RES_LOG2-3]
            resDNS =          tf.reduce_mean(tf.math.squared_difference(UVW[0,0,:,:], U))
            resDNS = resDNS + tf.reduce_mean(tf.math.squared_difference(UVW[0,1,:,:], V))
            resDNS = resDNS*ires2
            gradients_DNS = tape_DNS.gradient(resDNS, wl_synthesis.trainable_variables)
            opt.apply_gradients(zip(gradients_DNS, wl_synthesis.trainable_variables))

        if ((itDNS+1)%101 == 0):
            lr = lr_schedule(itDNS)
            print("DNS iterations:  it {0:3d}  residuals {1:3e}  lr {2:3e} ".format(itDNS, resDNS, lr))

        itDNS = itDNS+1

    # save new DNS field
    U_DNS = UVW_DNS[0, 0, :, :].numpy()
    V_DNS = UVW_DNS[0, 1, :, :].numpy()




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
            if ((itC+1)%101 == 0):
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
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, tstep, N, name="DNS")
                print_fields(U, V, P, C, tstep, NLES, name="LES")
                plot_spectrum(U_DNS, V_DNS, L, tstep)

            if (totTime<0.027722944+hf*delt and totTime>0.027722944-hf*delt):
                W_DNS = find_vorticity(U_DNS, V_DNS)
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, tstep, N, name="DNS")
                print_fields(U, V, P, C, tstep, NLES, name="LES")
                plot_spectrum(U_DNS, V_DNS, L, tstep)

            if (totTime<0.112046897+hf*delt and totTime>0.112046897-hf*delt):
                W_DNS = find_vorticity(U_DNS, V_DNS)
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, tstep, N, name="DNS")
                print_fields(U, V, P, C, tstep, NLES, name="LES")
                plot_spectrum(U_DNS, V_DNS, L, tstep)
        else:
            # save images
            if (tstep%print_img == 0):
                W_DNS = find_vorticity(U_DNS, V_DNS)
                print_fields(U_DNS, V_DNS, P_DNS, W_DNS, tstep, N, name="DNS")
                print_fields(U, V, P, C, tstep, NLES, name="LES")

            # write checkpoint
            if (tstep%print_ckp == 0):
                save_fields(totTime, tstep, U, V, P, C, B)

            # print spectrum
            if (tstep%print_spe == 0):
                plot_spectrum(U_DNS, V_DNS, L, tstep)




#---------------------------- end of the simulation

# save images
W_DNS = find_vorticity(U_DNS, V_DNS)
print_fields(U_DNS, V_DNS, P_DNS, W_DNS, tstep, N, name="DNS")
print_fields(U, V, P, C, tstep, NLES, name="LES")

# write checkpoint
save_fields(totTime, tstep, U, V, P, C, B)

# print spectrum
plot_spectrum(U_DNS, V_DNS, L, tstep)

# save center values
filename = "DNSfromLES_center_values.txt"
np.savetxt(filename, np.c_[DNS_cv[0:tstep+1,0], DNS_cv[0:tstep+1,1], DNS_cv[0:tstep+1,2], DNS_cv[0:tstep+1,3]], fmt='%1.4e')

filename = "LES_center_values.txt"
np.savetxt(filename, np.c_[LES_cv[0:tstep+1,0], LES_cv[0:tstep+1,1], LES_cv[0:tstep+1,2], LES_cv[0:tstep+1,3]], fmt='%1.4e')

print("Simulation succesfully completed!")
