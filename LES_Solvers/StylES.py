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
Uo = nc.zeros([N,N], dtype=DTYPE)   # old x-velocity
Vo = nc.zeros([N,N], dtype=DTYPE)   # old y-velocity
Po = nc.zeros([N,N], dtype=DTYPE)   # old pressure field
Co = nc.zeros([N,N], dtype=DTYPE)   # old passive scalar
pc = nc.zeros([N,N], dtype=DTYPE)   # pressure correction
Z  = nc.zeros([N,N], dtype=DTYPE)   # zero array




#---------------------------- initialize fields

# clean up and declarations
os.system("rm fields*")
os.system("rm Energy_spectrum*")
os.system("rm uvp_*")

pow2     = 2**(RES_LOG2-1)
DiffCoef = np.full([pow2, pow2], Dc)
NL_DNS   = np.zeros([BATCH_SIZE, NUM_CHANNELS, OUTPUT_DIM, OUTPUT_DIM])
NL       = np.zeros([BATCH_SIZE, NUM_CHANNELS, OUTPUT_DIM, OUTPUT_DIM])


# loading StyleGAN checkpoint and filter
checkpoint.restore(tf.train.latest_checkpoint("../" + CHKP_DIR))
tf.random.set_seed(1)
input_random = tf.random.uniform([BATCH_SIZE, LATENT_SIZE])
dlatents     = mapping_ave(input_random, training=False)
predictions  = synthesis_ave(dlatents, training=False)
UVP_DNS      = predictions[RES_LOG2-2]
UVP          = filter(UVP_DNS, training=False)


# find DNS and LES fields
U_DNS = UVP_DNS[0, 0, :, :].numpy()
V_DNS = UVP_DNS[0, 1, :, :].numpy()
P_DNS = UVP_DNS[0, 2, :, :].numpy()

U = UVP[0, 0, :, :].numpy()
V = UVP[0, 1, :, :].numpy()
P = UVP[0, 2, :, :].numpy()
B = nc.zeros([N,N], dtype=DTYPE)   # body force
C = nc.zeros([N,N], dtype=DTYPE)   # passive scalar


# print fields
print_fields(U, V, P, C, 0, dir)




#---------------------------- main time step loop

# init variables
tstep    = 0
resM_cpu = zero
resP_cpu = zero
resC_cpu = zero
res_cpu  = zero
its      = 0
totTime  = zero


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


# start loop
while (tstep<totSteps and totTime<finalTime):


    # save old values of U, V and P
    Uo[:,:] = U[:,:]
    Vo[:,:] = V[:,:]
    Po[:,:] = P[:,:]
    if (PASSIVE):
        Co[:,:] = C[:,:]

    # find non linear  terms
    NL_DNS[0, 0, :, :] = U_DNS*U_DNS
    NL_DNS[0, 1, :, :] = U_DNS*V_DNS
    NL_DNS[0, 2, :, :] = V_DNS*V_DNS

    NL = filter(NL_DNS, training=False)
    UU = NL[0, 0, :, :].numpy()
    UV = NL[0, 1, :, :].numpy()
    VV = NL[0, 2, :, :].numpy()


    # start outer loop on SIMPLE convergence
    it = 0
    res = large
    while (res>toll and it<maxIt):


        #---------------------------- solve momentum equations
        # x-direction
        Aw = DiffCoef
        Ae = DiffCoef
        As = DiffCoef
        An = DiffCoef
        Ao = rho*A*dl/delt

        Ap = Ao + Aw + Ae + As + An
        iApU = one/Ap
        sU  = Ao*Uo -(P - cr(P, -1, 0))*A + hf*(B + cr(B, -1, 0)) \
            - rho*A*cr(UU, 1, 0) + rho*A*cr(UU, -1, 0)      \
            - rho*A*cr(UV, 1, 0) + rho*A*cr(UV, -1, 0)

        itM  = 0
        resM = large
        while (resM>tollM and itM<maxIt):

            dd = sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0)
            U = solver_TDMAcyclic(-As, Ap, -An, dd, N)
            U = (sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0) + As*cr(U, 0, -1) + An*cr(U, 0, 1))*iApU
            resM = nc.sum(nc.abs(Ap*U - sU - Aw*cr(U, -1, 0) - Ae*cr(U, 1, 0) - As*cr(U, 0, -1) - An*cr(U, 0, 1)))
            resM = resM*iNN
            resM_cpu = convert(resM)
            # print("x-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
            itM = itM+1


        # y-direction
        Aw = DiffCoef
        Ae = DiffCoef
        As = DiffCoef
        An = DiffCoef
        Ao = rho*A*dl/delt
        Ap  = Ao + Aw + Ae + As + An
        iApV = one/Ap
        sV  = Ao*Vo -(P - cr(P, 0, -1))*A + hf*(B + cr(B, 0, -1))  \
            - rho*A*cr(VV, 1, 0) + rho*A*cr(VV, -1, 0)       \
            - rho*A*cr(UV, 1, 0) + rho*A*cr(UV, -1, 0)

        itM  = 0
        resM = one
        while (resM>tollM and itM<maxIt):

            dd = sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0)
            V = solver_TDMAcyclic(-As, Ap, -An, dd, N)
            V = (sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0) + As*cr(V, 0, -1) + An*cr(V, 0, 1))*iApV
            resM = nc.sum(nc.abs(Ap*V - sV - Aw*cr(V, -1, 0) - Ae*cr(V, 1, 0) - As*cr(V, 0, -1) - An*cr(V, 0, 1)))

            resM = resM*iNN
            resM_cpu = convert(resM)
            # print("y-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(it, resM_cpu))
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
            pc = solver_TDMAcyclic(-As, Ap, -An, dd, N)
            pc = (So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0) + As*cr(pc, 0, -1) + An*cr(pc, 0, 1))*iApP

            resP = nc.sum(nc.abs(Ap*pc - So - Aw*cr(pc, -1, 0) - Ae*cr(pc, 1, 0) - As*cr(pc, 0, -1) - An*cr(pc, 0, 1)))
            resP = resP*iNN

            resP_cpu = convert(resP)
            # print("Pressure correction:  it {0:3d}  residuals {1:3e}".format(itP, resP_cpu))
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
        # print("SIMPLE iterations:  it {0:3d}  residuals {1:3e}".format(it, res_cpu))




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
                C = solver_TDMAcyclic(-As, Ap, -An, dd, N)
                C = (Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0) + As*cr(C, 0, -1) + An*cr(C, 0, 1))*iApC

                resC = nc.sum(nc.abs(Ap*C - Ao*Co - Aw*cr(C, -1, 0) - Ae*cr(C, 1, 0) - As*cr(C, 0, -1) - An*cr(C, 0, 1)))
                resC = resC*iNN
                resC_cpu = convert(resC)
                # print("Passive scalar:  it {0:3d}  residuals {1:3e}".format(itC, resC_cpu))
                itC = itC+1

            # find integral of passive scalar
            totSca = convert(nc.sum(C))
            maxSca = convert(nc.max(C))
            print("Tot scalar {0:.8e}  max scalar {1:3e}".format(totSca, maxSca))

        it = it+1




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


        if (TEST_CASE == "HIT_2D"):
            if (totTime<0.010396104+hf*delt and totTime>0.010396104-hf*delt):
                print_fields(U, V, P, C, tstep, dir)
                plot_spectrum(U, V, L, tstep)

            if (totTime<0.027722944+hf*delt and totTime>0.027722944-hf*delt):
                print_fields(U, V, P, C, tstep, dir)
                plot_spectrum(U, V, L, tstep)

            if (totTime<0.112046897+hf*delt and totTime>0.112046897-hf*delt):
                print_fields(U, V, P, C, tstep, dir)
                plot_spectrum(U, V, L, tstep)

        else:
    
            # save images
            if (tstep%print_img == 0):
                print_fields(U, V, P, C, tstep, dir)

            # write checkpoint
            if (tstep%print_ckp == 0):
                save_fields(totTime, tstep, U, V, P, C, B)


            # print spectrum
            if (tstep%print_spe == 0):
                plot_spectrum(U, V, L, tstep)


# end of the simulation

# save images
print_fields(U, V, P, C, tstep, dir)

# write checkpoint
save_fields(totTime, tstep, U, V, P, C, B)

# print spectrum
plot_spectrum(U, V, L, tstep)

print("Simulation succesfully completed!")
