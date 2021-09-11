import numpy as np
import cupy as cp
import os

from time import time
from PIL import Image
from math import sqrt

from LES_parameters import *
from LES_BC import *
from LES_plot import *
from LES_lAlg import *


# start timing
tstart = time()



#---------------------------- define arrays
U_cpu = np.zeros([Nx,Ny], dtype=DTYPE)   # x-velocity
V_cpu = np.zeros([Nx,Ny], dtype=DTYPE)   # y-velocity 
P_cpu = np.zeros([Nx,Ny], dtype=DTYPE)   # pressure field
C_cpu = np.zeros([Nx,Ny], dtype=DTYPE)   # passive scalar
B_cpu = cp.zeros([Nx,Ny], dtype=DTYPE)   # body force

Uo = cp.zeros([Nx,Ny], dtype=DTYPE)   # old x-velocity
Vo = cp.zeros([Nx,Ny], dtype=DTYPE)   # old y-velocity
Po = cp.zeros([Nx,Ny], dtype=DTYPE)   # old pressure field
Co = cp.zeros([Nx,Ny], dtype=DTYPE)   # old passive scalar

iAp = cp.zeros([Nx,Ny], dtype=DTYPE)  # central coefficient
Ue  = cp.zeros([Nx,Ny], dtype=DTYPE)  # face x-velocities
Vn  = cp.zeros([Nx,Ny], dtype=DTYPE)  # face y-velocities  

pc  = cp.zeros([Nx,Ny], dtype=DTYPE)  # pressure correction

nU  = cp.zeros([Nx,Ny], dtype=DTYPE)
nV  = cp.zeros([Nx,Ny], dtype=DTYPE)
nPc = cp.zeros([Nx,Ny], dtype=DTYPE)
nC  = cp.zeros([Nx,Ny], dtype=DTYPE)
Z   = cp.zeros([Nx,Ny], dtype=DTYPE)




#---------------------------- set flow pressure, velocity fields and BCs
os.system("rm *fields.png")
os.system("rm Energy_spectrum.png")

# initial flow
init_flow(U_cpu, V_cpu, P_cpu, C_cpu)


# move to GPUs
U = cp.asarray(U_cpu)
V = cp.asarray(V_cpu)
P = cp.asarray(P_cpu)
C = cp.asarray(C_cpu)
B = cp.asarray(B_cpu)

save_fields(U, V, P, C, 0, dir)



# find face velocities first guess as forward difference (i.e. on side east and north)
Ue = hf*(cr(U, 1, 0) + U)
Vn = hf*(cr(V, 1, 0) + V)


#---------------------------- main time step loop
tstep   = 0
totTime = zero

# check divergence
div = cp.sum( (Ue - cr(Ue, -1, 0))*deltaY + (Vn - cr(Vn, 0, -1))*deltaY )
div_cpu = cp.asnumpy(div)


tend = time()
if (tstep%print_res == 0):
    print("Time [h] {0:.1f}   step {1:3d}   delt {2:3e}   iterations {3:3d}   residuals {4:3e}   div {5:3e}"
    .format((tend-tstart)/3600.0, tstep, delt, 0, zero, div_cpu))


firstIt = True 
while (tstep<totSteps):

    #---------------------------- save old values of U, V and P
    Uo = U
    Vo = V
    Po = P
    Co = C

    #---------------------------- outer loop on SIMPLE convergence
    it = 0
    res = large
    while (res>toll and it<maxIt):



        #---------------------------- find Rhie-Chow interpolation (PWIM)
        if (not firstIt):
            deltpX1 = hf*(cr(P, 2, 0) - P)    
            deltpX2 = hf*(cr(P, 1, 0) - cr(P, -1, 0))
            deltpX3 = (P - cr(P,  1, 0))

            deltpY1 = hf*(cr(P, 0, 2) - P)
            deltpY2 = hf*(cr(P, 0, 1) - cr(P, 0, -1))
            deltpY3 = (P - cr(P, 0,  1))

            Ue = hf*(cr(U, 1, 0) + U)             \
               + hf*deltpX1*cr(iAp, 1, 0)*deltaY  \
               + hf*deltpX2*iAp*deltaY            \
               + hf*deltpX3*(cr(iAp, 1, 0) + iAp)*deltaY

            Vn = hf*(cr(V, 0, 1) + V)              \
               + hf*deltpY1*cr(iAp, 0, 1)*deltaX  \
               + hf*deltpY2*iAp*deltaX            \
               + hf*deltpY3*(cr(iAp, 0, 1) + iAp)*deltaX
    
        firstIt = False
               


        #---------------------------- solve momentum equations
        itMom  = 0
        resMom = one

        Fw = rhoRef*hf*cr(Ue, -1, 0)
        Fe = rhoRef*hf*Ue
        Fs = rhoRef*hf*cr(Vn, 0, -1)
        Fn = rhoRef*hf*Vn

        Aw = DX + hf*(abs(Fw) + Fw)
        Ae = DX + hf*(abs(Fe) - Fe)
        As = DY + hf*(abs(Fs) + Fs)
        An = DY + hf*(abs(Fn) - Fn)
        Ao = rA/delt

        iAp = one/(Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs))

        while (resMom>tollMom and itMom<maxItMom):
            resMom = zero

            # x-direction
            rhs = Ao*Uo + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0) + As*cr(U, 0, -1) + An*cr(U, 0, 1)  \
                - hf*(cr(P, 1, 0) - cr(P, -1, 0))*deltaY                                       \
                + hf*(cr(B, 1, 0) + cr(B, -1, 0))
            nU = rhs*iAp

            resMom = cp.sum(abs(U - nU))

            # # y-direction
            rhs = Ao*Vo + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0) + As*cr(V, 0, -1) + An*cr(V, 0, 1)  \
                - hf*(cr(P, 0, 1) - cr(P, 0, -1))*deltaX                                   \
                + hf*(cr(B, 0, 1) + cr(B, 0, -1))
            nV = rhs*iAp

            resMom = resMom + cp.sum(abs(V-nV))/(2*Nx*Ny)

            itMom = itMom+1
            resMom_cpu = cp.asnumpy(resMom)
            #print("Momemtum iterations {0:3d}   residuals {1:3e}".format(itMom, resMom_cpu))

            U = nU
            V = nV



        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = large
        Aw = hf*rYY*(cr(iAp, -1,  0) + iAp)
        Ae = hf*rYY*(cr(iAp,  1,  0) + iAp)
        As = hf*rXX*(cr(iAp,  0, -1) + iAp)
        An = hf*rXX*(cr(iAp,  0,  1) + iAp)
        Ao = Aw+Ae+As+An
        So = -rY*(Ue-cr(Ue, -1, 0)) - rX*(Vn-cr(Vn, 0, -1))
        pc = Z
        while (resPc>tollPc and itPc<maxItPc):
            rhs = So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0) + As*cr(pc, 0, -1) + An*cr(pc, 0, 1)
            npc = rhs/Ao
            resPc = cp.sum(abs(pc-npc))/(Nx*Ny)

            if (itPc<maxItPc-1):
                resPc_cpu = cp.asnumpy(resPc)
                #print("Pressure correction iterations {0:3d}   residuals {1:3e}".format(itPc, resPc_cpu))
            else:
                # give warning if solution of the pressure correction is not achieved
                print("Attention: pressure correction solver not converged!!!")
                save_fields(U, V, pc, C, tstep, dir)
                exit()

            itPc = itPc+1
            pc = npc




        #---------------------------- update values using under relaxation factors
        res = zero
        deltpX1 = cr(pc, -1, 0) - cr(pc, 1, 0)
        deltpX2 = pc            - cr(pc, 1, 0)

        deltpY1 = cr(pc, 0, -1) - cr(pc, 0, 1)
        deltpY2 = pc            - cr(pc, 0, 1)

        nU  = alphaUV*hf*deltaY*iAp*deltpX1
        nV  = alphaUV*hf*deltaX*iAp*deltpY1
        nPc = alphaP*pc

        res = cp.sum(abs(nU) + abs(nV) + abs(nPc))/(Nx*Ny)

        U  = U + nU
        V  = V + nV
        P  = P + nPc
        Ue = Ue + alphaUV*hf*deltaY*(cr(iAp, 1, 0) + iAp)*deltpX2
        Vn = Vn + alphaUV*hf*deltaX*(cr(iAp, 0, 1) + iAp)*deltpY2

        it = it+1
        if (it%10 == 0):
            res_cpu = cp.asnumpy(res)
            print("SIMPLE iterations {0:3d}   residuals {1:3e}".format(it, res_cpu))




        #---------------------------- solve transport equation for passive scalar
        if (PASSIVE):

            # solve iteratively
            itC  = 0
            resC = one

            Fw = rhoRef*hf*cr(U, -1, 0)
            Fe = rhoRef*hf*U
            Fs = rhoRef*hf*cr(V, 0, -1)
            Fn = rhoRef*hf*V

            Aw = DX + hf*(abs(Fw) + Fw)
            Ae = DX + hf*(abs(Fe) - Fe)
            As = DY + hf*(abs(Fs) + Fs)
            An = DY + hf*(abs(Fn) - Fn)
            Ao = rA/delt

            while (resC>tollC and itC<maxItC):

                resC = zero
                rhs = Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0) + As*cr(C, 0, -1) + An*cr(C, 0, 1)
                nC = rhs/(Ao + (Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)))

                resC = cp.sum(nC - C)
                C = nC

                itC = itC+1
                resC_cpu = cp.asnumpy(resC)
                print("Iterations TDMA {0:3d}   residuals TDMA {1:3e}".format(itC, resC_cpu))

            # find integral of passive scalar
            totSca = cp.asnumpy(cp.sum(C))
            maxSca = cp.asnumpy(cp.max(C))
            print("Tot scalar {0:.8e}  max scalar {1:3e}".format(totSca, maxSca))




    #---------------------------- print update and save fields
    if (it==maxIt):
        print("Attention: SIMPLE solver not converged!!!")
        exit()

    else:
        # find new delt based on Courant number
        delt = CNum*deltaX/(sqrt(cp.max(U)*cp.max(U) + cp.max(V)*cp.max(V))+small)
        delt = min(delt, maxDelt)
        totTime = totTime + delt
        tstep = tstep+1

        # check divergence
        div = cp.sum( (Ue - cr(Ue, -1, 0))*deltaY + (Vn - cr(Vn, 0, -1))*deltaY )
        div_cpu = cp.asnumpy(div)  

        tend = time()
        if (tstep%print_res == 0):
            print("Time [h] {0:.1f}   step {1:3d}   delt {2:3e}   iterations {3:3d}   residuals {4:3e}   div {5:3e}"
            .format((tend-tstart)/3600., tstep, delt, it, res_cpu, div_cpu))

        if (tstep%print_img == 0):
            save_fields(U, V, P, C, tstep, dir)

