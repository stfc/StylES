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
U, V, P, C, B = init_flow()


save_fields(U, V, P, C, 0, dir)



# find face velocities first guess as forward difference (i.e. on side east and north)
Ue = hf*(cr(U, 1, 0) + U)
Vn = hf*(cr(V, 1, 0) + V)


#---------------------------- main time step loop
tstep    = 0
totTime  = zero
resM_cpu = zero
resP_cpu = zero
resC_cpu = zero
res_cpu  = zero
firstIt  = True
its      = 0

# check divergence
div = cp.sum(cp.abs((Ue - cr(Ue, -1, 0))*dXY + (Vn - cr(Vn, 0, -1))*dXY))/(Nx*Ny)
div_cpu = cp.asnumpy(div)  

# print values
tend = time()
if (tstep%print_res == 0):
    wtime = (tend-tstart)
    print("Wall time [s] {0:.1f}  steps {1:3d}  time {2:.2e}  delt {3:.2e}  resM {4:.2e}  "\
            "resP {5:.2e}  resC {6:.2e}  res {7:.2e}  its {8:3d}  div {9:.2e}"       \
    .format(wtime, tstep, totTime, delt, resM_cpu, resP_cpu, \
    resC_cpu, res_cpu, its, div_cpu))

while (tstep<totSteps):


    # save old values of U, V and P
    Uo = U
    Vo = V
    Po = P
    Co = C


    # start outer loop on SIMPLE convergence
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
               + hf*deltpX1*cr(iAp, 1, 0)*dXY  \
               + hf*deltpX2*iAp*dXY            \
               + hf*deltpX3*(cr(iAp, 1, 0) + iAp)*dXY

            Vn = hf*(cr(V, 0, 1) + V)              \
               + hf*deltpY1*cr(iAp, 0, 1)*dXY  \
               + hf*deltpY2*iAp*dXY            \
               + hf*deltpY3*(cr(iAp, 0, 1) + iAp)*dXY
    
        firstIt = False
               


        #---------------------------- solve momentum equations
        Fw = rhoRef*hf*cr(Ue, -1, 0)
        Fe = rhoRef*hf*Ue
        Fs = rhoRef*hf*cr(Vn, 0, -1)
        Fn = rhoRef*hf*Vn

        Aw = DX + hf*(cp.abs(Fw) + Fw)
        Ae = DX + hf*(cp.abs(Fe) - Fe)
        As = DY + hf*(cp.abs(Fs) + Fs)
        An = DY + hf*(cp.abs(Fn) - Fn)
        Ao = rA/delt

        Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
        iAp = one/Ap

        aa = -As
        bb = Ap
        cc = -An
        dd = Ao*Uo + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0)   \
            - hf*(cr(P, 1, 0) - cr(P, -1, 0))*dXY     \
            + hf*(cr(B, 1, 0) + cr(B, -1, 0))
        nU = solver_TDMAcyclic(aa, bb, cc, dd, Ny)

        resM = cp.sum(cp.abs(Ap*U - dd - As*cr(U, 0, -1) - An*cr(U, 0, 1)))

        aa = -As
        bb = Ap
        cc = -An
        dd = Ao*Vo + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0)   \
            - hf*(cr(P, 0, 1) - cr(P, 0, -1))*dXY                                   \
            + hf*(cr(B, 0, 1) + cr(B, 0, -1))
        nV = solver_TDMAcyclic(aa, bb, cc, dd, Ny)

        resM = resM + cp.sum(cp.abs(Ap*V - dd - As*cr(V, 0, -1) - An*cr(V, 0, 1)))
        resM = resM/(2*Nx*Ny)

        resM_cpu = cp.asnumpy(resM)
        # print("Momemtum iterations:  it {0:3d}  residuals {1:3e}".format(it, resM_cpu))

        U = nU
        V = nV



        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = large
        Aw = hf*rYY*(cr(iAp, -1,  0) + iAp)
        Ae = hf*rYY*(cr(iAp,  1,  0) + iAp)
        As = hf*rXX*(cr(iAp,  0, -1) + iAp)
        An = hf*rXX*(cr(iAp,  0,  1) + iAp)

        Ap = Aw+Ae+As+An
        So = -rY*(Ue-cr(Ue, -1, 0)) - rX*(Vn-cr(Vn, 0, -1))
        pc = Z

        aa = -As
        bb = Ap
        cc = -An
        dd = So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0)
        npc = solver_TDMAcyclic(aa, bb, cc, dd, Ny)

        resP = cp.sum(cp.abs(Ap*pc - dd - As*cr(pc, 0, -1) - An*cr(pc, 0, 1)))
        resP = resP/(Nx*Ny)

        resP_cpu = cp.asnumpy(resP)
        # print("Pressure correction:  it {0:3d}  residuals {1:3e}".format(it, resP_cpu))

        pc = npc




        #---------------------------- update values using under relaxation factors
        deltpX1 = cr(pc, -1, 0) - cr(pc, 1, 0)
        deltpX2 = pc            - cr(pc, 1, 0)

        deltpY1 = cr(pc, 0, -1) - cr(pc, 0, 1)
        deltpY2 = pc            - cr(pc, 0, 1)

        nU  = alphaUV*hf*dXY*iAp*deltpX1
        nV  = alphaUV*hf*dXY*iAp*deltpY1
        nPc = alphaP*pc

        res = cp.sum(cp.abs(So))
        res = res/(Nx*Ny)

        U  = U + nU
        V  = V + nV
        P  = P + nPc
        Ue = Ue + alphaUV*hf*dXY*(cr(iAp, 1, 0) + iAp)*deltpX2
        Vn = Vn + alphaUV*hf*dXY*(cr(iAp, 0, 1) + iAp)*deltpY2

        res_cpu = cp.asnumpy(res)
        # print("SIMPLE iterations:  it {0:3d}  residuals {1:3e}".format(it, res_cpu))




        #---------------------------- solve transport equation for passive scalar
        if (PASSIVE):

            # solve iteratively
            Fw = rhoRef*hf*cr(U, -1, 0)
            Fe = rhoRef*hf*U
            Fs = rhoRef*hf*cr(V, 0, -1)
            Fn = rhoRef*hf*V

            Aw = DX + hf*(cp.abs(Fw) + Fw)
            Ae = DX + hf*(cp.abs(Fe) - Fe)
            As = DY + hf*(cp.abs(Fs) + Fs)
            An = DY + hf*(cp.abs(Fn) - Fn)
            Ao = rA/delt

            Ap = Ao + (Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs))

            aa = -As
            bb = Ap
            cc = -An
            dd = Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0)
            nC = solver_TDMAcyclic(aa, bb, cc, dd, Ny)

            resC = cp.sum(cp.abs(Ap*C - dd - As*cr(C, 0, -1) - Ae*cr(C, 0, 1)))/(Nx*Ny)
            C = nC

            resC_cpu = cp.asnumpy(resC)
            # print("Passive scalar:  it {0:3d}  residuals {1:3e}".format(it, resC_cpu))

            # find integral of passive scalar
            totSca = cp.asnumpy(cp.sum(C))
            maxSca = cp.asnumpy(cp.max(C))
            # print("Tot scalar {0:.8e}  max scalar {1:3e}".format(totSca, maxSca))
        else:
            resC = zero

        it = it+1




    #---------------------------- print update and save fields
    if (it==maxIt):
        print("Attention: SIMPLE solver not converged!!!")
        exit()

    else:
        # find new delt based on Courant number
        cdelt = CNum*dXY/(sqrt(cp.max(U)*cp.max(U) + cp.max(V)*cp.max(V))+small)
        delt = cp.asnumpy(cdelt)
        delt = min(delt, maxDelt)
        totTime = totTime + delt
        tstep = tstep+1
        its = it

        # check divergence
        div = cp.sum(cp.abs((Ue - cr(Ue, -1, 0))*dXY + (Vn - cr(Vn, 0, -1))*dXY))/(Nx*Ny)
        div_cpu = cp.asnumpy(div)  

        # print values
        tend = time()
        if (tstep%print_res == 0):
            wtime = (tend-tstart)
            print("Wall time [s] {0:.1f}  steps {1:3d}  time {2:.2e}  delt {3:.2e}  resM {4:.2e}  "\
                "resP {5:.2e}  resC {6:.2e}  res {7:.2e}  its {8:3d}  div {9:.2e}"       \
            .format(wtime, tstep, totTime, delt, resM_cpu, resP_cpu, \
            resC_cpu, res_cpu, its, div_cpu))

        # save images
        if (tstep%print_img == 0):
            save_fields(U, V, P, C, tstep, dir)

