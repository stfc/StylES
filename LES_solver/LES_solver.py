import numpy as np
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
U  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # x-velocity
V  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # y-velocity 
P  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # pressure field
C  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # passive scalar

Uo = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # old x-velocity
Vo = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # old y-velocity
Po = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # old pressure field
Co = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # old passive scalar

iAp = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # central coefficient
Ue  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # face x-velocities
Vn  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # face y-velocities  

pc  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # pressure correction
nPc = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # pressure correction at new iteration

B   = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # body force

aa = np.zeros([Ny], dtype=DTYPE)  # aw coefficient for TDMA
bb = np.zeros([Ny], dtype=DTYPE)  # ap coefficient for TDMA
cc = np.zeros([Ny], dtype=DTYPE)  # ae coefficient for TDMA
dd = np.zeros([Ny], dtype=DTYPE)  # SU coefficient for TDMA

ee = np.zeros([Ny], dtype=DTYPE)  # SV coefficient for TDMA
ff = np.zeros([Ny], dtype=DTYPE)  # SV coefficient for TDMA
gg = np.zeros([Ny], dtype=DTYPE)  # SV coefficient for TDMA
hh = np.zeros([Ny], dtype=DTYPE)  # SV coefficient for TDMA

Cn = np.zeros([Nx,Ny], dtype=DTYPE)  # solution of TDMA





#---------------------------- set flow pressure, velocity fields and BCs
os.system("rm *fields.png")
os.system("rm Energy_spectrum.png")

# initial flow
init_flow(U, V, P, C)

apply_BCs(U, V, P, C, pc, Ue, Vn)
save_fields(U, V, P, C, 0, dir)



# find face velocities first guess as forward difference (i.e. on side east and north)
for i in range(1,Nx+1):
    for j in range(1,Ny+1):
        Ue[i,j] = hf*(U[i+1,j] + U[i,j])
        Vn[i,j] = hf*(V[i,j+1] + V[i,j])

apply_BCs(U, V, P, C, pc, Ue, Vn)


#---------------------------- main time step loop
tstep   = 0
totTime = zero

# check divergence
div = zero
for i in range(1,Nx+1):
    for j in range(1,Ny+1):
        div = div + (Ue[i,j] - Ue[i-1,j])*deltaY + (Vn[i,j] - Vn[i-1,j])*deltaY     

tend = time()
if (tstep%print_res == 0):
    print("Time [h] {0:.1f}   step {1:3d}   delt {2:3e}   iterations {3:3d}   residuals {4:3e}   div {5:3e}"
    .format((tend-tstart)/3600.0, tstep, delt, 0, zero, div))


while (tstep<totSteps):

    #---------------------------- save old values of U, V and P
    for i in range(0,Nx+2):
        for j in range(0,Ny+2):
            Uo[i,j] = U[i,j]
            Vo[i,j] = V[i,j]
            Po[i,j] = P[i,j]
            Co[i,j] = C[i,j]

    #---------------------------- outer loop on SIMPLE convergence
    it = 0
    res = large
    while (res>toll and it<maxIt):



        #---------------------------- find Rhie-Chow interpolation (PWIM)
        if (tstep>0):
            for i in range(1,Nx+1):
                for j in range(1,Ny+1):
                    if (i==Nx):  # periodic
                        deltpX1 = hf*(P[2,j] - P[i,j])
                    else:
                        deltpX1 = hf*(P[i+2,j] - P[i,j])    
                    deltpX2 = hf*(P[i+1,j] - P[i-1,j])
                    deltpX3 = hf*(P[i  ,j] - P[i+1,j])

                    if (j==Ny):  # periodic
                        deltpY1 = hf*(P[i,2] - P[i,j])
                    else:
                        deltpY1 = hf*(P[i,j+2] - P[i,j])
                    deltpY2 = hf*(P[i,j+1] - P[i,j-1])
                    deltpY3 = hf*(P[i,j  ] - P[i,j+1])

                    Ue[i,j] = hf*(U[i+1,j] + U[i,j])       \
                             + hf*deltpX1*iAp[i+1,j]*deltaY  \
                             + hf*deltpX2*iAp[i  ,j]*deltaY    \
                             + hf*deltpX3*(iAp[i+1,j] + iAp[i,j])*deltaY

                    Vn[i,j] = hf*(V[i,j+1] + V[i,j])       \
                             + hf*deltpY1*iAp[i,j+1]*deltaX  \
                             + hf*deltpY2*iAp[i,j  ]*deltaX  \
                             + hf*deltpY3*(iAp[i,j+1] + iAp[i,j])*deltaX

                    apply_BCs(U, V, P, C, pc, Ue, Vn)
                    
               


        #---------------------------- solve momentum equations
        itMom  = 0
        resMom = one
        while (resMom>tollMom and itMom<maxItMom):
            resMom = zero
            for i in range(1,Nx+1):
                for j in range(1,Ny+1):

                    Fw = rhoRef*hf*Ue[i-1,j  ]
                    Fe = rhoRef*hf*Ue[i  ,j  ]
                    Fs = rhoRef*hf*Vn[i  ,j-1]
                    Fn = rhoRef*hf*Vn[i  ,j  ]

                    Aw = DX + max(Fw,    zero)
                    Ae = DX + max(zero, -Fe)
                    As = DY + max(Fs,    zero)
                    An = DY + max(zero, -Fn)
                    Ao = rA/delt

                    iAp[i,j] = one/Ao

                    # x-direction
                    rhsU = Ao*Uo[i,j] - (Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs))*U[i,j]  \
                         + Aw*U[i-1,j] + Ae*U[i+1,j] + As*U[i,j-1] + An*U[i,j+1]        \
                         - hf*(Po[i+1,j] - Po[i-1,j])*deltaY                            \
                         + hf*(B[i+1,j]+B[i-1,j])
                    resMom = resMom + abs(U[i,j] - rhsU/Ao)
                    U[i,j] = rhsU/Ao

                    # y-direction
                    rhsV = Ao*Vo[i,j] - (Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs))*V[i,j]  \
                        +  Aw*V[i-1,j] + Ae*V[i+1,j] + As*V[i,j-1] + An*V[i,j+1]        \
                        - hf*(Po[i,j+1] - Po[i,j-1])*deltaX                             \
                        + hf*(B[i,j+1]+B[i,j-1])
                    resMom = resMom + abs(V[i,j] - rhsV/Ao)
                    V[i,j] = rhsV/Ao

            itMom = itMom+1
            #print("Momemtum iterations {0:3d}   residuals {1:3e}".format(itMom, resMom))

        apply_BCs(U, V, P, C, pc, Ue, Vn)
        if (DEBUG):
            save_fields(U, V, P, C, tstep, dir)




        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = large
        while (resPc>tollPc and itPc<maxItPc):
            resPc = 0.e0
            for i in range(1,Nx+1):
                for j in range(1,Ny+1):
                    Aw = hf*rYY*(iAp[i-1,j  ] + iAp[i,j])
                    Ae = hf*rYY*(iAp[i+1,j  ] + iAp[i,j])
                    As = hf*rXX*(iAp[i  ,j-1] + iAp[i,j])
                    An = hf*rXX*(iAp[i  ,j+1] + iAp[i,j])
                    Ao = Aw+Ae+As+An
                    So = Aw*pc[i-1,j] + Ae*pc[i+1,j] + As*pc[i,j-1] + An*pc[i,j+1]  \
                       -rY*(Ue[i,j]-Ue[i-1,j]) - rX*(Vn[i,j]-Vn[i,j-1])

                    resPc = resPc + abs(pc[i,j] - So/Ao)
                    pc[i,j] = So/Ao


            if (itPc<maxItPc-1):
                #print("Pressure correction iterations {0:3d}   residuals {1:3e}".format(itPc, resPc))

                apply_BCs(U, V, P, C, pc, Ue, Vn)
                if (DEBUG):
                    save_fields(U, V, P, C, tstep, dir)

            else:
                # give warning if solution of the pressure correction is not achieved
                print("Attention: pressure correction solver not converged!!!")
                save_fields(U, V, pc, tstep, dir)
                exit()

            itPc = itPc+1




        #---------------------------- update values using under relaxation factors
        res = zero
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):

                deltpX1 = pc[i-1,j] - pc[i+1,j]
                deltpX2 = pc[i  ,j] - pc[i+1,j]

                deltpY1 = pc[i,j-1] - pc[i,j+1]              
                deltpY2 = pc[i,j  ] - pc[i,j+1]

                prevP = P[i,j]
                prevU = U[i,j] 
                prevV = V[i,j]

                P[i,j] = P[i,j] + alphaP*pc[i,j]
                U[i,j] = U[i,j] + alphaUV*hf*deltaY*iAp[i,j]*deltpX1
                V[i,j] = V[i,j] + alphaUV*hf*deltaX*iAp[i,j]*deltpY1
                Ue[i,j] = Ue[i,j] + alphaUV*hf*deltaY*(iAp[i+1,j  ] + iAp[i,j])*deltpX2
                Vn[i,j] = Vn[i,j] + alphaUV*hf*deltaX*(iAp[i  ,j+1] + iAp[i,j])*deltpY2

                res = res + abs(prevP - P[i,j]) + abs(prevU - U[i,j]) + abs(prevV - V[i,j])

        apply_BCs(U, V, P, C, pc, Ue, Vn)
        if (DEBUG):
            save_fields(U, V, P, C, tstep, dir)

        it = it+1
        print("SIMPLE iterations {0:3d}   residuals {1:3e}".format(it, res))




        #---------------------------- solve transport equation for passive scalar
        if (PASSIVE):
            itC  = 0
            resC = one
            while (resC>tollC and itC<maxItC):

                resC = zero
                for i in range(1, Nx+1):
                    for j in range(1, Ny+1):
        
                        Fw = rhoRef*hf*U[i-1,j  ]
                        Fe = rhoRef*hf*U[i  ,j  ]
                        Fs = rhoRef*hf*V[i  ,j-1]
                        Fn = rhoRef*hf*V[i  ,j  ]

                        Aw = DX + max(Fw,    zero)
                        Ae = DX + max(zero, -Fe)
                        As = DY + max(Fs,    zero)
                        An = DY + max(zero, -Fn)
                        Ao = rA/delt

                        aa[j-1] = -As
                        bb[j-1] = Ao + (Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs))
                        cc[j-1] = -An
                        dd[j-1] = Ao*Co[i,j] + Aw*C[i-1,j] + Ae*C[i+1,j]

                    Cn[i-1,:] = solver_TDMAcyclic(aa, bb, cc, dd, Ny)

                    for j in range(1,Ny+1):
                        resC = resC + abs(Cn[i-1,j-1] - C[i,j])
                        C[i,j] = Cn[i-1,j-1]

                apply_BCs(U, V, P, C, pc, Ue, Vn)

                itC = itC+1
                #print("Iterations TDMA {0:3d}   residuals TDMA {1:3e}".format(itC, resC))

                if (DEBUG):
                    save_fields(U, V, P, C, tstep, dir)



            # find integral of passive scalar
            totSca=zero
            for i in range(1,Nx+1):
                for j in range(1,Ny+1):
                    totSca = totSca + C[i,j]
            print("Tot scalar {0:.8e}  max scalar {1:3e}".format(totSca, np.max(C)))




    #---------------------------- print update and save fields
    if (it==maxIt):
        print("Attention: SIMPLE solver not converged!!!")
        exit()

    else:
        # find new delt based on Courant number
        delt = CNum*deltaX/(sqrt(np.max(U)*np.max(U) + np.max(V)*np.max(V))+small)
        delt = min(delt, maxDelt)
        totTime = totTime + delt
        tstep = tstep+1

        # check divergence
        div = zero
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                div = div + (Ue[i,j] - Ue[i-1,j])*deltaY + (Vn[i,j] - Vn[i-1,j])*deltaY     

        tend = time()
        if (tstep%print_res == 0):
            print("Time [h] {0:.1f}   step {1:3d}   delt {2:3e}   iterations {3:3d}   residuals {4:3e}   div {5:3e}"
            .format((tend-tstart)/3600., tstep, delt, it, res, div))

        if (tstep%print_img == 0):
            save_fields(U, V, P, C, tstep, dir)

