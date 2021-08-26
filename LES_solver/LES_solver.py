import numpy as np
import os

from PIL import Image
from LES_parameters import *
from boundary_conditions import *
from plot import *
from matrix_solvers import *



# define arrays
U     = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # x-velocity
V     = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # y-velocity 
ApU   = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # pewssure correction
ApV   = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # pewssure correction
newU  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)
newV  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)

B     = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # body force
P     = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # pressure field
pc    = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # pewssure correction
newPc = np.zeros([Nx+2,Ny+2], dtype=DTYPE)




#---clean up
os.system("rm *fields.png")


#-----------------------------------INITIALIZE: set flow pressure, velocity fields and BCs
if (TEST_CASE=="Poiseuielle_x"):

    dir = 0
    BCs = [0, 0, 1, 1]    # w-periodic, e-periodic, s-wall, n-wall

    DeltaPresX = 1.0e-4    # apply a constant pressure gradient along x-direction
    DeltaPresY = zero

    # initial flow
    for i in range(Nx+2):
        for j in range(Ny+2):
            U[i][j] = zero
            V[i][j] = zero
            P[i][j] = pRef


if (TEST_CASE=="Poiseuielle_y"):
    
    dir = 1
    BCs = [1, 1, 0, 0]    # w-periodic, e-periodic, s-wall, n-wall

    DeltaPresX = zero     # apply a constant pressure gradient along x-direction
    DeltaPresY = 1.0e-4

    # initial flow
    for i in range(Nx+2):
        for j in range(Ny+2):
            U[i][j] = zero
            V[i][j] = zero
            P[i][j] = pRef


apply_BCs(U, True, BCs)
apply_BCs(V, True, BCs)
apply_BCs(P, False, BCs)



#-----------------------------------START: time step loop
tstep=0
time=0.0e0
while (tstep<totSteps):


    #-------------PREDICTOR STEP: solve momentum equations

    # find new velocity field in x-direction
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            Fw = rhoRef*hf*(U[i][j] + U[i-1][j  ])
            Fe = rhoRef*hf*(U[i][j] + U[i+1][j  ])
            Fs = rhoRef*hf*(V[i][j] + V[i  ][j-1])
            Fn = rhoRef*hf*(V[i][j] + V[i  ][j+1])

            Aw = D + max(Fw,    zero)
            Ae = D + max(zero, -Fe)
            As = D + max(Fs,    zero)
            An = D + max(zero, -Fn)
            Ao = rA/delt

            ApU[i][j] = Ao

            rhs = (Ao - (Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn)))*U[i][j]    \
                + Aw*U[i-1][j] + Ae*U[i+1][j] + As*U[i][j-1] + An*U[i][j+1] \
                - (P[i+1,j] - P[i-1,j])*A + hf*(B[i+1,j]+B[i-1,j]) + DeltaPresX

            newU[i][j] = rhs/Ao




    # find new velocity field in y-direction
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            Fw = rhoRef*hf*(U[i][j] + U[i-1][j  ])
            Fe = rhoRef*hf*(U[i][j] + U[i+1][j  ])
            Fs = rhoRef*hf*(V[i][j] + V[i  ][j-1])
            Fn = rhoRef*hf*(V[i][j] + V[i  ][j+1])

            Aw = D + max(Fw,    zero)
            Ae = D + max(zero, -Fe)
            As = D + max(Fs,    zero)
            An = D + max(zero, -Fn)
            Ao = rA/delt

            ApV[i][j] = Ao 

            rhs = (Ao - (Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn)))*V[i][j]     \
                +  Aw*V[i-1][j] + Ae*V[i+1][j] + As*V[i][j-1] + An*V[i][j+1] \
                - (P[i,j+1] - P[i,j-1])*A + hf*(B[i,j+1]+B[i,j-1]) + DeltaPresY

            newV[i][j] = rhs/Ao



    apply_BCs(newU, True, BCs)
    apply_BCs(newV, True, BCs)





    #-------------CORRECTOR STEP: solve pressure correction equation

    # find pressure correction
    it  = 0
    res = 1.e0
    while (res>toll and it<maxIt):
        res = 0.e0
        for j in range(1,Ny+1):
            ss = one
            nn = one
            if (BCs[2]==1 and j==1):
                ss = zero
            if (BCs[3]==1 and j==Ny):
                nn = zero

            for i in range(1,Nx+1):
                ww = one
                ee = one
                if (BCs[0]==1 and i==1):
                    ww = zero
                if (BCs[1]==1 and i==Nx):
                    ee = zero

                rhs = rA*(ww*pc[i-1][j]   + ee*pc[i+1][j]   + ss*pc[i][j-1]   + nn*pc[i][j+1]) \
                    + rA*(ww*newU[i-1][j] - ee*newU[i+1][j] + ss*newV[i][j-1] - nn*newV[i][j+1])

                newPc[i][j] = rhs / (4.0e0*rA - (4.0e0-ww-ee-ss-nn)*rA)

                res = res + abs(newPc[i][j] - pc[i][j])

        for j in range(1,Ny+1):
            for i in range(1,Nx+1):
                pc[i][j] = newPc[i][j]

        apply_BCs(pc, False, BCs)

        it = it+1

    if (it==100):
        print("Solver not converged!!!")




    #-------------UPDATE: update values using under relaxation factors
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            P[i][j] = P[i][j] + alphaP*newPc[i][j]
            Ucorr = newU[i][j] + A/ApU[i][j]*(newPc[i-1][j] - newPc[i+1][j])
            Vcorr = newV[i][j] + A/ApV[i][j]*(newPc[i][j-1] - newPc[i][j+1])
            U[i][j] = alphaU*Ucorr + (one-alphaU)*U[i][j]
            V[i][j] = alphaV*Vcorr + (one-alphaV)*V[i][j]


    apply_BCs(U, True, BCs)
    apply_BCs(V, True, BCs)
    apply_BCs(P, False, BCs)


    # print update
    if (tstep%print_res == 0):
        print("Step {0:3d}   time {1:3f}   iterations {2:3d}   residuals {3:3e}".format(tstep, time, it, res))


    # save images
    if (tstep%print_img == 0):
        save_fields(U, V, P, tstep, dir)


    tstep=tstep+1
    time = time + delt








#---------------extra pieces

    # #-------------2nd CORRECTOR STEP: solve pressure correction equation

    # # find pressure correction
    # for j in range(1,Ny+1):
    #     ss = one
    #     nn = one
    #     if (BCs[2]==1 and j==1):
    #         ss = zero
    #     if (BCs[3]==1 and j==Ny):
    #         nn = zero

    #     for i in range(1,Nx+1):
    #         ww = one
    #         ee = one
    #         if (BCs[0]==1 and i==1):
    #             ww = zero
    #         if (BCs[1]==1 and i==Nx):
    #             ee = zero

    #         Fw = rhoRef*hf*(U[i][j] + U[i-1][j  ])
    #         Fe = rhoRef*hf*(U[i][j] + U[i+1][j  ])
    #         Fs = rhoRef*hf*(V[i][j] + V[i  ][j-1])
    #         Fn = rhoRef*hf*(V[i][j] + V[i  ][j+1])

    #         Aw = D + max(Fw,    zero)
    #         Ae = D + max(zero, -Fe)
    #         As = D + max(Fs,    zero)
    #         An = D + max(zero, -Fn)

    #         mPcc[i  ][j  ] = 4.0e0*rA - (4.0e0-ww-ee-ss-nn)*rA
    #         mPcc[i-1][j  ] = ww*rA
    #         mPcc[i+1][j  ] = ee*rA
    #         mPcc[i  ][j-1] = ss*rA
    #         mPcc[i  ][j+1] = nn*rA

    #         sumAu = ww*Aw*(Unnw[i-1][j] - Unew[i-1][j])  \
    #               + ee*Ae*(Unnw[i+1][j] - Unew[i-1][j])  \
    #               + ss*As*(Unnw[i][j-1] - Unew[i][j-1])  \
    #               + nn*An*(Unnw[i][j+1] - Unew[i][j+1])

    #         sumAv = ww*Aw*(Vnnw[i-1][j] - Vnew[i-1][j])  \
    #               + ee*Ae*(Vnnw[i+1][j] - Vnew[i-1][j])  \
    #               + ss*As*(Vnnw[i][j-1] - Vnew[i][j-1])  \
    #               + nn*An*(Vnnw[i][j+1] - Vnew[i][j+1])

    #         bPcc[i] = rA*sumAu*(ApU[i-1][j] - ApU[i+1][j]) + rA*sumAv*(ApV[i][j-1] - ApV[i][j+1])


    # newPcc = linalg.solve(mPcc, bPcc)



    # #---------find new velocity field in x-direction
    # for i in range(1,Nx+1):
    #     for j in range(1,Ny+1):
    #         Fw = rhoRef*hf*(U[i][j] + U[i-1][j  ])
    #         Fe = rhoRef*hf*(U[i][j] + U[i+1][j  ])
    #         Fs = rhoRef*hf*(V[i][j] + V[i  ][j-1])
    #         Fn = rhoRef*hf*(V[i][j] + V[i  ][j+1])

    #         Aw = D + max(Fw,    zero)
    #         Ae = D + max(zero, -Fe)
    #         As = D + max(Fs,    zero)
    #         An = D + max(zero, -Fn)
    #         Ao = rhoRef*A/delt

    #         ww = one
    #         ee = one
    #         ss = one
    #         nn = one

    #         if (i==1):
    #             if (BCs[0]==0):
    #                 Aw = zero
    #                 ww = zero
    #             elif (BCs[0]==1):   
    #                 Aw = zero
    #                 ww = one

    #         if (i==Nx):
    #             if (BCs[1]==0):
    #                 Ae = zero
    #                 ee = zero
    #             elif (BCs[1]==1):   
    #                 Aw = zero
    #                 ee = one

    #         if (j==1):
    #             if (BCs[2]==0):
    #                 As = zero
    #                 ss = zero
    #             elif (BCs[2]==1):   
    #                 As = zero
    #                 ss = one

    #         if (j==Ny):
    #             if (BCs[3]==0):
    #                 An = zero
    #                 nn = zero
    #             elif (BCs[3]==1):   
    #                 An = zero
    #                 nn = one

    #         Ap = Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn) + Ao

    #         rhs = Aw*U[i-1][j] + Ae*U[i+1][j] + As*U[i][j-1] + An*U[i][j+1] + Ao*U[i][j]       \
    #             + Aw*(one-ww)*U[Nx][j] + Ae*(one-ee)*U[1][j] + As*(one-ss)*V[i][Ny] + An*(one-nn)*U[i][1]  \
    #             - (P[i+1,j] - P[i-1,j])*A + hf*(B[i+1,j]+B[i-1,j]) + DeltaPresX

    #         AwU[i-1][j-1] = Aw
    #         AeU[i-1][j-1] = Ae
    #         AsU[i-1][j-1] = As
    #         AnU[i-1][j-1] = An
    #         ApU[i-1][j-1] = Ap

    #         bU[i-1][j-1] = rhs


    # # solve matrix
    # newU = solve_2D(U[1:Nx], AwU, AeU, AsU, AnU, ApU, bU, Nx, Ny)




