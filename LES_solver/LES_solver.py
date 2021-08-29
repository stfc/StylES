import numpy as np
import os

from PIL import Image
from LES_parameters import *
from LES_BC import *
from LES_plot import *



#---------------------------- define arrays
U  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # x-velocity
V  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # y-velocity 
P  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # pressure field

Uo = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # old x-velocity
Vo = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # old y-velocity
Po = np.zeros([Nx+2,Ny+2], dtype=DTYPE)   # old pressure field

iAp = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # central coefficient
Ue  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # face x-velocities
Vn  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # face y-velocities  

pc  = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # pressure correction
nPc = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # pressure correction at new iteration

B   = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # body force




#---------------------------- set flow pressure, velocity fields and BCs
os.system("rm *fields.png")

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





#---------------------------- main time step loop
tstep=0
time=0.0e0
while (tstep<totSteps):

    #---------------------------- save old values of U, V and P
    for i in range(0,Nx+2):
        for j in range(0,Ny+2):
            Uo[i][j] = U[i][j]
            Vo[i][j] = V[i][j]
            Po[i][j] = P[i][j]

    #---------------------------- outer loop on SIMPLE convergence
    it = 0
    res = 1.e0
    while (res>toll and it<maxIt):


        # find face velocities first guess as forward difference (i.e. on side east and north)
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                Ue[i][j] = hf*(U[i+1][j] + U[i][j])
                Vn[i][j] = hf*(V[i][j+1] + V[i][j])

        apply_BCs(Ue, True, BCs)
        apply_BCs(Vn, True, BCs)



        #---------------------------- find Rhei-Chow interpolation (PWIM)
        if (it>0):
            for i in range(1,Nx):
                for j in range(1,Ny):

                    Ue[i][j] = hf*(U[i+1][j] + U[i][j])  \
                             + hf*hf*(P[i+2][j]-P[i  ][j])/iAp[i+1][j]  \
                             + hf*hf*(P[i+1][j]-P[i-1][j])/iAp[i  ][j]  \
                             + hf*(one/iAp[i+1][j] + one/iAp[i][j])*(P[i][j] - P[i+1][j])*deltaY

                    Vn[i][j] = hf*(V[i][j+1] + V[i][j])  \
                             + hf*hf*(P[i][j+2]-P[i][j  ])/iAp[i][j+1]  \
                             + hf*hf*(P[i][j+1]-P[i][j-1])/iAp[i][j  ]  \
                             + hf*(one/iAp[i][j+1] + one/iAp[i][j])*(P[i][j] - P[i][j+1])*deltaX

            apply_BCs(Ue, True, BCs)
            apply_BCs(Vn, True, BCs)



        #---------------------------- solve momentum equations
        for i in range(1,Nx+1):
            ww = one
            ee = one
            if (BCs[0]==1 and i==1):
                ww = zero
            if (BCs[1]==1 and i==Nx):
                ee = zero

            for j in range(1,Ny+1):
                ss = one
                nn = one
                if (BCs[2]==1 and j==1):
                    ss = zero
                if (BCs[3]==1 and j==Ny):
                    nn = zero

                Fw = rhoRef*hf*Ue[i-1][j  ]
                Fe = rhoRef*hf*Ue[i  ][j  ]
                Fs = rhoRef*hf*Vn[i  ][j-1]
                Fn = rhoRef*hf*Vn[i  ][j  ]

                Aw = DX + max(Fw,    zero)
                Ae = DX + max(zero, -Fe)
                As = DY + max(Fs,    zero)
                An = DY + max(zero, -Fn)
                Ao = rA/delt

                iAp[i][j] = one/Ao

                # x-direction
                rhs = (Ao - (Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn)))*Uo[i][j]    \
                    + Aw*Uo[i-1][j] + Ae*Uo[i+1][j] + As*Uo[i][j-1] + An*Uo[i][j+1] \
                    - hf*( ee*Po[i+1,j] + (one-ee)*Po[i,j]          \
                          -ww*Po[i-1,j] - (one-ww)*Po[i,j])*deltaX  \
                    + hf*(B[i+1,j]+B[i-1,j]) + DeltaPresX
                U[i][j] = rhs/Ao

                # y-direction
                rhs = (Ao - (Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn)))*Vo[i][j]     \
                    +  Aw*Vo[i-1][j] + Ae*Vo[i+1][j] + As*Vo[i][j-1] + An*Vo[i][j+1] \
                    - hf*( nn*Po[i,j+1] + (one-nn)*Po[i,j]           \
                          -ss*Po[i,j-1] - (one-ss)*Po[i,j])*deltaY   \
                    + hf*(B[i,j+1]+B[i,j-1]) + DeltaPresY
                V[i][j] = rhs/Ao

        apply_BCs(U, True, BCs)
        apply_BCs(V, True, BCs)





        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = 1.e0
        while (resPc>tollPc and itPc<maxItPc):
            resPc = 0.e0
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

                    Aw = -ww*hf*rYY*(iAp[i-1][j  ] + iAp[i][j])
                    Ae = -ee*hf*rYY*(iAp[i+1][j  ] + iAp[i][j])
                    As = -ss*hf*rXX*(iAp[i  ][j-1] + iAp[i][j])
                    An = -nn*hf*rXX*(iAp[i  ][j+1] + iAp[i][j])
                    Ao = -(Aw+Ae+As+An)
                    So = -(rY*(Ue[i][j]-Ue[i-1][j]) + rX*(Vn[i][j]-Vn[i][j-1]))

                    nPc[i][j] = (So - Aw*pc[i-1][j] - Ae*pc[i+1][j] - As*pc[i][j-1] - An*pc[i][j+1])/Ao

                    resPc = resPc + abs(nPc[i][j] - pc[i][j])

            print("Iterations {0:3d}   residuals {1:3e}".format(itPc, resPc))

            apply_BCs(nPc, False, BCs)
            itPc = itPc+1


        # give warning if solution of the pressure correction is not achieved
        if (itPc==maxItPc):
            print("Attention: pressure correction solver not converged!!!")

        # copy new pc values
        for j in range(1,Ny+1):
            for i in range(1,Nx+1):
                pc[i][j] = nPc[i][j]


        #---------------------------- update values using under relaxation factors
        res = zero
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                prevP = P[i][j]
                prevU = U[i][j] 
                prevV = V[i][j]
                P[i][j] = P[i][j] + alphaP*pc[i][j]
                U[i][j] = U[i][j] + alphaUV*hf*deltaY*iAp[i][j]*(pc[i-1][j  ] - pc[i+1][j  ])
                V[i][j] = V[i][j] + alphaUV*hf*deltaX*iAp[i][j]*(pc[i  ][j-1] - pc[i  ][j+1])
                Ue[i][j] = Ue[i][j] + alphaUV*hf*deltaY*(iAp[i+1][j  ] + iAp[i][j])*(pc[i][j] - pc[i+1][j])
                Vn[i][j] = Vn[i][j] + alphaUV*hf*deltaX*(iAp[i  ][j+1] + iAp[i][j])*(pc[i][j] - pc[i][j+1])

                res = res + abs(prevP - P[i][j]) + abs(prevU - U[i][j]) + abs(prevV - V[i][j])

        apply_BCs(U, True, BCs)
        apply_BCs(V, True, BCs)
        apply_BCs(P,  False, BCs)

        it = it+1
        print("Iterations {0:3d}   residuals {1:3e}".format(it, res))




    #---------------------------- print update and save fields
    if (it==maxIt):
        print("Attention: SIMPLE solver not converged!!!")

    if (tstep%print_res == 0):
        print("Step {0:3d}   time {1:3f}   delt {2:3f}   iterations {3:3d}   residuals {4:3e}"
        .format(tstep, time, delt, it, res))

    if (tstep%print_img == 0):
        save_fields(U, V, P, tstep, dir)

    # find new delt based on Courant number
    delt = min(CNum*deltaX/(np.max(U)+small), CNum*deltaY/(np.max(V)+small))
    time = time + delt
    tstep = tstep+1








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

    #         bPcc[i] = rA*sumAu*(iAp[i-1][j] - iAp[i+1][j]) + rA*sumAv*(iAp[i][j-1] - iAp[i][j+1])


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
    #         iAp[i-1][j-1] = Ap

    #         bU[i-1][j-1] = rhs


    # # solve matrix
    # nU = solve_2D(U[1:Nx], AwU, AeU, AsU, AnU, iAp, bU, Nx, Ny)




