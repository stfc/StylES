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

# initial flow
for i in range(Nx+2):
    for j in range(Ny+2):
        U[i][j] = zero
        V[i][j] = zero
        P[i][j] = pRef

apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)




# find face velocities first guess as forward difference (i.e. on side east and north)
for i in range(1,Nx+1):
    for j in range(1,Ny+1):
        Ue[i][j] = hf*(U[i+1][j] + U[i][j])
        Vn[i][j] = hf*(V[i][j+1] + V[i][j])

apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)


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
    res = large
    while (res>toll and it<maxIt):


        #---------------------------- find Rhei-Chow interpolation (PWIM)
        if (tstep>0):
            for i in range(1,Nx+1):
                ipp = i+2
                if (i==Nx):
                    if (BCs[0]==0):
                        ipp=0
                    else:
                        ipp=i+1
                for j in range(1,Ny+1):
                    jpp = j+2
                    if (j==Ny):
                        if (BCs[2]==0):
                            jpp=0
                        else:
                            jpp=j+1

                    Ue[i][j] = hf*(U[i+1][j] + U[i][j])  \
                             + hf*hf*(P[ipp][j]-P[i  ][j])*iAp[i+1][j]  \
                             + hf*hf*(P[i+1][j]-P[i-1][j])*iAp[i  ][j]  \
                             + hf*(iAp[i+1][j] + iAp[i][j])*(P[i][j] - P[i+1][j])*deltaY

                    Vn[i][j] = hf*(V[i][j+1] + V[i][j])  \
                             + hf*hf*(P[i][jpp]-P[i][j  ])*iAp[i][j+1]  \
                             + hf*hf*(P[i][j+1]-P[i][j-1])*iAp[i][j  ]  \
                             + hf*(iAp[i][j+1] + iAp[i][j])*(P[i][j] - P[i][j+1])*deltaX

            apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)


        #---------------------------- solve momentum equations
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):

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

                ww, ee, ss, nn, rhsU, rhsV = findBCMomCoeff(i, j, U, V, Po, Uin, Vin, Pin, Pout, rhoRef)

                # x-direction
                rhsU = rhsU + (Ao - (Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn)))*Uo[i][j]    \
                    + Aw*Uo[i-1][j] + Ae*Uo[i+1][j] + As*Uo[i][j-1] + An*Uo[i][j+1]      \
                    - hf*(ee*Po[i+1,j] - ww*Po[i-1,j])*deltaY                            \
                    + hf*(B[i+1,j]+B[i-1,j]) + DeltaPresX
                U[i][j] = rhsU/Ao

                # y-direction
                rhsV = rhsV + (Ao - (Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn)))*Vo[i][j]    \
                    +  Aw*Vo[i-1][j] + Ae*Vo[i+1][j] + As*Vo[i][j-1] + An*Vo[i][j+1]     \
                    - hf*(nn*Po[i,j+1] - ss*Po[i,j-1])*deltaX                            \
                    + hf*(B[i,j+1]+B[i,j-1]) + DeltaPresY
                V[i][j] = rhsV/Ao


        apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)

        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = large
        while (resPc>tollPc and itPc<maxItPc):
            resPc = 0.e0
            for j in range(1,Ny+1):
                ss = one
                nn = one
                if (BCs[2]>0 and j==1):
                    ss = zero
                if (BCs[3]>0 and j==Ny):
                    nn = zero

                for i in range(1,Nx+1):
                    ww = one
                    ee = one
                    if (BCs[0]>0 and i==1):
                        ww = zero
                    if (BCs[1]>0 and i==Nx):
                        ee = zero

                    Aw = -ww*hf*rYY*(iAp[i-1][j  ] + iAp[i][j])
                    Ae = -ee*hf*rYY*(iAp[i+1][j  ] + iAp[i][j])
                    As = -ss*hf*rXX*(iAp[i  ][j-1] + iAp[i][j])
                    An = -nn*hf*rXX*(iAp[i  ][j+1] + iAp[i][j])
                    Ao = -(Aw+Ae+As+An)
                    So = -(rY*(Ue[i][j]-Ue[i-1][j]) + rX*(Vn[i][j]-Vn[i][j-1]))

                    nPc      = (So - Aw*pc[i-1][j] - Ae*pc[i+1][j] - As*pc[i][j-1] - An*pc[i][j+1])/Ao
                    resPc    = resPc + abs(nPc - pc[i][j])
                    pc[i][j] = nPc

            #print("Iterations {0:3d}   residuals {1:3e}".format(itPc, resPc))

            # copy new pc values
            if (itPc<maxItPc-1):
                apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)
            else:
                # give warning if solution of the pressure correction is not achieved
                print("Attention: pressure correction solver not converged!!!")
                save_fields(nPc, pc, nPc-pc, tstep, dir)
                exit()

            itPc = itPc+1


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

        apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)

        it = it+1
        #print("Iterations {0:3d}   residuals {1:3e}".format(it, res))


    #---------------------------- print update and save fields
    if (it==maxIt):
        print("Attention: SIMPLE solver not converged!!!")
        exit()

    else:
        if (tstep%print_res == 0):
            print("Step {0:3d}   time {1:3f}   delt {2:3f}   iterations {3:3d}   residuals {4:3e}"
            .format(tstep, time, delt, it, res))

        if (tstep%print_img == 0):
            save_fields(U, V, P, tstep, dir)

        # find new delt based on Courant number
        delt = min(CNum*deltaX/(abs(np.max(U))+small), CNum*deltaY/(abs(np.max(V))+small))
        time = time + delt
        tstep = tstep+1
