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

save_fields(U, V, P, 0, dir)




# find face velocities first guess as forward difference (i.e. on side east and north)
for i in range(1,Nx+1):
    for j in range(1,Ny+1):
        Ue[i][j] = hf*(U[i+1][j] + U[i][j])
        Vn[i][j] = hf*(V[i][j+1] + V[i][j])

apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)


#---------------------------- main time step loop
tstep=0
time=0.0e0

print("Step {0:3d}   time {1:3f}   delt {2:3f}   iterations {3:3d}   residuals {4:3e}"
    .format(tstep, time, delt, 0, zero))

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
                for j in range(1,Ny+1):
                    deltpX2 = hf*(P[i+1][j] - P[i-1][j])
                    deltpX3 = hf*(P[i  ][j] - P[i+1][j])

                    if (i==1):
                        deltpX1 = hf*(P[i+2][j] - P[i  ][j])                
                        if   (BCs[0]==0):  # periodic
                            pass
                        elif (BCs[0]==1):  # no-slip (wall)
                            deltpX2 = hf*(P[i+1][j]+P[i][j]) - P[i][j]  # first order approx Pw=Po
                        elif (BCs[0]==2):  # fixed inlet velocity
                            deltpX2 = hf*(P[i+1][j]+P[i][j]) - P[i][j]  # first order approx Pw=Po
                        elif (BCs[0]==3):  # fixed inlet pressure
                            deltpX2 = hf*(P[i+1][j]+P[i][j]) - Pin
                        elif (BCs[0]==4):  # fixed outlet pressure
                            deltpX2 = hf*(P[i+1][j]+P[i][j]) - Pout
                    elif (i==Nx):
                        if   (BCs[1]==0):  # periodic
                            deltpX1 = hf*(P[2][j] - P[i  ][j])                
                        elif (BCs[1]==1):  # no-slip (wall)
                            deltpX1 = (P[i][j] - P[i+1][j])/onePfive  # first order approx Pe=Po
                        elif (BCs[1]==2):  # fixed inlet velocity
                            deltpX1 = (P[i][j] - P[i+1][j])/onePfive   # first order approx Pw=Po
                        elif (BCs[1]==3):  # fixed inlet pressure
                            deltpX1 = (P[i][j] - Pin)/onePfive
                        elif (BCs[1]==4):  # fixed outlet pressure
                            deltpX1 = (P[i][j] - Pout)/onePfive
                    else:
                        deltpX1 = hf*(P[i+2][j] - P[i  ][j])                

                    deltpY2 = hf*(P[i][j+1] - P[i][j-1])
                    deltpY3 = hf*(P[i][j  ] - P[i][j+1])

                    if (j==1):
                        deltpY1 = hf*(P[i][j+2] - P[i][j  ])                
                        if   (BCs[2]==0):  # periodic
                            pass
                        elif (BCs[2]==1):  # no-slip (wall)
                            deltpY2 = hf*(P[i][j+1]+P[i][j]) - P[i][j]  # first order approx Pw=Po
                        elif (BCs[2]==2):  # fixed inlet velocity
                            deltpY2 = hf*(P[i][j+1]+P[i][j]) - P[i][j]  # first order approx Pw=Po
                        elif (BCs[2]==3):  # fixed inlet pressure
                            deltpY2 = hf*(P[i][j+1]+P[i][j]) - Pin
                        elif (BCs[2]==4):  # fixed outlet pressure
                            deltpY2 = hf*(P[i][j+1]+P[i][j]) - Pout
                    elif (j==Ny):
                        if   (BCs[3]==0):  # periodic
                            deltpY1 = hf*(P[i][2] - P[i][j  ]) 
                        elif (BCs[3]==1):  # no-slip (wall)
                            deltpY1 = (P[i][j] - P[i][j+1])/onePfive  # first order approx Pe=Po
                        elif (BCs[3]==2):  # fixed inlet velocity
                            deltpY1 = (P[i][j] - P[i][j+1])/onePfive   # first order approx Pw=Po
                        elif (BCs[3]==3):  # fixed inlet pressure
                            deltpY1 = (P[i][j] - Pin)/onePfive
                        elif (BCs[3]==4):  # fixed outlet pressure
                            deltpY1 = (P[i][j] - Pout)/onePfive
                    else:
                        deltpY1 = hf*(P[i][j+2] - P[i][j  ])                

                    Ue[i][j] = hf*(U[i+1][j] + U[i][j])       \
                             + hf*deltpX1*iAp[i+1][j]*deltaY  \
                             + hf*deltpX2*iAp[i  ][j]*deltaY    \
                             + hf*deltpX3*(iAp[i+1][j] + iAp[i][j])*deltaY

                    Vn[i][j] = hf*(V[i][j+1] + V[i][j])       \
                             + hf*deltpY1*iAp[i][j+1]*deltaX  \
                             + hf*deltpY2*iAp[i][j  ]*deltaX  \
                             + hf*deltpY3*(iAp[i][j+1] + iAp[i][j])*deltaX

            apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)

        # save_fields(Ue, Vn, P, tstep, dir)

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
        # save_fields(U, V, P, tstep, dir)

        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = large
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
                    pc[i][j] = nPc[i][j]

            if (itPc<maxItPc-1):
                #print("Iterations {0:3d}   residuals {1:3e}".format(itPc, resPc))
                for i in range(1,Nx+1):
                    for j in range(1,Ny+1):
                        pc[i][j] = nPc[i][j]

                apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)
            else:
                # give warning if solution of the pressure correction is not achieved
                print("Attention: pressure correction solver not converged!!!")
                save_fields(U, V, pc, tstep, dir)
                exit()

            itPc = itPc+1


        # save_fields(U, V, pc, tstep, dir)

        #---------------------------- update values using under relaxation factors
        res = zero
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):

                deltpX1 = pc[i-1][j] - pc[i+1][j]
                deltpX2 = pc[i  ][j] - pc[i+1][j]

                if (i==1):
                    if   (BCs[0]==0):  # periodic
                        pass
                    elif (BCs[0]==1):  # no-slip (wall)
                        deltpX1 = (pc[i][j] - pc[i+1][j])/onePfive
                        deltpX2 =  pc[i][j] - pc[i+1][j]
                    elif (BCs[0]==2):  # fixed inlet velocity
                        deltpX1 = (pc[i][j] - pc[i+1][j])/onePfive
                        deltpX2 =  pc[i][j] - pc[i+1][j]
                    elif (BCs[0]==3):  # fixed inlet pressure
                        deltpX1 = (zero     - pc[i+1][j])/onePfive
                        deltpX2 =  pc[i][j] - pc[i+1][j]
                    elif (BCs[0]==4):  # fixed outlet pressure
                        deltpX1 = (zero     - pc[i+1][j])/onePfive
                        deltpX2 =  pc[i][j] - pc[i+1][j]

                if (i==Nx):
                    if   (BCs[1]==0):  # periodic
                        pass
                    elif (BCs[1]==1):  # no-slip (wall)
                        deltpX1 = (pc[i-1][j] - pc[i][j])/onePfive
                        deltpX2 = (pc[i  ][j] - pc[i][j])*hf
                    elif (BCs[1]==2):  # fixed inlet velocity
                        deltpX1 = (pc[i-1][j] - pc[i][j])/onePfive
                        deltpX2 = (pc[i  ][j] - pc[i][j])*hf
                    elif (BCs[1]==3):  # fixed inlet pressure
                        deltpX1 = (pc[i-1][j] - zero)/onePfive
                        deltpX2 = (pc[i  ][j] - zero)*hf
                    elif (BCs[1]==4):  # fixed outlet pressure
                        deltpX1 = (pc[i-1][j] - zero)/onePfive
                        deltpX2 = (pc[i  ][j] - zero)*hf

                deltpY1 = pc[i][j-1] - pc[i][j+1]               
                deltpY2 = pc[i][j  ] - pc[i][j+1]

                if (j==1):
                    if   (BCs[2]==0):  # periodic
                        pass
                    elif (BCs[2]==1):  # no-slip (wall)
                        deltpY1 = (pc[i][j] - pc[i][j+1])/onePfive
                        deltpY2 =  pc[i][j] - pc[i][j+1]
                    elif (BCs[2]==2):  # fixed inlet velocity
                        deltpY1 = (pc[i][j] - pc[i][j+1])/onePfive
                        deltpY2 =  pc[i][j] - pc[i][j+1]
                    elif (BCs[2]==3):  # fixed inlet pressure
                        deltpY1 = (zero     - pc[i][j+1])/onePfive
                        deltpY2 =  pc[i][j] - pc[i][j+1]
                    elif (BCs[2]==4):  # fixed outlet pressure
                        deltpY1 = (zero     - pc[i][j+1])/onePfive
                        deltpY2 =  pc[i][j] - pc[i][j+1]

                if (j==Ny):
                    if   (BCs[3]==0):  # periodic
                        pass
                    elif (BCs[3]==1):  # no-slip (wall)
                        deltpY1 = (pc[i][j-1] - pc[i][j])/onePfive
                        deltpY2 = (pc[i][j  ] - pc[i][j])*hf
                    elif (BCs[3]==2):  # fixed inlet velocity
                        deltpY1 = (pc[i][j-1] - pc[i][j])/onePfive
                        deltpY2 = (pc[i][j  ] - pc[i][j])*hf
                    elif (BCs[3]==3):  # fixed inlet pressure
                        deltpY1 = (pc[i][j-1] - zero)/onePfive
                        deltpY2 = (pc[i][j  ] - zero)*hf
                    elif (BCs[3]==4):  # fixed outlet pressure
                        deltpY1 = (pc[i][j-1] - zero)/onePfive
                        deltpY2 = (pc[i][j  ] - zero)*hf

                prevP = P[i][j]
                prevU = U[i][j] 
                prevV = V[i][j]
                P[i][j] = P[i][j] + alphaP*pc[i][j]
                U[i][j] = U[i][j] + alphaUV*hf*deltaY*iAp[i][j]*deltpX1
                V[i][j] = V[i][j] + alphaUV*hf*deltaX*iAp[i][j]*deltpY1
                Ue[i][j] = Ue[i][j] + alphaUV*hf*deltaY*(iAp[i+1][j  ] + iAp[i][j])*deltpX2
                Vn[i][j] = Vn[i][j] + alphaUV*hf*deltaX*(iAp[i  ][j+1] + iAp[i][j])*deltpY2

                res = res + abs(prevP - P[i][j]) + abs(prevU - U[i][j]) + abs(prevV - V[i][j])

        apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout)
        # save_fields(U, V, P, tstep, dir)
        it = it+1
        #print("Iterations {0:3d}   residuals {1:3e}".format(it, res))




    #---------------------------- print update and save fields
    if (it==maxIt):
        print("Attention: SIMPLE solver not converged!!!")
        exit()

    else:
        # find new delt based on Courant number
        delt = min(CNum*deltaX/(abs(np.max(U))+small), CNum*deltaY/(abs(np.max(V))+small))
        time = time + delt
        tstep = tstep+1

        if (tstep%print_res == 0):
            print("Step {0:3d}   time {1:3f}   delt {2:3f}   iterations {3:3d}   residuals {4:3e}"
            .format(tstep, time, delt, it, res))

        if (tstep%print_img == 0):
            save_fields(U, V, P, tstep, dir)

