import numpy as np
import os

from PIL import Image
from LES_parameters import *
from boundary_conditions import *
from plot import *



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


apply_BCs(U, V, P, BCs)



#-----------------------------------START: main iteration loop
it=0
res=1.0
while (res>toll and it<maxIt):


    # copy over values
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            pc[i][j] = newPc[i][j]


    apply_BCs(U, V, P, BCs)


    if (it%1000 == 0):
        save_fields(U, V, P, it, dir)


    #-------------PREDICTOR STEP: solve momentum equations
    #
    # Remember: the grid is backward staggered!!
    #

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

            ApU[i][j] = Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn)

            rhs = Aw*U[i-1][j] + Ae*U[i+1][j] + As*U[i][j-1] + An*U[i][j+1] \
                - (P[i+1,j] - P[i-1,j])*A + hf*(B[i+1,j]+B[i-1,j]) + DeltaPresX

            newU[i][j] = rhs/ApU[i][j]




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

            ApV[i][j] = Aw + Ae + As + An + (Fe-Fw) + (Fs-Fn)

            rhs = Aw*V[i-1][j] + Ae*V[i+1][j] + As*V[i][j-1] + An*V[i][j+1] \
                - (P[i,j+1] - P[i,j-1])*A + hf*(B[i,j+1]+B[i,j-1]) + DeltaPresY

            newV[i][j] = rhs/ApV[i][j]



    apply_BCs(newU, newV, newPc, BCs)




    #-------------CORRECTOR STEP: solve pressure correction equation

    # find pressure correction
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




    #-------------UPDATE: update values using under relaxation factors
    res = 0.e0
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            P[i][j] = P[i][j] + alphaP*newPc[i][j]
            Ucorr = newU[i][j] + A/ApU[i][j]*(newPc[i-1][j] - newPc[i+1][j])
            Vcorr = newV[i][j] + A/ApV[i][j]*(newPc[i][j-1] - newPc[i][j+1])
            U[i][j] = alphaU*Ucorr + (one-alphaU)*U[i][j]
            V[i][j] = alphaV*Vcorr + (one-alphaV)*V[i][j]
            res = res + abs(newU[i][j] - U[i][j]) + abs(newV[i][j] - V[i][j])



    # print update
    it = it+1
    if (it%print_res == 0):
        print("Iteration {0:3d} residual {1:6e}".format(it, res))


if (it==100):
    print("Solver not converged!!!")





#---------------extra pieces
