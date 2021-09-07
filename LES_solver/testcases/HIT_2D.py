import numpy as np
import math
import matplotlib.pyplot as plt

from LES_constants import *


TEST_CASE = "2D_HIT"
totSteps  = 10
print_res = 1
print_img = 1
rhoRef    = 1.0e0          # density (water)    [kg/m3]
nuRef     = 1.0e-10         # viscosity (water)  [Pa*s]
pRef      = 101325.0e0     # reference pressure (1 atm) [Pa]

Lx   = 0.1e0     # system dimension in x-direction   [m]
Ly   = 0.1e0     # system dimension in y-direction   [m]
Nx   = 32         # number of points in x-direction   [-]
Ny   = 32        # number of points in y-direction   [-]
CNum = 0.5        # Courant number 
delt = 1.e-6    # initial guess for delt

BCs        = [0, 0, 0, 0]    # Boundary conditions: W,E,S,N   0-periodic, 1-wall, 2-fixed inlet velocity
Uin        = 0.0             # inlet x-velocity. Set to -1 if not specified
Vin        = 0.0             # inlet y-velocity. Set to -1 if not specified  
Pin        = -1              # inlet pressure. Set to -1 if not specified
Pout       = -1              # outlet pressure. Set to -1 if not specified
DeltaPresX = zero            # apply a constant pressure gradient along x-direction
DeltaPresY = zero            # apply a constant pressure gradient along y-direction
dir        = 0               # cross direction for plotting results

E = np.zeros([Ny], dtype=DTYPE)  # enery spectrum
k = np.zeros([Ny], dtype=DTYPE)  # wave number



def init_flow(U, V, P, C):
    
    # Use Saad procedure adapted for 2D flow
    M  = 100              # number of modes
    kp = 50.0e0*2*np.pi   # peak value
    Q  = 1.0e-18              # constant to normalize energy spectrum  
    deltaX = Lx/Nx
    deltaY = Ly/Ny
    deltaZ = Lx/Nx
    E = np.zeros([M], dtype=DTYPE)  # enery spectrum
    k = np.zeros([M], dtype=DTYPE)  # wave number


    # find max and min wave numbers
    k0   = 2*np.pi/Lx     #same in each direction
    kmax = np.pi/(Lx/Nx)  #same in each direction


    # find k and E
    for m in range(M):
        k[m] = k0 + (kmax-k0)/M*m
        E[m] = Q*k[m]**8*np.exp(-4*(k[m]/kp)**2) + 1.0e-100

    plt.plot(k/(two*np.pi), E)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which='both')
    plt.savefig("Energy_spectrum.png")
    
    np.random.seed(0)

    # find velocity components u and v
    for i in range(0,Nx+2):

        print("Find velocity component for i: ", i)

        for j in range(0,Ny+2):
            P[i][j] = pRef
            U[i][j] = zero
            V[i][j] = zero
    
            for m in range(M-1):
                qm = np.sqrt(E[m]*(k[m+1]-k[m]))

                # compute km vectors
                thetaM  = np.random.uniform()
                phiM    = np.random.uniform()
                theta2M = np.random.uniform()
                phi2M   = np.random.uniform()

                kxm = np.sin(thetaM)*np.cos(phiM)
                kym = np.sin(thetaM)*np.sin(phiM)
                kzm = np.cos(thetaM)

                # compute tilde km
                kxt = two/deltaX*np.sin(hf*kxm*deltaX)
                kyt = two/deltaY*np.sin(hf*kym*deltaY)
                kzt = two/deltaZ*np.sin(hf*kzm*deltaZ)

                # set z component to zero
                zetax = np.sin(theta2M)*np.cos(phi2M)
                zetay = zetax*kyt/kxt
                zetaz = np.cos(theta2M)

                # compute the unit vectors
                sigmax =  zetay*kzt - zetaz*kyt
                sigmay = -(zetax*kzt - zetaz*kxt)
                sigmaz =  zetax*kyt - zetay*kxt

                sigma = np.sqrt(sigmax*sigmax + sigmay*sigmay + sigmaz*sigmaz)

                sigmax =  sigmax/sigma
                sigmay =  sigmay/sigma
                sigmaz =  sigmaz/sigma

                # verify orthogonality
                #kk = np.sum(kxt*sigmax + kyt*sigmay + kzt*sigmaz)
                #print(" Orthogonality ok and sigma ",kk, sigmaz) 

                # compute u and v
                x = i*deltaX
                y = j*deltaY
                z = zero

                KX = kxm*x + kym*y + kzm*z 

                U[i][j] = U[i][j] + two*qm*np.cos(KX + phiM)*sigmax
                V[i][j] = V[i][j] + two*qm*np.cos(KX + phiM)*sigmay





