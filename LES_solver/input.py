import math
from LES_constants import *

TEST_CASE = "Pulse"
totSteps  = 15
print_res = 1
print_img = 1
rhoRef    = 1000.0e0       # density (water)    [kg/m3]
nuRef     = 1.0016e-3      # viscosity (water)  [Pa*s]
pRef      = 101325.0e0     # reference pressure (1 atm) [Pa]

Lx   = 0.01e0     # system dimension in x-direction   [m]
Ly   = 0.01e0     # system dimension in y-direction   [m]
Nx   = 21         # number of points in x-direction   [-]
Ny   = 21         # number of points in y-direction   [-]
CNum = 0.5        # Courant number 
delt = 1.e-3    # initial guess for delt


BCs        = [0, 0, 0, 0]    # Boundary conditions: W,E,S,N   0-periodic, 1-wall, 2-fixed inlet velocity
Uin        = 0.0             # inlet x-velocity. Set to -1 if not specified
Vin        = 0.0             # inlet y-velocity. Set to -1 if not specified  
Pin        = -1              # inlet pressure. Set to -1 if not specified
Pout       = -1              # outlet pressure. Set to -1 if not specified
DeltaPresX = zero            # apply a constant pressure gradient along x-direction
DeltaPresY = zero            # apply a constant pressure gradient along y-direction
dir        = 0               # cross direction for plotting results
puA        = 1.0e0           # amplitude pulse
puC        = Lx/2            # peak center
puS        = 1.0e-3          # standard deviation
Umean      = 0.001e0          # mean flow in x
Vmean      = 0.001e0           # mean flow in y



def init_flow(U, V, P, C):
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            U[i][j] = Umean
            V[i][j] = Vmean
            P[i][j] = pRef
            dist = ((i-1)*Lx/(Nx-1)-puC)**2 + ((j-1)*Ly/(Ny-1)-puC)**2
            C[i][j] = puA*math.exp(-dist/(2*puS**2))


