import math
from LES_constants import *



TEST_CASE = "Pulse"
PASSIVE   = True
totSteps  = 10
print_res = 10
print_img = 10
rhoRef    = 1000.0e0       # density (water)    [kg/m3]
nuRef     = 1.0016e-5      # viscosity (water)  [Pa*s]
pRef      = 101325.0e0     # reference pressure (1 atm) [Pa]

Lx      = 0.01e0     # system dimension in x-direction   [m]
Ly      = 0.01e0     # system dimension in y-direction   [m]
Nx      = 51         # number of points in x-direction   [-]
Ny      = 51         # number of points in y-direction   [-]
deltaX  = Lx/Nx
deltaY  = Ly/Ny
CNum    = 0.5        # Courant number 
delt    = 1.e-3    # initial guess for delt
maxDelt = 1.e-3    # initial guess for delt


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
Umean      = 1.0e-4          # mean flow in x
Vmean      = 1.0e-4           # mean flow in y



def init_flow(U, V, P, C):
    for i in range(Nx):
        for j in range(Ny):
            U[i,j] = Umean
            V[i,j] = Vmean
            P[i,j] = pRef
            dist = (i*Lx/(Nx-1)-puC)**2 + (j*Ly/(Ny-1)-puC)**2
            C[i,j] = puA*math.exp(-dist/(2*puS**2))


