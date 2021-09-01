from LES_constants import *

TEST_CASE = "Pipe_x"
totSteps  = 101
print_res = 1
print_img = 1
rhoRef    = 1000.0e0       # density (water)    [kg/m3]
nuRef     = 1.0016e-3      # viscosity (water)  [Pa*s]
pRef      = 101325.0e0     # reference pressure (1 atm)  [Pa]

Lx   = 1.0e0     # system dimension in x-direction   [m]
Ly   = 1.0e0     # system dimension in y-direction   [m]
Nx   = 10         # number of points in x-direction   [-]
Ny   = 10         # number of points in y-direction   [-]
CNum = 0.5e0        # Courant number 
delt = 0.001e0      # initial guess for delt

BCs        = [3, 4, 1, 1]    # Boundary conditions: W,E,S,N   0-periodic, 1-wall, 2-fixed inlet velocity
                             # 3-fixed inlet pressure, 4-fixed outlet pressure
Uin        = -1              # inlet x-velocity. Set to -1 if not specified
Vin        = -1              # inlet y-velocity. Set to -1 if not specified  
Pin        = 1.001e0*pRef          # inlet pressure. Set to -1 if not specified
Pout       = pRef            # outlet pressure. Set to -1 if not specified
DeltaPresX = zero            # apply a constant pressure gradient along x-direction
DeltaPresY = zero            # apply a constant pressure gradient along y-direction
dir        = 0               # cross direction for plotting results

