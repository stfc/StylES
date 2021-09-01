from LES_constants import *

TEST_CASE = "Cavity_north"
totSteps  = 100
print_res = 10
print_img = 10
rhoRef    = 1000.0e0       # density (water)    [kg/m3]
nuRef     = 1.0016e-3      # viscosity (water)  [Pa*s]
pRef      = 101325.0e0     # reference pressure (1 atm) [Pa]

Lx   = 0.1e0     # system dimension in x-direction   [m]
Ly   = 0.1e0     # system dimension in y-direction   [m]
Nx   = 20         # number of points in x-direction   [-]
Ny   = 20         # number of points in y-direction   [-]
CNum = 0.5        # Courant number 
delt = 0.00001    # initial guess for delt

BCs        = [1, 1, 1, 2]    # Boundary conditions: W,E,S,N   0-periodic, 1-wall, 2-fixed inlet velocity
Uin        = 1.0             # inlet x-velocity. Set to -1 if not specified
Vin        = 0.0             # inlet y-velocity. Set to -1 if not specified  
Pin        = -1              # inlet pressure. Set to -1 if not specified
Pout       = -1              # outlet pressure. Set to -1 if not specified
DeltaPresX = zero            # apply a constant pressure gradient along x-direction
DeltaPresY = zero            # apply a constant pressure gradient along y-direction
dir        = 0               # cross direction for plotting results
