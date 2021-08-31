from LES_constants import *

TEST_CASE = "Poiseuielle_x"
totSteps  = 1000
print_res = 100
print_img = 100
rhoRef    = 1.225e0     # density (air)         [kg/m3]
nuRef     = 1.81e-5     # viscosity (air)       [Pa*s]
pRef      = 101325.0e0     # reference pressure (1 atm)  [Pa]

Lx   = 0.38e0     # system dimension in x-direction   [m]
Ly   = 0.18e0     # system dimension in y-direction   [m]
Nx   = 38         # number of points in x-direction   [-]
Ny   = 18         # number of points in y-direction   [-]
CNum = 0.5        # Courant number 
delt = 0.001      # initial guess for delt

BCs        = [0, 0, 1, 1]    # Boundary conditions: W,E,S,N   0-periodic, 1-wall, 2-fixed inlet velocity
Uin        = -1              # inlet x-velocity. Set to -1 if not specified
Vin        = -1              # inlet y-velocity. Set to -1 if not specified  
Pin        = -1              # inlet pressure. Set to -1 if not specified
Pout       = -1              # outlet pressure. Set to -1 if not specified
DeltaPresX = 1.0e-4          # apply a constant pressure gradient along x-direction
DeltaPresY = zero            # apply a constant pressure gradient along y-direction
dir        = 0               # cross direction for plotting results

