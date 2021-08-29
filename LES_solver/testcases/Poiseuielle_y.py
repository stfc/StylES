from LES_constants import *

TEST_CASE = "Poiseuielle_y"
totSteps  = 101
print_res = 10
print_img = 10
rhoRef    = 1.225e0     # density (air)                     [kg/m3]
nuRef     = 1.81e-5     # viscosity (air)                   [Pa*s]
pRef      = 101325.0e0     # reference pressure (1 atm)

Lx   = 0.18e0     # system dimension in x-direction   [m]
Ly   = 0.38e0     # system dimension in y-direction   [m]
Nx   = 18         # number of points in x-direction   [-]
Ny   = 38         # number of points in y-direction   [-]
CNum = 0.5        # Courant number 
delt = 0.001      # initial guess for delt

BCs = [1, 1, 0, 0]    # Boundary conditions: E,W,S,N   0-periodic, 1-wall
DeltaPresX = zero     # apply a constant pressure gradient along x-direction
DeltaPresY = 1.0e-4   # apply a constant pressure gradient along y-direction
dir = 1               # cross direction for plotting results

