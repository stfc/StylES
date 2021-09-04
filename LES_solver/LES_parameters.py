import os

from LES_constants import *

# define runtime parameters
DEBUG     = False
PATH      = "./"
DTYPE     = "float64"
maxIt     = 100000
maxItPc   = 100000
maxItC    = 100000
maxDelt   = 1.0e0
toll      = 1.0e-6    # tollerance for convergence of SIMPLE
tollPc    = 1.0e-6    # tollerance for convergence of Pressure correction
tollTDMA  = 1.0e-12    # tollerance for convergence of TDMA
alphaP    = 0.1       # pressure relaxation factor
alphaUV   = 0.1       # velocity relaxation factor


# define test case parameters
TEST_CASE = "testcases/Pulse.py"
cmd = "cp " + TEST_CASE + " input.py"
os.system(cmd)

from input import *


# find case dependent parameters
deltaX    = Lx/Nx          #                                   [m]
deltaY    = Ly/Ny          #                                   [m]
A         = deltaX*deltaY  # Area                              [m2] 
DX        = nuRef/deltaX   # diffusion conductance term in x
DY        = nuRef/deltaY   # diffusion conductance term in y
rA        = rhoRef*A
rX        = rhoRef*deltaX            
rY        = rhoRef*deltaY
rXX       = rhoRef*deltaX*deltaX
rYY       = rhoRef*deltaY*deltaY


