from testcases.Pulse import *


# define runtime parameters
DEBUG     = False
PATH      = "./"
maxIt     = 100000
maxItMom  = 100000
maxItPc   = 100000
maxItC    = 100000
toll      = 1.0e-6    # tollerance for convergence of SIMPLE
alphaP    = 0.1      # pressure relaxation factor
alphaUV   = 0.1       # velocity relaxation factor


# find case dependent parameters
A         = dXY*dXY  # Area                              [m2] 
DX        = nuRef/dXY   # diffusion conductance term in x
DY        = nuRef/dXY   # diffusion conductance term in y
rA        = rhoRef*A
rX        = rhoRef*dXY            
rY        = rhoRef*dXY
rXX       = rhoRef*dXY*dXY
rYY       = rhoRef*dXY*dXY


