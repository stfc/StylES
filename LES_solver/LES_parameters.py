import os
import importlib
import sys
import cupy as cp

from LES_constants import *
from testcases.HIT_2D import *

# wrapper for cp.roll
def cr(phi, i, j):
    return cp.roll(phi, (-i, -j), axis=(0,1))


# define runtime parameters
DEBUG     = False
PATH      = "./"
maxIt     = 100000
maxItMom  = 100000
maxItPc   = 100000
maxItC    = 100000
toll      = 1.0e-6    # tollerance for convergence of SIMPLE
tollMom   = 1.0e-6    # tollerance for convergence of Pressure correction
tollPc    = 1.0e-6    # tollerance for convergence of Pressure correction
tollC     = 1.0e-12    # tollerance for convergence of TDMA
alphaP    = 0.1      # pressure relaxation factor
alphaUV   = 0.1       # velocity relaxation factor


# define test case parameters


# find case dependent parameters
A         = deltaX*deltaY  # Area                              [m2] 
DX        = nuRef/deltaX   # diffusion conductance term in x
DY        = nuRef/deltaY   # diffusion conductance term in y
rA        = rhoRef*A
rX        = rhoRef*deltaX            
rY        = rhoRef*deltaY
rXX       = rhoRef*deltaX*deltaX
rYY       = rhoRef*deltaY*deltaY


