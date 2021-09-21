from testcases.HIT_2D import *


# define runtime parameters
DEBUG     = False
PATH      = "./"
maxIt     = 100000
toll      = 1.0e-3   # tollerance for convergence of SIMPLE
tollM     = 1.0e-3
tollP     = 1.0e-3
tollC     = 1.0e-3
alphaP    = 0.1e0      # pressure relaxation factor
alphaUV   = 0.1e0       # velocity relaxation factor


# find case dependent parameters
A  = dl        # Area                              [m2] 
Dc = nu/dl*A   # diffusion conductance term in x


