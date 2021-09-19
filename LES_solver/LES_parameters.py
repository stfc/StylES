from testcases.Pulse import *


# define runtime parameters
DEBUG     = False
PATH      = "./"
maxIt     = 100000
toll      = 1.0e-6   # tollerance for convergence of SIMPLE
tollM     = 1.0e-6
tollP     = 1.0e-6
tollC     = 1.0e-6
alphaP    = 0.1e0      # pressure relaxation factor
alphaUV   = 0.1e0       # velocity relaxation factor


# find case dependent parameters
A  = dl        # Area                              [m2] 
Dc = nu/dl   # diffusion conductance term in x


