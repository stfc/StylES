from testcases.HIT_2D.HIT_2D import *


# define runtime parameters
DEBUG     = False
PATH      = "./"
maxIt     = 100000
maxItDNS  = 100000
toll      = 1.0e-5   # tollerance for convergence of SIMPLE
tollM     = 1.0e-3
tollP     = 1.0e-3
tollC     = 1.0e-3
tollDNS   = 1.0e-4
alphaP    = 0.1e0      # pressure relaxation factor
alphaUV   = 0.1e0       # velocity relaxation factor
lrDNS     = 0.01

# find case dependent parameters
A  = dl        # Area                              [m2] 
Dc = nu/dl*A   # diffusion conductance term in x


