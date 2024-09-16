# define runtime parameters
from matplotlib.pyplot import xlim


DEBUG      = False
PATH       = "./"
maxItDNS   = 100000
toll       = 1.0e-4   # tollerance for convergence of SIMPLE
tollM      = 1.0e-3
tollP      = 1.0e-3
tollC      = 1.0e-3
tollDNS    = 1.0e-5
alphaP     = 0.1e0      # pressure relaxation factor
alphaUV    = 0.1e0       # velocity relaxation factor
uRef       = 100.0e0
useLogSca  = True
