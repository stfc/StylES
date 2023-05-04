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

xLogLim    = [1.0e0, 1000]   # to do: to make nmore general
yLogLim    = [1.e-8, 0.1]
xLinLim    = [0.0e0, 600]
yLinLim    = [0.0e0, 1.0]

# xLogLim    = [1.0e-1, 1.e+3]
# yLogLim    = [1.e-11, 1.e+2]
# xLinLim    = [0.0e0, 600]
# yLinLim    = [0.0e0, 0.1]

