# define runtime parameters
from matplotlib.pyplot import xlim


DEBUG      = False
PATH       = "./"
maxIt      = 100000
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
xLogLim    = [1.0e0, 10000]
yLogLim    = [1.e-14, 0.1]
xLinLim    = [0.0e0, 600]
yLinLim    = [0.0e0, 0.1]


# learning rate for DNS field
lrDNS_POLICY = "EXPONENTIAL"   # "EXPONENTIAL" or "PIECEWISE"
lrDNS        = 0.1     # exponential policy initial learning rate
lrDNS_RATE   = 1.0       # exponential policy decay rate
lrDNS_STEP   = maxItDNS     # exponential policy decay step
lrDNS_EXP_ST = False      # exponential policy staircase
lrDNS_BOUNDS = [100, 200, 300]             # piecewise policy bounds
lrDNS_VALUES = [100.0, 50.0, 20.0, 10.0]   # piecewise policy values



# learning rate for reconstruction field
lrREC_POLICY = "EXPONENTIAL"   # "EXPONENTIAL" or "PIECEWISE"
lrREC        = 1.0     # exponential policy initial learning rate
lrREC_RATE   = 1.0       # exponential policy decay rate
lrREC_STEP   = maxItDNS     # exponential policy decay step
lrREC_EXP_ST = False      # exponential policy staircase
lrREC_BOUNDS = [100, 200, 300]             # piecewise policy bounds
lrREC_VALUES = [100.0, 50.0, 20.0, 10.0]   # piecewise policy values

