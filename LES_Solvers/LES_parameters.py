# define runtime parameters
DEBUG      = False
PATH       = "./"
maxIt      = 100000
maxItDNS   = 10000
toll       = 1.0e-5   # tollerance for convergence of SIMPLE
tollM      = 1.0e-3
tollP      = 1.0e-3
tollC      = 1.0e-3
tollDNS    = 1.0e-7
alphaP     = 0.1e0      # pressure relaxation factor
alphaUV    = 0.1e0       # velocity relaxation factor
uRef       = 10.0e0

# learning rate for DNS field
lrDNS_POLICY = "EXPONENTIAL"   # "EXPONENTIAL" or "PIECEWISE"
lrDNS        = 1.0      # exponential policy initial learning rate
lrDNS_RATE   = 0.1        # exponential policy decay rate
lrDNS_STEP   = maxItDNS     # exponential policy decay step
lrDNS_EXP_ST = False      # exponential policy staircase
lrDNS_BOUNDS = [100, 200, 300]             # piecewise policy bounds
lrDNS_VALUES = [100.0, 50.0, 20.0, 10.0]   # piecewise policy values


