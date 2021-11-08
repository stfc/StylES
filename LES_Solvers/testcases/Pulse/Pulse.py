import matplotlib.pyplot as plt

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

from LES_functions  import *



TEST_CASE = "Pulse"
PASSIVE   = True
RESTART   = False
totSteps  = 100
print_res = 10
print_img = 10
print_ckp = totSteps + 1
print_spe = totSteps + 1

rho     = 1000.0e0       # density (water)    [kg/m3]
nu      = zero       # viscosity (water)  [Pa*s]
pRef    = 101325.0e0     # reference pressure (1 atm) [Pa]
L       = 1.0e-2     # system dimension in x-direction   [m]
N       = 41         # number of points in x-direction   [-]
iNN     = one/(N*N)
dl      = L/N
CNum    = 0.5e-0    # Courant number 
delt    = 1.0e-2    # initial guess for delt
maxDelt = 1.0e-2    # initial guess for delt
dir     = 0               # cross direction for plotting results
puA     = one           # amplitude pulse
puC     = L/two            # pulse center
puS     = 1.0e-3          # standard deviation
Umean   = zero             # mean flow in x
Vmean   = 1.0e-2           # mean flow in y



def init_fields():

    U = nc.zeros([N,N], dtype=DTYPE)
    V = nc.zeros([N,N], dtype=DTYPE)
    P = nc.zeros([N,N], dtype=DTYPE)
    C = nc.zeros([N,N], dtype=DTYPE)
    B = nc.zeros([N,N], dtype=DTYPE)

    xyp = nc.linspace(hf*dl, L-hf*dl, N)
    X, Y = nc.meshgrid(xyp, xyp)

    U[:,:] = Umean
    V[:,:] = Vmean
    P[:,:] = pRef
    dist2 = (X-puC)**2 + (Y-puC)**2
    C = puA*nc.exp(-dist2/(2*puS**2))


    # set remaining fiels
    totTime = zero


    return U, V, P, C, B, totTime