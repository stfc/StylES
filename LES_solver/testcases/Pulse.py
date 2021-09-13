import cupy as cp

from LES_constants import *



TEST_CASE = "Pulse"
PASSIVE   = True
RESTART   = False
totSteps  = 100
print_res = 10
print_img = 10
print_ckp = totSteps + 1
print_spe = totSteps + 1

rhoRef  = 1000.0e0       # density (water)    [kg/m3]
nuRef   = 1.0016e-5      # viscosity (water)  [Pa*s]
pRef    = 101325.0e0     # reference pressure (1 atm) [Pa]
Lx      = 0.01e0     # system dimension in x-direction   [m]
Ly      = 0.01e0     # system dimension in y-direction   [m]
Nx      = 51         # number of points in x-direction   [-]
Ny      = 51         # number of points in y-direction   [-]
dXY     = Lx/Nx
dXY     = Ly/Ny
CNum    = 0.5        # Courant number 
delt    = 1.e-3    # initial guess for delt
maxDelt = 1.e-3    # initial guess for delt
Uin     = 0.0             # inlet x-velocity. Set to -1 if not specified
Vin     = 0.0             # inlet y-velocity. Set to -1 if not specified  
Pin     = -1              # inlet pressure. Set to -1 if not specified
Pout    = -1              # outlet pressure. Set to -1 if not specified
dir     = 0               # cross direction for plotting results
puA     = 1.0e0           # amplitude pulse
puC     = Lx/2            # pulse center
puS     = 1.0e-3          # standard deviation
Umean   = 1.0e-4          # mean flow in x
Vmean   = 1.0e-4           # mean flow in y



def init_fields():

    U = cp.zeros([Nx,Ny], dtype=DTYPE)
    V = cp.zeros([Nx,Ny], dtype=DTYPE)
    P = cp.zeros([Nx,Ny], dtype=DTYPE)
    C = cp.zeros([Nx,Ny], dtype=DTYPE)
    B = cp.zeros([Nx,Ny], dtype=DTYPE)

    xyp = cp.linspace(hf*dXY, Lx-hf*dXY, Nx)
    X, Y = cp.meshgrid(xyp, xyp)

    U[:,:] = Umean
    V[:,:] = Vmean
    P[:,:] = pRef
    dist2 = (X-puC)**2 + (Y-puC)**2
    C = puA*cp.exp(-dist2/(2*puS**2))


    # set remaining fiels
    totTime = zero


    return U, V, P, C, B, totTime