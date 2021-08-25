# define constants
zero  = 0.0e0
small = 1.0e-10
hf    = 0.5e0
one   = 1.0e0

# define runtime parameters
TEST_CASE = "Poiseuielle_x"
DTYPE     = "float64"
maxIt     = 10001
print_res = 10
save_img  = 1000
alphaP    = 0.5   # pressure relaxation factor
alphaU    = 0.5   # x-velocity relaxation factor
alphaV    = 0.5   # y-velocity relaxation factor
PATH      = "./"
toll      = 1.0e-6    #tollerance for convergence


# define physical parameters
Lx        = 0.32e0      # system dimension in x,y-direction [m]
rhoRef    = 1.225e0     # density (air)                     [kg/m3]
nuRef     = 1.81e-5     # viscosity (air)                   [Pa*s]

if (TEST_CASE == "Poiseuielle_x"):
    Nx = 38         # number of points in x-direction   [-]
    Ny = 18         # number of points in x-direction   [-]

if (TEST_CASE == "Poiseuielle_y"):
    Nx = 18         # number of points in x-direction   [-]
    Ny = 38         # number of points in x-direction   [-]

delta     = Lx/Nx       # we assume same delta in x,y       [m]
A         = delta       # Area                              [m2] 
pRef      = 101325.0e0  # reference pressure (1 atm)
D         = nuRef/delta # diffusion conductance term
rA        = rhoRef*A
