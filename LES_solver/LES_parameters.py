# define constants
zero  = 0.0e0
small = 1.0e-10
hf    = 0.5e0
one   = 1.0e0

# define runtime parameters
TEST_CASE = "Poiseuielle_x"
DTYPE     = "float64"
totSteps  = 101
delt      = 1.0
maxIt     = 1000001
print_res = 10
print_img = 10
alphaP    = 1.0   # pressure relaxation factor
alphaU    = 1.0   # x-velocity relaxation factor
alphaV    = 1.0   # y-velocity relaxation factor
PATH      = "./"
toll      = 1.0e-6    #tollerance for convergence


# define physical parameters
rhoRef    = 1.225e0     # density (air)                     [kg/m3]
nuRef     = 1.81e-5     # viscosity (air)                   [Pa*s]

if (TEST_CASE == "Poiseuielle_x"):
    Lx = 0.38e0     # system dimension in x-direction   [m]
    Ly = 0.18e0     # system dimension in y-direction   [m]
    Nx = 38         # number of points in x-direction   [-]
    Ny = 18         # number of points in y-direction   [-]

if (TEST_CASE == "Poiseuielle_y"):
    Lx = 0.18e0     # system dimension in x-direction   [m]
    Ly = 0.38e0     # system dimension in y-direction   [m]
    Nx = 18         # number of points in x-direction   [-]
    Ny = 38         # number of points in y-direction   [-]

delta     = Lx/Nx       # we assume same delta in x,y       [m]
A         = delta       # Area                              [m2] 
pRef      = 101325.0e0  # reference pressure (1 atm)
D         = nuRef/delta # diffusion conductance term
rA        = rhoRef*A
