# define constants
zero  = 0.0e0
small = 1.0e-10
hf    = 0.5e0
one   = 1.0e0

# define runtime parameters
PATH      = "./"
DTYPE     = "float64"
maxIt     = 100
maxItPc   = 100
toll      = 1.0e-6    #tollerance for convergence of SIMPLE
tollPc    = 1.0e-6    #tollerance for convergence of Pressure correction
alphaP    = 0.1       # pressure relaxation factor
alphaUV   = 0.1       # velocity relaxation factor


# define case parameters
TEST_CASE = "Poiseuielle_x"
totSteps  = 101
delt      = 0.01
print_res = 10
print_img = 10
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

deltaX    = Lx/Nx          #                                   [m]
deltaY    = Ly/Ny          #                                   [m]
A         = deltaX*deltaY  # Area                              [m2] 
pRef      = 101325.0e0     # reference pressure (1 atm)
DX        = nuRef/deltaX   # diffusion conductance term in x
DY        = nuRef/deltaY   # diffusion conductance term in y
rA        = rhoRef*A
rX        = rhoRef*deltaX            
rY        = rhoRef*deltaY
rXX       = rhoRef*deltaX*deltaX
rYY       = rhoRef*deltaY*deltaY
