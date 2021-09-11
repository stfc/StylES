import numpy as np
import matplotlib.pyplot as plt
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../TurboGenPY/')

from numpy import pi, sqrt, sin, cos, pi
from tkespec import *
from cudaturbo import *

from LES_constants import *



TEST_CASE = "2D_HIT"
PASSIVE   = False
totSteps  = 1000
print_res = 10
print_img = 10

pRef      = 101325.0e0     # reference pressure (1 atm) [Pa]
Lx        = two*pi*0.145e0     # system dimension in x-direction   [m]
Ly        = two*pi*0.145e0    # system dimension in y-direction   [m]
Nx        = 16         # number of points in x-direction   [-]
Ny        = 16        # number of points in y-direction   [-]
deltaX    = Lx/Nx
deltaY    = Ly/Ny
CNum      = 0.5        # Courant number 
delt      = 1.e-3 #0.001*deltaX*0.001072
maxDelt   = 1.e-3 #0.001*deltaX*0.001072    # initial guess for delt: 0.001072 is the eddy turnover time
BCs       = [0, 0, 0, 0]    # Boundary conditions: W,E,S,N   0-periodic, 1-wall, 2-fixed inlet velocity
dir       = 1               # cross direction for plotting results

Re        = 60             # based on integral length l0 = sqrt(2*U^2/W^2) where W is the enstropy
rhoRef    = 1.0e0          # density                    [kg/m3]
nuRef     = 1.87e-4        # dynamic viscosity          [Pa*s]  This should be found from Re. See excel file.
M         = 5000           # number of modes
kp        = 50.0e0*2*pi    # peak value
Q         = 3.4e-20      # constant to normalize energy spectrum. Adjust to have maximum E=0.059  
METHOD    = 0              # 0-In house, 1-Saad git repo, 2-OpenFOAM
ThreeDim  = False


def init_flow(U, V, P, C):

    # set variables
    np.random.seed(0)

    if (ThreeDim):
        Lz = Lx
        Nz = Nx
        deltaZ = Lz/Nz
        u = np.zeros([Nx,Ny,Nz], dtype=DTYPE)  # enery spectrum
        v = np.zeros([Nx,Ny,Nz], dtype=DTYPE)  # enery spectrum
        w = np.zeros([Nx,Ny,Nz], dtype=DTYPE)  # enery spectrum
    else:
        u = np.zeros([Nx,Ny], dtype=DTYPE)  # enery spectrum
        v = np.zeros([Nx,Ny], dtype=DTYPE)  # enery spectrum

    E = np.zeros([M], dtype=DTYPE)  # enery spectrum
    k = np.zeros([M], dtype=DTYPE)  # wave number
    P = pRef
    C = zero

    # find max and min wave numbers
    k0   = two*pi/Lx     #same in each direction
    kmax = pi/(Lx/Nx)  #same in each direction

    # find k and E
    maxEm = zero
    km = np.linspace(k0, kmax, M)
    dk = (kmax-k0)/M
    E = Q*(km**8)*np.exp(-4*(km/kp)**2)
    maxEm = np.max(E)
    print("Energy peak is ", maxEm)

    plt.plot(km, E, 'bo-', linewidth=0.5, markersize=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which='both')

    plt.savefig("Energy_spectrum.png")


    #-------------- Start Saad procedure
    if (METHOD == 0): 

        if (ThreeDim):
 
            # find random angles
            nu     = np.random.uniform(0.0, 1.0, M)
            thetaM = np.arccos(2.0 * nu - 1.0);
            phiM   = 2.0*pi*np.random.uniform(0.0, 1.0, M)
            psi    = np.random.uniform(-pi*hf, pi*hf, M)

            nu1     = np.random.uniform()
            thetaM1 = np.arccos(2.0*nu1 - 1.0);
            phiM1   = 2.0*pi*np.random.uniform()

            # compute km 
            kxm = sin(thetaM)*cos(phiM)
            kym = sin(thetaM)*sin(phiM)
            kzm = cos(thetaM)

            # compute kt
            kxt = two/deltaX*sin(hf*km*kxm*deltaX)
            kyt = two/deltaY*sin(hf*km*kym*deltaY)
            kzt = two/deltaZ*sin(hf*km*kzm*deltaZ)

            # compute intermediate vector
            zetax = sin(thetaM1)*cos(phiM1)
            zetay = zetax*kyt/kxt
            zetaz = cos(thetaM1)

            # compute the unit vectors
            sigmax =  zetay*kzt - zetaz*kyt
            sigmay = -(zetax*kzt - zetaz*kxt)
            sigmaz =  zetax*kyt - zetay*kxt
            sigma = sqrt(sigmax*sigmax + sigmay*sigmay + sigmaz*sigmaz)
            sigmax =  sigmax/sigma
            sigmay =  sigmay/sigma
            sigmaz =  sigmaz/sigma

            # verify orthogonality
            #kk = np.sum(kxt*sigmax + kyt*sigmay + kzt*sigmaz)
            #print(" Orthogonality ok and sigma ",kk, sigmaz) 

            # find energy levels
            qm = sqrt(E*dk)

            # loop over all points
            for k in range(Nz):
                for j in range(Ny):
                    for i in range(Nx):

                        # compute u, v and w
                        x = hf*deltaX + i*deltaX
                        y = hf*deltaY + j*deltaY
                        z = hf*deltaZ + k*deltaZ

                        arg = km*(kxm*x + kym*y + kzm*z - psi)

                        bmx = 2.0 * qm * cos(arg - kxm * hf*deltaX)
                        bmy = 2.0 * qm * cos(arg - kym * hf*deltaY)
                        bmz = 2.0 * qm * cos(arg - kzm * hf*deltaZ)

                        u[i, j, k] = np.sum(bmx * sigmax)
                        v[i, j, k] = np.sum(bmy * sigmay)
                        w[i, j, k] = np.sum(bmz * sigmaz)

            # find spectrum
            knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum(u, v, w, Lx, Ly, Lz, True)

        else:

            # find random angles
            nu      = np.random.uniform(0.0, 1.0, M)
            thetaM  = np.arccos(2.0 * nu - 1.0);
            phiM    = 2.0*pi*np.random.uniform(0.0, 1.0, M)
            psi     = np.random.uniform(-pi*hf, pi*hf, M)

            nu1     = np.random.uniform(0.0, 1.0, M)
            thetaM1 = np.arccos(2.0*nu1 - 1.0);

            # compute km
            kxm = sin(thetaM)
            kym = cos(thetaM)

            # compute kt
            kxt = two/deltaX*sin(hf*km*kxm*deltaX)
            kyt = two/deltaY*sin(hf*km*kym*deltaY)

            # compute the unit vectors
            sigmax =  sin(thetaM1)
            sigmay = -sigmax*kxt/kyt
            sigma = sqrt(sigmax*sigmax + sigmay*sigmay)
            sigmax =  sigmax/sigma
            sigmay =  sigmay/sigma

            # verify orthogonality
            #kk = np.sum(kxt*sigmax + kyt*sigmay)
            #print(" Orthogonality is ",kk) 

            # find energy levels
            qm = sqrt(E*dk)

            # loop over all points
            for j in range(Ny):
                for i in range(Nx):

                    # compute u, v and w
                    x = hf*deltaX + i*deltaX
                    y = hf*deltaY + j*deltaY
 
                    arg = km*(kxm*x + kym*y - psi)

                    bmx = 2.0 * qm * cos(arg - kxm * hf*deltaX)
                    bmy = 2.0 * qm * cos(arg - kym * hf*deltaY)

                    u[i, j] = np.sum(bmx * sigmax)
                    v[i, j] = np.sum(bmy * sigmay)


        # find spectrum
        knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(u, v, Lx, Ly, True)

        print("StylES nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'yo-', linewidth=0.5, markersize=2)

        plt.savefig("Energy_spectrum.png")

    elif (METHOD == 1): 

        #-------------------- use Saad github implementation
        u, v, w = generate_isotropic_turbulence(Lx, Ly, Lz, Nx, Ny, Nz, M, k0, E)
        if (ThreeDim):
            knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum(u, v, w, Lx, Ly, Lz, True)
        else:
            knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(u[:,:,0], v[:,:,0], Lx, Ly, True)

        print("Saad nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'yo-', linewidth=0.5, markersize=2)

        plt.savefig("Energy_spectrum.png")

    elif (METHOD == 2): 

        #--------------------  read from OpenFOAM
        if (ThreeDim):

            pass

        else:

            # read velocity field
            filename = "0.001_U"
            fr = open(filename, "r")
            line = fr.readline()
            while (("internalField" in line) == False):
                    line = fr.readline()

            line = fr.readline()   # read number of points
            line = fr.readline()   # read (
            line = fr.readline()   # read first triple u,v,w

            i=0
            j=0
            k=0

            while k<(Nx*Ny):
                line = line.replace('(','')
                line = line.replace(')','')

                u[Nx-j-1,i] = float(line.split()[0])
                v[Nx-j-1,i] = float(line.split()[1])

                newline = str(i) + "   " + str(j) + "    " + line

                k = k+1
                i = k % Nx
                j = int(k/Nx)
                line = fr.readline()

            fr.close()

            knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(u, v, Lx, Ly, True)

        print("OpenFoam Nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'ro-', linewidth=0.5, markersize=2)

        plt.savefig("Energy_spectrum.png")

    for i in range(Nx):
        for j in range(Ny):
            U[i,j] = u[i,j]
            V[i,j] = v[i,j]