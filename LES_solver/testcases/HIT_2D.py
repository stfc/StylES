import matplotlib.pyplot as plt
import sys

import numpy as np
import cupy as cp

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

from LES_functions  import *


# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../TurboGenPY/')

import spectra


TEST_CASE = "2D_HIT"
PASSIVE   = False
RESTART   = False 
totSteps  = 100000
print_res = 10
print_img = 1000
print_ckp = totSteps + 1
print_spe = print_img
Nx        = 256         # number of points in x-direction   [-]
Ny        = 256       # number of points in y-direction   [-]

pRef      = 1.0e0     # reference pressure (1 atm) [Pa]
rhoRef    = 1.0e0          # density                    [kg/m3]
nuRef     = 1.87e-4        # dynamic viscosity          [Pa*s]  This should be found from Re. See excel file.
Re        = 60             # based on integral length l0 = sqrt(2*U^2/W^2) where W is the enstropy
M         = 5000           # number of modes
kp        = 50.0*2*pi    # peak value
Q         = 3.4e-20      # constant to normalize energy spectrum. Adjust to have maximum E=0.059  
METHOD    = 0              # 0-In house, 1-Saad git repo, 2-OpenFOAM
Lx        = two*pi*0.145e0    # system dimension in x-direction   [m]
Ly        = two*pi*0.145e0    # system dimension in y-direction   [m]
dXY       = Lx/Nx
CNum      = 0.5        # Courant number 
delt      = 3.0e-6 #dXY*0.001072    # initial guess for delt: 0.001072 is the eddy turnover time
maxDelt   = 3.0e-6 #dXY*0.001072
dir       = 1               # cross direction for plotting results



def init_fields():

    # set variables
    cp.random.seed(0)

    E = cp.zeros([M], dtype=DTYPE)  # enery spectrum
    k = cp.zeros([M], dtype=DTYPE)  # wave number

    # find max and min wave numbers
    k0   = 100.0 #two*pi/Lx     #same in each direction
    kmax = 600.0 #pi/(Lx/Nx)  #same in each direction

    # find k and E
    km = cp.linspace(k0, kmax, M)
    dk = (kmax-k0)/M
    E = Q*(km**8)*cp.exp(-4*(km/kp)**2)
    maxEm_cpu = cp.asnumpy(cp.max(E))
    print("Energy peak is ", maxEm_cpu)

    km_cpu = cp.asnumpy(km)
    E_cpu = cp.asnumpy(E)

    ykm3_cpu = 1.e3*km_cpu**(-3)
    plt.plot(km_cpu, ykm3_cpu, '-', linewidth=0.5, markersize=2)

    ykm4_cpu = 1.e5*km_cpu**(-4)
    plt.plot(km_cpu, ykm4_cpu, '-', linewidth=0.5, markersize=2)

    plt.plot(km_cpu, E_cpu, 'bo-', linewidth=0.5, markersize=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1.0e-7, 0.1])
    plt.grid(True, which='major')
    plt.legend(('k^-3', 'k^-4', 'input'),  loc='upper right')
    plt.savefig("Energy_spectrum.png")

    #-------------- Start Saad procedure
    if (METHOD == 0): 

        # do everything on GPU
        U = cp.zeros([Nx,Ny], dtype=DTYPE)
        V = cp.zeros([Nx,Ny], dtype=DTYPE)
        P = cp.zeros([Nx,Ny], dtype=DTYPE)
        C = cp.zeros([Nx,Ny], dtype=DTYPE)
        B = cp.zeros([Nx,Ny], dtype=DTYPE)

        xyp  = cp.linspace(hf*dXY, Lx-hf*dXY, Nx)
        X, Y = cp.meshgrid(xyp, xyp)


        # find random angles
        nu      = cp.random.uniform(zero, one, M)
        thetaM  = cp.arccos(two*nu - one);
        psi     = cp.random.uniform(-pi*hf, pi*hf, M)

        # compute km
        kxm = sin(thetaM)
        kym = cos(thetaM)

        # compute kt
        kxt = two/dXY*sin(hf*km*kxm*dXY)
        kyt = two/dXY*sin(hf*km*kym*dXY)

        # compute the unit vectors
        sigmax =-kyt
        sigmay = kxt
        sigma = sqrt(sigmax*sigmax + sigmay*sigmay)
        sigmax =  sigmax/sigma
        sigmay =  sigmay/sigma

        # verify orthogonality
        # kk = cp.sum(kxt*sigmax + kyt*sigmay)
        # print(" Orthogonality is ",kk) 

        # find energy levels
        qm = sqrt(E*dk)

        # compute u, v and w
        for kk in range(M):
            arg = km[kk]*(kxm[kk]*X + kym[kk]*Y - psi[kk])

            bmx = two * qm[kk] * cos(arg - kxm[kk] * hf*dXY)
            bmy = two * qm[kk] * cos(arg - kym[kk] * hf*dXY)

            U = U + bmx * sigmax[kk]
            V = V + bmy * sigmay[kk]


        # find spectrum
        U_cpu = cp.asnumpy(U)
        V_cpu = cp.asnumpy(V)

        knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, Lx, Ly, True)

        print("StylES nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'yo-', linewidth=0.5, markersize=2)

        plt.savefig("Energy_spectrum.png")

        # set remaining fiels
        totTime = zero
        P[:,:] = pRef

        # return fields
        if (USE_GPU):
            return U, V, P, C, B, totTime
        else:
            U_cpu = cp.asnumpy(U)
            V_cpu = cp.asnumpy(V)
            P_cpu = cp.asnumpy(P)
            C_cpu = cp.asnumpy(C)
            B_cpu = cp.asnumpy(B)
            totTime_cpu = cp.asnumpy(totTime)
            return U_cpu, V_cpu, P_cpu, C_cpu, B_cpu, totTime_cpu


    elif (METHOD == 1):          #-------------------- use Saad github implementation

        # do everthing on CPU
        U_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        V_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        P_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        C_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        B_cpu = np.zeros([Nx,Ny], dtype=DTYPE)

        inputspec = 'dal_spectrum'
        whichspec = getattr(spectra, inputspec)().evaluate

        U_cpu, V_cpu = generate_isotropic_turbulence_2d(Lx, Ly, Nx, Ny, M, k0, whichspec)

        knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, Lx, Ly, True)

        print("Saad nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'yo-', linewidth=0.5, markersize=2)

        plt.savefig("Energy_spectrum.png")

        # set remaining fiels
        totTime_cpu = zero
        P_cpu[:,:] = pRef

        # return fields
        if (USE_GPU):
            U = cp.asarray(U_cpu)
            V = cp.asarray(V_cpu)
            P = cp.asarray(P_cpu)
            C = cp.asarray(C_cpu)
            B = cp.asarray(B_cpu)
            totTime = cp.asarray(totTime_cpu)
            return U, V, P, C, B, totTime
        else:
            return U_cpu, V_cpu, P_cpu, C_cpu, B_cpu, totTime_cpu
    

    elif (METHOD == 2):         #--------------------  read from OpenFOAM

        U_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        V_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        P_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        C_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        B_cpu = np.zeros([Nx,Ny], dtype=DTYPE)

        # read velocity field
        filename = "results/OpenFOAM/U_0.000"
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

            U_cpu[Nx-j-1,i] = np.float64(line.split()[0])
            V_cpu[Nx-j-1,i] = np.float64(line.split()[1])

            newline = str(i) + "   " + str(j) + "    " + line

            k = k+1
            i = k % Nx
            j = int(k/Nx)
            line = fr.readline()

        fr.close()

        knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, Lx, Ly, True)

        print("OpenFoam Nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'ro-', linewidth=0.5, markersize=2)

        plt.savefig("Energy_spectrum.png")

        # set remaining fiels
        totTime_cpu = zero
        P_cpu[:,:] = pRef

        # return fields
        if (USE_GPU):
            U = cp.asarray(U_cpu)
            V = cp.asarray(V_cpu)
            P = cp.asarray(P_cpu)
            C = cp.asarray(C_cpu)
            B = cp.asarray(B_cpu)
            totTime = cp.asarray(totTime_cpu)
            return U, V, P, C, B, totTime
        else:
            return U_cpu, V_cpu, P_cpu, C_cpu, B_cpu, totTime_cpu
    

