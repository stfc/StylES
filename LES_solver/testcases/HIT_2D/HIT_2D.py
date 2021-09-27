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


TEST_CASE = "HIT_2D"
PASSIVE   = False
RESTART   = True
finalTime = 1.1547
totSteps  = 1
print_res = 1
print_img = totSteps+1
print_ckp = totSteps+1
print_spe = totSteps+1
N         = 256      # number of points   [-]
iNN       = one/(N*N)

pRef      = 1.0e0     # reference pressure (1 atm) [Pa]
rho       = 1.0e0          # density                    [kg/m3]
nu        = 1.87e-4        # dynamic viscosity          [Pa*s]  This should be found from Re. See excel file.
Re        = 60             # based on integral length l0 = sqrt(2*U^2/W^2) where W is the enstropy
M         = 5000           # number of modes
METHOD    = 0              # 0-In house, 1-Saad git repo, 2-OpenFOAM
L         = 0.95      # system dimension   [m]
dl        = L/N
CNum      = 0.5        # Courant number 
delt      = 1.0e-6    # initial guess for delt: 0.001072 is the eddy turnover time
maxDelt   = 1.0e-3
dir       = 1               # cross direction for plotting results



def init_fields():

    # set variables
    cp.random.seed(0)

    E = cp.zeros([M], dtype=DTYPE)  # enery spectrum
    k = cp.zeros([M], dtype=DTYPE)  # wave number

    # find max and min wave numbers
    k0   = two*pi/L     #same in each direction
    kmax = 600.0 #pi/(L/N)  #same in each direction

    # find k and E
    km = cp.linspace(k0, kmax, M)
    dk = (kmax-k0)/M
    inputspec = 'ld_spectrum'
    especf = getattr(spectra, inputspec)().evaluate
    km_cpu = cp.asnumpy(km)
    E_cpu = especf(km_cpu)
    E_cpu = cp.clip(E_cpu, zero)
    E = cp.asarray(E_cpu)
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
    #plt.xlim([0.0e0, 600])        
    #plt.ylim([1.0e-7, 0.2])
    plt.grid(True, which='both')
    plt.legend(('k^-3', 'k^-4', 'input'),  loc='upper right')
    plt.savefig("Energy_spectrum.png")

    #-------------- Start Saad procedure
    if (METHOD == 0): 

        # do everything on GPU
        U = cp.zeros([N,N], dtype=DTYPE)
        V = cp.zeros([N,N], dtype=DTYPE)
        P = cp.zeros([N,N], dtype=DTYPE)
        C = cp.zeros([N,N], dtype=DTYPE)
        B = cp.zeros([N,N], dtype=DTYPE)

        xyp  = cp.linspace(hf*dl, L-hf*dl, N)
        X, Y = cp.meshgrid(xyp, xyp)


        # find random angles
        nu      = cp.random.uniform(zero, one, M)
        thetaM  = cp.arccos(two*nu - one);
        psi     = cp.random.uniform(-pi*hf, pi*hf, M)

        # compute km
        kxm = sin(thetaM)
        kym = cos(thetaM)

        # compute kt
        kxt = two/dl*sin(hf*km*kxm*dl)
        kyt = two/dl*sin(hf*km*kym*dl)

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

            bmx = two * qm[kk] * cos(arg - kxm[kk] * hf*dl)
            bmy = two * qm[kk] * cos(arg - kym[kk] * hf*dl)

            U = U + bmx * sigmax[kk]
            V = V + bmy * sigmay[kk]


        # find spectrum
        U_cpu = cp.asnumpy(U)
        V_cpu = cp.asnumpy(V)

        knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)

        # print("StylES nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'yo-', linewidth=0.5, markersize=2)
        plt.legend(('k^-3', 'k^-4', 'input','t=0'),  loc='upper right')
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
        U_cpu = np.zeros([N,N], dtype=DTYPE)
        V_cpu = np.zeros([N,N], dtype=DTYPE)
        P_cpu = np.zeros([N,N], dtype=DTYPE)
        C_cpu = np.zeros([N,N], dtype=DTYPE)
        B_cpu = np.zeros([N,N], dtype=DTYPE)

        inputspec = 'dal_spectrum'
        whichspec = getattr(spectra, inputspec)().evaluate

        U_cpu, V_cpu = generate_isotropic_turbulence_2d(L, L, N, N, M, k0, whichspec)

        knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)

        # print("Saad nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'yo-', linewidth=0.5, markersize=2)
        plt.legend(('k^-3', 'k^-4', 'input','t=0'),  loc='upper right')
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

        U_cpu = np.zeros([N,N], dtype=DTYPE)
        V_cpu = np.zeros([N,N], dtype=DTYPE)
        P_cpu = np.zeros([N,N], dtype=DTYPE)
        C_cpu = np.zeros([N,N], dtype=DTYPE)
        B_cpu = np.zeros([N,N], dtype=DTYPE)

        list_files = ["2D_DNS_ReT60_N1024/0.155/U"]
        #list_files = ["0.000_U", "0.0342_U", "0.0912_U", "0.3686_U", "0.3724_U"]
        #list_files = ["0.032_256", "0.195_256", "0.792_256", "1.095_256"]
        #list_files = ["0.072_512", "0.195_512", "0.792_512", "1.095_512"]

        for c in list_files:
            #val = 0.003*(c+1)

            # read velocity field
            #sval = "{:.3f}".format(val)
            filename = "results/OpenFOAM/" + c
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

            while k<(N*N):
                line = line.replace('(','')
                line = line.replace(')','')

                U_cpu[N-j-1,i] = np.float64(line.split()[0])
                V_cpu[N-j-1,i] = np.float64(line.split()[1])

                newline = str(i) + "   " + str(j) + "    " + line

                k = k+1
                i = k % N
                j = int(k/N)
                line = fr.readline()

            fr.close()

            knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)

            #print("OpenFoam Nyquist limit is ", knyquist)
            plt.plot(wave_numbers, tke_spectrum, 'ro-', linewidth=0.5, markersize=2)
            plt.legend(('k^-3', 'k^-4', 'input','t=0'),  loc='upper right')
            plt.savefig("Energy_spectrum.png")

            filename = "Energy_spectrum_" + "134te.txt"
            np.savetxt(filename, np.c_[wave_numbers, tke_spectrum], fmt='%1.4e')   # use exponential notation


            # # read pressure field
            # filename = "results/OpenFOAM/2D_DNS_ReT60_N256/" + sval + "/p"
            # fr = open(filename, "r")
            # line = fr.readline()
            # while (("internalField" in line) == False):
            #         line = fr.readline()

            # line = fr.readline()   # read number of points
            # line = fr.readline()   # read (
            # line = fr.readline()   # read first triple u,v,w

            # i=0
            # j=0
            # k=0

            # while k<(N*N):
            #     line = line.replace('(','')
            #     line = line.replace(')','')

            #     P_cpu[N-j-1,i] = np.float64(line.split()[0])

            #     newline = str(i) + "   " + str(j) + "    " + line

            #     k = k+1
            #     i = k % N
            #     j = int(k/N)
            #     line = fr.readline()

            # fr.close()


        # set remaining fiels
        totTime_cpu = zero

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
    

