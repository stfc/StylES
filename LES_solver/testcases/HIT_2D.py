import matplotlib.pyplot as plt

import cupy as cp

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

from LES_functions  import *




TEST_CASE = "2D_HIT"
PASSIVE   = False
RESTART   = False 
totSteps  = 1000
print_res = 10
print_img = 100
print_ckp = totSteps + 1
print_spe = print_img
Nx        = 256         # number of points in x-direction   [-]
Ny        = 256       # number of points in y-direction   [-]

pRef      = 1.0e0     # reference pressure (1 atm) [Pa]
rhoRef    = 1.0e0          # density                    [kg/m3]
nuRef     = 1.87e-4        # dynamic viscosity          [Pa*s]  This should be found from Re. See excel file.
Re        = 60             # based on integral length l0 = sqrt(2*U^2/W^2) where W is the enstropy
M         = 5000           # number of modes
kp        = 50.0e0*2*pi    # peak value
Q         = 3.4e-20      # constant to normalize energy spectrum. Adjust to have maximum E=0.059  
METHOD    = 0              # 0-In house, 1-Saad git repo, 2-OpenFOAM
Lx        = two*pi*0.145e0     # system dimension in x-direction   [m]
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
    k0   = two*pi/Lx     #same in each direction
    kmax = pi/(Lx/Nx)  #same in each direction

    # find k and E
    km = cp.linspace(k0, kmax, M)
    dk = (kmax-k0)/M
    E = Q*(km**8)*cp.exp(-4*(km/kp)**2)
    maxEm_cpu = cp.asnumpy(cp.max(E))
    print("Energy peak is ", maxEm_cpu)

    km_cpu = cp.asnumpy(km)
    E_cpu = cp.asnumpy(E)

    plt.plot(km_cpu, E_cpu, 'bo-', linewidth=0.5, markersize=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which='both')

    plt.savefig("Energy_spectrum.png")


    #-------------- Start Saad procedure
    if (METHOD == 0): 

        U = cp.zeros([Nx,Ny], dtype=DTYPE)
        V = cp.zeros([Nx,Ny], dtype=DTYPE)
        P = nc.zeros([Nx,Ny], dtype=DTYPE)
        C = nc.zeros([Nx,Ny], dtype=DTYPE)
        B = nc.zeros([Nx,Ny], dtype=DTYPE)

        xyp  = cp.linspace(hf*dXY, Lx-hf*dXY, Nx)
        X, Y = cp.meshgrid(xyp, xyp)


        # find random angles
        nu      = cp.random.uniform(0.0, 1.0, M)
        thetaM  = cp.arccos(2.0 * nu - 1.0);
        psi     = cp.random.uniform(-pi*hf, pi*hf, M)

        nu1     = cp.random.uniform(0.0, 1.0, M)
        thetaM1 = cp.arccos(2.0*nu1 - 1.0);

        # compute km
        kxm = sin(thetaM)
        kym = cos(thetaM)

        # compute kt
        kxt = two/dXY*sin(hf*km*kxm*dXY)
        kyt = two/dXY*sin(hf*km*kym*dXY)

        # compute the unit vectors
        sigmax =  sin(thetaM1)
        sigmay = -sigmax*kxt/kyt
        sigma = sqrt(sigmax*sigmax + sigmay*sigmay)
        sigmax =  sigmax/sigma
        sigmay =  sigmay/sigma

        # verify orthogonality
        #kk = cp.sum(kxt*sigmax + kyt*sigmay)
        #print(" Orthogonality is ",kk) 

        # find energy levels
        qm = sqrt(E*dk)

        # compute u, v and w
        for kk in range(M):
            arg = km[kk]*(kxm[kk]*X + kym[kk]*Y - psi[kk])

            bmx = 2.0 * qm[kk] * cos(arg - kxm[kk] * hf*dXY)
            bmy = 2.0 * qm[kk] * cos(arg - kym[kk] * hf*dXY)

            U = U + bmx * sigmax[kk]
            V = V + bmy * sigmay[kk]


        # find spectrum
        U_cpu = cp.asnumpy(U)
        V_cpu = cp.asnumpy(V)

        knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, Lx, Ly, True)

        print("StylES nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'yo-', linewidth=0.5, markersize=2)

        plt.savefig("Energy_spectrum.png")


    elif (METHOD == 1):          #-------------------- use Saad github implementation

        u, v, w = generate_isotropic_turbulence(Lx, Ly, Lz, Nx, Ny, Nz, M, k0, E)
        if (ThreeDim):
            knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum(u, v, w, Lx, Ly, Lz, True)
        else:
            knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(u[:,:,0], v[:,:,0], Lx, Ly, True)

        print("Saad nyquist limit is ", knyquist)
        plt.plot(wave_numbers, tke_spectrum, 'yo-', linewidth=0.5, markersize=2)

        plt.savefig("Energy_spectrum.png")



    elif (METHOD == 2):         #--------------------  read from OpenFOAM

        U_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        V_cpu = np.zeros([Nx,Ny], dtype=DTYPE)
        P = np.zeros([Nx,Ny], dtype=DTYPE)
        C = np.zeros([Nx,Ny], dtype=DTYPE)
        B = np.zeros([Nx,Ny], dtype=DTYPE)

        # read velocity field
        filename = "U_0.000"
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

            U_cpu[Nx-j-1,i] = float(line.split()[0])
            V_cpu[Nx-j-1,i] = float(line.split()[1])

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

        if (USE_CUPY):
            U = cp.asarray(U_cpu)
            V = cp.asarray(V_cpu)
        else:
            U = U_cpu
            V = V_cpu
    

    # set remaining fiels
    totTime = zero
    P[:,:] = pRef

    if (USE_CUPY):
        return U, V, P, C, B, totTime
    else:
        return U_cpu, V_cpu, P, C, B, totTime
