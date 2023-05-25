import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import sys

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect
from LES_plot import *
from LES_functions import *

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../../codes/TurboGenPY/')

from tkespec import compute_tke_spectrum2d
from isoturb import generate_isotropic_turbulence_2d



#----------------------------- initiliaze
PATH_ANIMAT   = "./analysis_comparison/plots/"
FIND_MIXMAX   = True
DTYPE         = 'float32'
DIR           = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
L             = 50.176 
N_DNS2        = 128
delx          = L/N
dely          = L/N

os.system("rm -rf analysis_comparison/plots/*")
os.system("rm -rf analysis_comparison/fields/*")
os.system("rm -rf analysis_comparison/energy/*")

os.system("mkdir -p analysis_comparison/plots/")
os.system("mkdir -p analysis_comparison/fields/")
os.system("mkdir -p analysis_comparison/energy/")

time_DNS         = []
time_StylES      = []

Energy_DNS       = []
Energy_StylES    = []

n_cDNS       = []
n_cStylES    = []
phi_cDNS     = []
phi_cStylES  = []
vort_cDNS    = []
vort_cStylES = []



#----------------------------- functions
def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def save_fields(totTime, U, V, P, filename="restart.npz"):
    np.savez(filename, t=totTime, U=U, V=V, P=P)




#----------------------------- loop over DNS and StylES
#listRUN = ["DNS", "StylES_m3"]
listRUN = ["DNS", "StylES_m1"]
for lrun in listRUN:
    
    if (lrun=='DNS'):
        MODE        = 'READ_NETCDF'
        PATH_NETCDF = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/data_DNS/"
        STIME       = 0    # starting time to take as first image
        FTIME       = 11   # final time from netCFD
        ITIME       = 1    # skip between STIME, FTIME, ITIME
        DELT        = 0.01  # delt equal to timestep in BOUT++ input file
    elif (lrun=='StylES_m1'):
        MODE        = 'READ_NUMPY'
        PATH_NUMPY  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/results_bout/fields/DNS/"
        files       = os.listdir(PATH_NUMPY)
        STIME       = 0    # starting time to take as first image
        FTIME       = len(files)
        ITIME       = 1   # cd - skip between time steps when reading NUMPY arrays
        DELT        = 1.0  # delt equal to timestep in BOUT++ input file
    elif (lrun=='StylES_m2'):
        MODE        = 'READ_NUMPY'
        PATH_NUMPY  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/results_StylES_10tu_tollm2/fields/DNS/"
        files       = os.listdir(PATH_NUMPY)
        STIME       = 0    # starting time to take as first image
        FTIME       = len(files)
        ITIME       = 1   # cd - skip between time steps when reading NUMPY arrays
        DELT        = 1.0  # delt equal to timestep in BOUT++ input file
    elif (lrun=='StylES_m3'):
        MODE        = 'READ_NUMPY'
        PATH_NUMPY  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/results_StylES_10tu_tollm3/fields/DNS/"
        files       = os.listdir(PATH_NUMPY)
        STIME       = 0    # starting time to take as first image
        FTIME       = len(files)
        ITIME       = 1   # cd - skip between time steps when reading NUMPY arrays
        DELT        = 1.0  # delt equal to timestep in BOUT++ input file
    elif (lrun=='StylES_m4'):
        MODE        = 'READ_NUMPY'
        PATH_NUMPY  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/results_StylES_10tu_tollm4/fields/DNS/"
        files       = os.listdir(PATH_NUMPY)
        STIME       = 0    # starting time to take as first image
        FTIME       = len(files)
        ITIME       = 1   # cd - skip between time steps when reading NUMPY arrays
        DELT        = 1.0  # delt equal to timestep in BOUT++ input file

    #------------ read data
    if (lrun=='DNS'):
        print("reading " + PATH_NETCDF)

        os.chdir(PATH_NETCDF)
        
        n    = collect("n",    xguards=False, info=False)
        phi  = collect("phi",  xguards=False, info=False)
        vort = collect("vort", xguards=False, info=False)

        cont_DNS = 0
        for t in range(STIME,FTIME,ITIME):
            simtime      = t*DELT
            n_DNS    = n[t,:,0,:]
            phi_DNS  = phi[t,:,0,:]
            vort_DNS = vort[t,:,0,:]
            
            n_cDNS.append(      n_DNS[N_DNS2, N_DNS2])
            phi_cDNS.append(  phi_DNS[N_DNS2, N_DNS2])
            vort_cDNS.append(vort_DNS[N_DNS2, N_DNS2])
            
            gradV_phi_DNS = np.sqrt(((cr(phi_DNS, 1, 0) - cr(phi_DNS, -1, 0))/(2.0*delx))**2 \
                + ((cr(phi_DNS, 0, 1) - cr(phi_DNS, 0, -1))/(2.0*dely))**2)

            time_DNS.append(simtime)
            E = 0.5*L**2*np.sum(n_DNS**2 + gradV_phi_DNS**2)
            Energy_DNS.append(E)

            cont_DNS = cont_DNS+1

            print ("done for file time step ", str(t))
                        
        os.chdir("../../../../../StylES/bout_interfaces/")
        
    elif ('StylES' in lrun):
        print("reading " + PATH_NUMPY)
        
        files = os.listdir(PATH_NUMPY)
        nfiles = len(files)
        cont_StylES = 0
        for i,file in enumerate(sorted(files)):
            if (i%ITIME==0):
                filename    = PATH_NUMPY + file
                data        = np.load(filename)
                simtime     = np.cast[DTYPE](data['simtime'])
                n_StylES    = np.cast[DTYPE](data['U'])
                phi_StylES  = np.cast[DTYPE](data['V'])
                vort_StylES = np.cast[DTYPE](data['P'])
                
                n_cStylES.append(      n_StylES[N_DNS2, N_DNS2])
                phi_cStylES.append(  phi_StylES[N_DNS2, N_DNS2])
                vort_cStylES.append(vort_StylES[N_DNS2, N_DNS2])
                            
                gradV_phi_StylES = np.sqrt(((cr(phi_StylES, 1, 0) - cr(phi_StylES, -1, 0))/(2.0*delx))**2 \
                    + ((cr(phi_StylES, 0, 1) - cr(phi_StylES, 0, -1))/(2.0*dely))**2)

                time_StylES.append(simtime)
                E = 0.5*L**2*np.sum(n_StylES**2 + gradV_phi_StylES**2)
                Energy_StylES.append(E)
                
                cont_StylES = cont_StylES+1

                print ("done for file " + filename + " at simtime " + str(simtime))
                


#-----------------------------  plot energy
filename="./analysis_comparison/DNS_vs_StylES_UVP.txt"
np.savez(filename, time_DNS=time_DNS, Energy_DNS=Energy_DNS, time_StylES=time_StylES, Energy_StylES=Energy_StylES)



#-----------------------------  plot fields
plt.plot(time_DNS,    n_cDNS,       color='k', linestyle='solid',   label=r"$n$ DNS")
plt.plot(time_DNS,    phi_cDNS,     color='r', linestyle='solid',   label=r"$\phi$ DNS")
plt.plot(time_DNS,    vort_cDNS,    color='b', linestyle='solid',   label=r"$\omega$ DNS")

cl = plt.cm.jet(np.linspace(0,1,3*len(listRUN)))
ls = ['dotted', 'dashed']
for i in range(len(listRUN)-1):
    st = cont_StylES
    plt.plot(time_StylES[i*st:(i+1)*st], n_cStylES[i*st:(i+1)*st],    color='k', linewidth=0.5, linestyle=ls[i],  label=r"$n$ StylES_m" + str(i+1))
    plt.plot(time_StylES[i*st:(i+1)*st], phi_cStylES[i*st:(i+1)*st],  color='r', linewidth=0.5, linestyle=ls[i],  label=r"$\phi$ StylES_m" + str(i+1))
    plt.plot(time_StylES[i*st:(i+1)*st], vort_cStylES[i*st:(i+1)*st], color='b', linewidth=0.5, linestyle=ls[i],  label=r"$\omega$ StylES_m" + str(i+1))

    filename="./analysis_comparison/DNS_vs_StylES_cUVP_m" + str(i+1)
    np.savez(filename, time_DNS=time_DNS, time_StylES=time_StylES, \
        n_cDNS=n_cDNS, phi_cDNS=phi_cDNS, vort_cDNS=vort_cDNS, \
        n_cStylES=n_cStylES[i*st:(i+1)*st], phi_cStylES=phi_cStylES[i*st:(i+1)*st], vort_cStylES=vort_cStylES[i*st:(i+1)*st])

# plt.legend()
plt.savefig('./analysis_comparison/DNS_vs_StylES_UVP.png', dpi=200)
plt.close()



#-----------------------------  plot SME
n_cStylES_int    = np.interp(time_DNS, time_StylES, n_cStylES)
phi_cStylES_int  = np.interp(time_DNS, time_StylES, phi_cStylES)
vort_cStylES_int = np.interp(time_DNS, time_StylES, vort_cStylES)

SME_n    = ((n_cDNS    - n_cStylES_int)**2)
SME_phi  = ((phi_cDNS  - phi_cStylES_int)**2)
SME_vort = ((vort_cDNS - vort_cStylES_int)**2)

plt.plot(time_DNS, SME_n)
plt.plot(time_DNS, SME_phi)
plt.plot(time_DNS, SME_vort)

plt.savefig('./analysis_comparison/SME_n.png')
plt.close()