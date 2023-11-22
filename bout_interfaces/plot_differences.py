import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import sys

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect

sys.path.insert(0, '../')
sys.path.insert(0, '../LES_Solvers/')

from parameters import *
from LES_plot import *
from LES_functions import *

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../../codes/TurboGenPY/')

from tkespec import compute_tke_spectrum2d
from isoturb import generate_isotropic_turbulence_2d



#----------------------------- parameters
PATH_NETCDF = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/data_DNS/"
PATH_ANIMAT = "./results_comparison/plots/"
FIND_MIXMAX = True
DTYPE       = 'float32'
DIR         = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
L           = 50.176 
N_DNS2      = 2**(RES_LOG2-FIL)
delx        = L/N
dely        = L/N
listRUN     = ["DNS", 1, 2]
PATH_BOUTHW = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/"



#----------------------------- initiliaze
os.system("rm -rf results_comparison")
os.system("mkdir results_comparison")

cst = []
cst.append(0)

time_DNS    = []
time_StylES = []

Energy_DNS    = []
Energy_StylES = []

n_cDNS    = []
n_cStylES = []
p_cDNS    = []
p_cStylES = []
v_cDNS    = []
v_cStylES = []

n_tDNS    = []
n_tStylES = []
p_tDNS    = []
p_tStylES = []
v_tDNS    = []
v_tStylES = []


#----------------------------- functions
def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def save_fields(totTime, U, V, P, filename="restart.npz"):
    np.savez(filename, t=totTime, U=U, V=V, P=P)




#----------------------------- loop over DNS and StylES
cont_StylES = 0
for lrun in listRUN:
    
    if (lrun=='DNS'):
        MODE        = 'READ_NETCDF'
        STIME       = 0    # starting time to take as first image
        ITIME       = 1    # skip between STIME, FTIME, ITIME

        # find number of timesteps
        CWD = os.getcwd()
        os.chdir(PATH_NETCDF)
        n = collect("n",    xguards=False, info=False)
        FTIME = len(n)
        os.chdir(CWD)

        # find number of initial time
        # os.chdir(CWD)
        # data = np.load(FILE_DNS_fromGAN)
        # TGAP = np.cast[DTYPE](data['simtime'])
        TGAP = 0.0
        print("starting time ", TGAP)

        # find timestep
        file = open(PATH_NETCDF + "BOUT.inp", 'r')
        for line in file:
            if "timestep" in line:
                DELT = float(line.split()[2])
                print("timestep DNS", DELT)
                break
    else:
        tail        = str(lrun)
        MODE        = 'READ_NUMPY'
        PATH_FILES  = PATH_BOUTHW + "results_StylES_m" + tail +"/fields/"
        files       = os.listdir(PATH_FILES)
        STIME       = 0   # starting time to take as first image
        FTIME       = len(files)
        ITIME       = 1   # skip between time steps when reading NUMPY arrays

    #------------ read data
    if (lrun=='DNS'):
        print("reading " + PATH_NETCDF)

        os.chdir(PATH_NETCDF)
        
        n    = collect("n",    xguards=False, info=False)
        phi  = collect("phi",  xguards=False, info=False)
        vort = collect("vort", xguards=False, info=False)

        cont_DNS = 0
        for t in range(STIME,FTIME,ITIME):
            simtime = t*DELT + TGAP
            n_DNS   = n[t,:,0,:]
            p_DNS   = phi[t,:,0,:]
            v_DNS   = vort[t,:,0,:]
            
            n_cDNS.append(n_DNS[N_DNS2, N_DNS2])
            p_cDNS.append(p_DNS[N_DNS2, N_DNS2])
            v_cDNS.append(v_DNS[N_DNS2, N_DNS2])

            n_tDNS.append(n_DNS)
            p_tDNS.append(p_DNS)
            v_tDNS.append(v_DNS)
            
            gradV_p_DNS = np.sqrt(((cr(p_DNS, 1, 0) - cr(p_DNS, -1, 0))/(2.0*delx))**2 \
                + ((cr(p_DNS, 0, 1) - cr(p_DNS, 0, -1))/(2.0*dely))**2)

            time_DNS.append(simtime)
            E = 0.5*L**2*np.sum(n_DNS**2 + gradV_p_DNS**2)
            Energy_DNS.append(E)

            cont_DNS = cont_DNS+1

            print ("done for file time step ", str(t))
                        
        os.chdir("../../../../../StylES/bout_interfaces/")
        
    else:
        print("reading " + PATH_FILES)
        
        files = os.listdir(PATH_FILES)
        nfiles = len(files)
        for i,file in enumerate(sorted(files)):
            if (i%ITIME==0):
                filename = PATH_FILES + file
                data     = np.load(filename)
                simtime  = np.cast[DTYPE](data['simtime'])
                n_StylES = np.cast[DTYPE](data['U'])
                p_StylES = np.cast[DTYPE](data['V'])
                v_StylES = np.cast[DTYPE](data['P'])
                
                nval = n_StylES[N_DNS2, N_DNS2]
                pval = p_StylES[N_DNS2, N_DNS2]
                vval = v_StylES[N_DNS2, N_DNS2]
                
                n_cStylES.append(nval)
                p_cStylES.append(pval)
                v_cStylES.append(vval)

                n_tStylES.append(n_StylES)
                p_tStylES.append(p_StylES)
                v_tStylES.append(v_StylES)

                # find energy spectra
                gradV_p_StylES = np.sqrt(((cr(p_StylES, 1, 0) - cr(p_StylES, -1, 0))/(2.0*delx))**2 \
                    + ((cr(p_StylES, 0, 1) - cr(p_StylES, 0, -1))/(2.0*dely))**2)

                time_StylES.append(simtime)
                E = 0.5*L**2*np.sum(n_StylES**2 + gradV_p_StylES**2)
                Energy_StylES.append(E)
                
                # # find flux
                # flux = ((cr(gradV_p_StylES, 1, 0) - cr(p_StylES, -1, 0))/(2.0*delx))**2 \
                    
                #     cr(n x gradV_p_StylES
                
                # cflux.append(flux[N_DNS2, N_DNS2])
                
                cont_StylES = cont_StylES+1

                # print ("done for file " + filename + " at simtime " + str(simtime))
                
        cst.append(cont_StylES)
        print("number of total files: ", nfiles)



#--------------------------------------------------------  plots
cl = ['r', 'b', 'g', 'y']
ls = ['solid', 'dotted', 'dashed', 'dashdot']


# energy vs time
i = 0
for lrun in listRUN:
    if (lrun=='DNS'):
        label = r"DNS"
    elif (lrun==1):
        label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
    elif (lrun==2):
        label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
    elif (lrun==3):
        label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-3}$"
    elif (lrun==4):
        label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-4}$"

    if (lrun=='DNS'):
        plt.plot(time_DNS, Energy_DNS, color='k', linewidth=0.5, linestyle='solid', label=label)
    else:
        i1 = cst[i]
        i2 = cst[i+1]
        plt.plot(time_StylES[i1:i2], Energy_StylES[i1:i2], color=cl[i], linewidth=0.5, linestyle='dashed', label=label)
        i=i+1

plt.legend(fontsize="10", frameon=False)
plt.savefig('./results_comparison/energy_vs_time.png')
plt.close()
   
    
# fields
for nf in range(3):
    i = 0
    for lrun in listRUN:
        if (lrun=='DNS'):
            if (nf==0):
                plt.plot(time_DNS, n_cDNS, color='k', linewidth=0.5, linestyle='solid', label=r"$n$ DNS")
            elif (nf==1):
                plt.plot(time_DNS, p_cDNS, color='k', linewidth=0.5, linestyle='solid', label=r"$\phi$ DNS")
            else:
                plt.plot(time_DNS, v_cDNS, color='k', linewidth=0.5, linestyle='solid', label=r"$\omega$ DNS")
        else:
            tail = str(lrun)
            i1 = cst[i]
            i2 = cst[i+1]
            
            if (lrun==1):
                n_label = r"$n$ StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
                p_label = r"$\phi$ StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
                v_label = r"$\omega$ StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
            elif (lrun==2):
                n_label = r"$n$ StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
                p_label = r"$\phi$ StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
                v_label = r"$\omega$ StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
            elif (lrun==3):
                n_label = r"$n$ StylES  with $\epsilon_{REC}=$" + r"$10^{-3}$"
                p_label = r"$\phi$ StylES  with $\epsilon_{REC}=$" + r"$10^{-3}$"
                v_label = r"$\omega$ StylES  with $\epsilon_{REC}=$" + r"$10^{-3}$"
            elif (lrun==4):
                n_label = r"$n$ StylES  with $\epsilon_{REC}=$" + r"$10^{-4}$"
                p_label = r"$\phi$ StylES  with $\epsilon_{REC}=$" + r"$10^{-4}$"
                v_label = r"$\omega$ StylES with $\epsilon_{REC}=$" + r"$10^{-4}$"

            if (nf==0):                        
                plt.plot(time_StylES[i1:i2], n_cStylES[i1:i2], color=cl[i], linewidth=0.5, linestyle='dashed', label=n_label)
            elif (nf==1):
                plt.plot(time_StylES[i1:i2], p_cStylES[i1:i2], color=cl[i], linewidth=0.5, linestyle='dashed', label=p_label)
            else:
                plt.plot(time_StylES[i1:i2], v_cStylES[i1:i2], color=cl[i], linewidth=0.5, linestyle='dashed', label=v_label)

            i = i+1

    plt.legend(fontsize="10", frameon=False)
    plt.xlabel("time units [$\omega^{-1}_i$]")
    plt.ylabel("fields")
    if (nf==0):
        plt.savefig('./results_comparison/DNS_vs_StylES_n.png', dpi=200)
    elif (nf==1):
        plt.savefig('./results_comparison/DNS_vs_StylES_phi.png', dpi=200)
    elif (nf==2):
        plt.savefig('./results_comparison/DNS_vs_StylES_vort.png', dpi=200)
    plt.close()

exit(0)


# fields
colDNS = np.linspace(0,1,num=len(time_DNS))
for f in range(3):

    if (f==0):
        fDNS    = n_cDNS
        fStylES = n_cStylES
        cmap    = 'jet'
        label   = r"$n$"
    elif (f==1):
        fDNS    = p_cDNS
        fStylES = p_cStylES
        cmap    = 'jet'
        label   = r"$\phi$"
    elif (f==2):
        fDNS    = v_cDNS
        fStylES = v_cStylES
        cmap    = 'jet'
        label   = r"$\omega$"

    i = 0
    for lrun in listRUN:
            
        if (lrun=='DNS'):
            marker  = '^'
            plt.scatter(time_DNS, fDNS, c=colDNS, vmax=2, cmap=cmap, marker=marker, s=5, label=label + " DNS")
        else:
            tail = str(lrun)
            i1 = cst[i]
            i2 = cst[i]+cst[i+1]
            
            if (tail=="2"):
                marker  = 'o'
                n_label   = r"$n$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
                p_label   = r"$\phi$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
                v_label   = r"$\omega$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
                colStylES = np.linspace(0,0.5,num=len(time_StylES[i1:i2]))
                vmax      = 1.5
            else:
                marker  = '*'
                n_label   = r"$n$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
                p_label   = r"$\phi$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
                v_label   = r"$\omega$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
                colStylES = np.linspace(0,1,num=len(time_StylES[i1:i2]))
                vmax      = 1
            
            plt.scatter(time_StylES[i1:i2], fStylES[i1:i2], c=colStylES, vmax=vmax, cmap=cmap, marker=marker, s=5, label=label + " StylES")
                    
            i = i+1

    plt.legend(fontsize="10", loc ="lower left", frameon=False)
    plt.xlabel("time units [$\omega^{-1}_i$]")
    plt.ylabel(label)
    plt.savefig('./results_comparison/DNS_vs_StylES_UVP_' + str(f) + '.png', dpi=200)
    plt.close()






# MSE on images
sumcst = 0
cst.append(0)
for j in range(len(listRUN)-1):

    id = 0
    MSE_n = []
    MSE_p = []
    MSE_v = []
    for i in range (cst[j+1]):
        ii = sumcst + i
        if (time_StylES[ii]>=time_DNS[id]):
            MSE_n.append(np.mean((n_tDNS[id] - n_tStylES[ii])**2))
            MSE_p.append(np.mean((p_tDNS[id] - p_tStylES[ii])**2))
            MSE_v.append(np.mean((v_tDNS[id] - v_tStylES[ii])**2))

            id = id+1

    sumcst = sumcst + cst[j+1]

    if (j==0):
        # n_label = r"$n$ with $\epsilon_{REC}=0.01$"
        # p_label = r"$\phi$ with $\epsilon_{REC}=0.01$"
        # v_label = r"$\omega$ with $\epsilon_{REC}=0.01$"
        n_label = r"$n$"
        p_label = r"$\phi$"
        v_label = r"$\omega$"
    elif (j==1):
        n_label = r"$n$ with $\epsilon_{REC}=0.0075$"
        p_label = r"$\phi$ with $\epsilon_{REC}=0.0075$"
        v_label = r"$\omega$ with $\epsilon_{REC}=0.0075$"
    elif (j==2):
        n_label = r"$n$ with $\epsilon_{REC}=0.005$"
        p_label = r"$\phi$ with $\epsilon_{REC}=0.005$"
        v_label = r"$\omega$ with $\epsilon_{REC}=0.005$"
    
    if (j==0):
        plt.plot(time_DNS[0:id], MSE_n, color='k', linewidth=0.5, linestyle=ls[j], label=n_label)
        plt.plot(time_DNS[0:id], MSE_p, color='r', linewidth=0.5, linestyle=ls[j], label=p_label)
        plt.plot(time_DNS[0:id], MSE_v, color='b', linewidth=0.5, linestyle=ls[j], label=v_label)
    elif (j==1):
        plt.plot(time_DNS[0:id], MSE_n, color='k', linewidth=0.5, linestyle=ls[j])
        plt.plot(time_DNS[0:id], MSE_p, color='r', linewidth=0.5, linestyle=ls[j])
        plt.plot(time_DNS[0:id], MSE_v, color='b', linewidth=0.5, linestyle=ls[j])
    elif (j==2):
        plt.plot(time_DNS[0:id], MSE_n, color='k', linewidth=0.5, linestyle=ls[j], dashes=(5,10))
        plt.plot(time_DNS[0:id], MSE_p, color='r', linewidth=0.5, linestyle=ls[j], dashes=(5,10))
        plt.plot(time_DNS[0:id], MSE_v, color='b', linewidth=0.5, linestyle=ls[j], dashes=(5,10))
        

plt.legend(fontsize="10", loc ="upper left", frameon=False)
plt.xlabel("time units [$\omega^{-1}_i$]")
plt.ylabel("MSE")
plt.savefig('./results_comparison/MSE_images.png')
plt.close()

exit()




# # MSE on images
# cmap = 'jet'
# sumcst = 0
# cst.append(0)
# for j in range(len(listRUN)-1):

#     id = 0
#     MSE_n = []
#     MSE_p = []
#     MSE_v = []
#     for i in range (cst[j+1]):
#         ii = sumcst + i
#         if (time_StylES[ii]>=time_DNS[id]):
#             MSE_n.append(np.mean((n_tDNS[id] - n_tStylES[ii])**2))
#             MSE_p.append(np.mean((p_tDNS[id] - p_tStylES[ii])**2))
#             MSE_v.append(np.mean((v_tDNS[id] - v_tStylES[ii])**2))

#             id = id+1

#     sumcst = sumcst + cst[j+1]

#     if (j==0):
#         n_label   = r"$n$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
#         p_label   = r"$\phi$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
#         v_label   = r"$\omega$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
#         colStylES = np.linspace(0,0.5,num=len(MSE_n))
#         vmax      = 1.5        
#     else:
#         n_label   = r"$n$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#         p_label   = r"$\phi$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#         v_label   = r"$\omega$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#         colStylES = np.linspace(0,1,num=len(MSE_n))
#         vmax      = 1        

#     plt.scatter(time_DNS[0:id], MSE_n, c=colStylES, vmax=vmax, cmap=cmap, marker='^', s=5, label=n_label)
#     plt.scatter(time_DNS[0:id], MSE_p, c=colStylES, vmax=vmax, cmap=cmap, marker='o', s=5, label=p_label)
#     plt.scatter(time_DNS[0:id], MSE_v, c=colStylES, vmax=vmax, cmap=cmap, marker='*', s=5, label=v_label)
            
# plt.legend(fontsize="10", loc ="upper right", frameon=False)
# plt.xlabel("time units [$\omega^{-1}_i$]")
# plt.ylabel("MSE")
# plt.savefig('./results_comparison/MSE_images.png')
# plt.close()