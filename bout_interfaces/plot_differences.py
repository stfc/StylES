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

from tkespec import compute_tke_spectrum2d_3v
from isoturb import generate_isotropic_turbulence_2d



#----------------------------- parameters
PATH_NETCDF  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/data_DNS/"
PATH_ANIMAT  = "./results_comparison/plots/"
FIND_MIXMAX  = True
DTYPE        = 'float32'
DIR          = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
L            = 50.176 
N1           = N_DNS-1
listRUN      = ["DNS",1]
ITIME_DNS    = 1
ITIME_StylES = 1
PATH_BOUTHW  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/"
FIND_DIFFS   = True


#----------------------------- initiliaze
os.system("rm -rf results_comparison")
os.system("mkdir results_comparison")
if (FIND_DIFFS):
    os.system("rm -rf results_comparison/plot_diffs")
    os.system("mkdir results_comparison/plot_diffs")
    
cst = []
cst.append(0)

time_DNS    = []
time_StylES = []

dVdx_DNS         = []
dVdy_DNS         = []
dVdx_StylES      = []
dVdy_StylES      = []
Energy_DNS       = []
Energy_StylES    = []
rflux_DNS        = []
pflux_DNS        = []
rflux_StylES     = []
pflux_StylES     = []
enstrophy_DNS    = []
enstrophy_StylES = []

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

cl = ['r', 'b', 'g', 'y']
ls = ['solid', 'dotted', 'dashed', 'dashdot']

# to use mathcal:
plt.rcParams['mathtext.default'] = 'regular'
#..or
# ax1 = plt.gca()
# ax1.draw = wrap_rcparams(ax1.draw, {"mathtext.default":'regular'})

plt.matplotlib.rcParams.update({'font.size': 16})
plt.matplotlib.rcParams.update({'figure.autolayout': True})

#----------------------------- functions
def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def save_fields(totTime, U, V, P, filename="restart.npz"):
    np.savez(filename, t=totTime, U=U, V=V, P=P)

def wrap_rcparams(f, params):
    def _f(*args, **kw):
        backup = {key:plt.rcParams[key] for key in params}
        plt.rcParams.update(params)
        f(*args, **kw)
        plt.rcParams.update(backup)
    return _f




#----------------------------- loop over DNS and StylES
cont_StylES = 0
for lrun in listRUN:
    
    if (lrun=='DNS'):
        MODE        = 'READ_NETCDF'
        STIME       = 0    # starting time to take as first image
        ITIME       = ITIME_DNS    # skip between STIME, FTIME, ITIME

        # find number of timesteps
        CWD = os.getcwd()
        os.chdir(PATH_NETCDF)
        n = collect("n",    xguards=False, info=False)
        FTIME = int(len(n)/ITIME_DNS)
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
        FTIME       = int(len(files)/ITIME_StylES)
        ITIME       = ITIME_StylES   # skip between time steps when reading NUMPY arrays

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
            
            # find energy spectra
            dVdx = (-cr(p_DNS, 2, 0) + 8*cr(p_DNS, 1, 0) - 8*cr(p_DNS, -1,  0) + cr(p_DNS, -2,  0))/(12.0*DELX_LES)
            dVdy = (-cr(p_DNS, 0, 2) + 8*cr(p_DNS, 0, 1) - 8*cr(p_DNS,  0, -1) + cr(p_DNS,  0, -2))/(12.0*DELY_LES)
            E = 0.5*L**2*np.sum(n_DNS**2 + dVdx**2 + dVdy**2)*DELX*DELY
            dVdx_DNS.append(dVdx)
            dVdy_DNS.append(dVdy)
            Energy_DNS.append(E)

            # find enstrophy
            e_DNS = 0.5*np.sum((n_DNS - v_DNS)**2)*DELX*DELY
            enstrophy_DNS.append(e_DNS)
                
            # find flux
            Vx_DNS = -((cr(p_DNS, 0, 1) - cr(p_DNS, 0,-1))/(2.0*DELY))  # vx = -dpdy
            Vy_DNS =  ((cr(p_DNS, 1, 0) - cr(p_DNS,-1, 0))/(2.0*DELX))  # vy =  dpdx
            rflux = np.sum(n_DNS[N1,:]*Vx_DNS[N1,:])/L
            pflux = np.sum(n_DNS[:,N1]*Vy_DNS[:,N1])/L
            rflux_DNS.append(rflux)
            pflux_DNS.append(pflux)

            # find time
            time_DNS.append(simtime)

            cont_DNS = cont_DNS+1

            # print ("done for file time step ", str(t))
                        
        os.chdir("../../../../../StylES/bout_interfaces/")
        print("number of total files: ", cont_DNS)

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
                dVdx = (-cr(p_StylES, 2, 0) + 8*cr(p_StylES, 1, 0) - 8*cr(p_StylES, -1,  0) + cr(p_StylES, -2,  0))/(12.0*DELX_LES)
                dVdy = (-cr(p_StylES, 0, 2) + 8*cr(p_StylES, 0, 1) - 8*cr(p_StylES,  0, -1) + cr(p_StylES,  0, -2))/(12.0*DELY_LES)
                E = 0.5*L**2*np.sum(n_StylES**2 + dVdx**2 + dVdy**2)*DELX*DELY
                dVdx_StylES.append(dVdx)
                dVdy_StylES.append(dVdy)
                Energy_StylES.append(E)

                # find enstrophy
                e_StylES = 0.5*np.sum((n_StylES - v_StylES)**2)*DELX*DELY
                enstrophy_StylES.append(e_StylES)

                # find flux
                Vx_StylES = -((cr(p_StylES, 0, 1) - cr(p_StylES, 0,-1))/(2.0*DELY))  # vx = -dpdy
                Vy_StylES =  ((cr(p_StylES, 1, 0) - cr(p_StylES,-1, 0))/(2.0*DELX))  # vy =  dpdx
                rflux = np.sum(n_StylES[N1,:]*Vx_StylES[N1,:])/L
                pflux = np.sum(n_StylES[:,N1]*Vy_StylES[:,N1])/L

                rflux_StylES.append(rflux)
                pflux_StylES.append(pflux)

                # find time
                time_StylES.append(simtime)
                
                # print ("done for file " + filename + " at simtime " + str(simtime))
                if (FIND_DIFFS and lrun==listRUN[-1]):
                    filename = "./results_comparison/plot_diffs/diff_vort_" + str(i).zfill(4) + ".png"
                    print("plotting diffrences for" + filename)
                    ii = cont_StylES%FTIME
                    print_fields_3(v_tDNS[ii], v_StylES, v_tDNS[ii]-v_StylES, filename=filename, plot='diff', \
                        labels=[r"DNS $\zeta$", r"StylES $\zeta$", r"diff $\zeta$"], \
                        Umin=-INIT_SCA, Umax=INIT_SCA, Vmin=-INIT_SCA, Vmax=INIT_SCA, Pmin=-INIT_SCA, Pmax=INIT_SCA)

                cont_StylES = cont_StylES+1
                
        cst.append(cont_StylES)
        print("number of total files: ", nfiles)

if (FIND_DIFFS):
    anim_file = './results_comparison/plot_diffs/animation_diff_vort.gif'
    filenames = glob.glob("./results_comparison/plot_diffs/*.png")
    filenames = sorted(filenames)

    with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
        for filename in filenames:
            print(filename)
            image = imageio.v2.imread(filename)
            writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)

    import tensorflow_docs.vis.embed as embed
    embed.embed_file(anim_file)


#--------------------------------------------------------  plots

print("plot comparison")
# full fields
kd = 0
k = cst[len(listRUN)-2]

minv = np.min(n_tDNS[kd])
maxv = np.max(n_tDNS[kd])
filename = "./results_comparison/ndiff_t0.png"
labels = [r'$n_{DNS}$', r'$n_{StylES}$', r'$n_{DNS}$ - $n_{StylES}$']
print_fields_3new(n_tDNS[kd], n_tStylES[k], n_tDNS[kd]-n_tStylES[k], filename=filename, diff=True, labels=labels, \
    Umin=minv, Umax=maxv, Vmin=minv, Vmax=maxv, Pmin=minv, Pmax=maxv)

minv = np.min(p_tDNS[kd])
maxv = np.max(p_tDNS[kd])
filename = "./results_comparison/pdiff_t0.png"
labels = [r'$\phi_{DNS}$', r'$\phi_{StylES}$', r'$\phi_{DNS}$ - $\phi_{StylES}$']
print_fields_3new(p_tDNS[kd], p_tStylES[k], p_tDNS[kd]-p_tStylES[k], filename=filename, diff=True, labels=labels, \
    Umin=minv, Umax=maxv, Vmin=minv, Vmax=maxv, Pmin=minv, Pmax=maxv)

minv = np.min(v_tDNS[kd])
maxv = np.max(v_tDNS[kd])
filename = "./results_comparison/vdiff_t0.png"
labels = [r'$\zeta_{DNS}$', r'$\zeta_{StylES}$', r'$\zeta_{DNS}$ - $\zeta_{StylES}$']
print_fields_3new(v_tDNS[kd], v_tStylES[k], v_tDNS[kd]-v_tStylES[k], filename=filename, diff=True, labels=labels, \
    Umin=minv, Umax=maxv, Vmin=minv, Vmax=maxv, Pmin=minv, Pmax=maxv)



# spectra
print("plot spectra")

t = FTIME-1
listtk = [(kd, cst[len(listRUN)-2]),(FTIME-1, cst[len(listRUN)-1]-1)]

# verify DNS and StylES have same amount of data
for t,k in listtk:
    _, wave_numbers, tke_spectrum = compute_tke_spectrum2d_3v(n_tDNS[t], dVdx_DNS[t], dVdy_DNS[t], L, L, L, True)
    plt.plot(wave_numbers, tke_spectrum, label='DNS at t=' + str(int(time_DNS[t])))
    _, wave_numbers, tke_spectrum = compute_tke_spectrum2d_3v(n_tStylES[k], dVdx_StylES[k], dVdy_StylES[k], L, L, L, True)
    plt.plot(wave_numbers, tke_spectrum, label='StylES at t=' + str(int(time_DNS[t])))

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r'k [$\rho_i^{-1}$]')
    plt.ylabel(r'$\mathcal{F}(E)$')
    plt.legend(frameon=False)
    plt.savefig("./results_comparison/energy_t" + str(t) + ".png", dpi=300)
    plt.close()

    # _, wave_numbers, tke_spectrum = compute_tke_spectrum2d_3v((n_tDNS[t]-v_DNS[t]), (n_tDNS[t]-v_DNS[t]), L, L, L, True)
    # plt.plot(wave_numbers, tke_spectrum, label='DNS at t=' + str(t))
    # _, wave_numbers, tke_spectrum = compute_tke_spectrum2d_3v((n_tStylES[k]-v_tStylES[k]), (n_tStylES[k]-v_tStylES[k]), L, L, L, True)
    # plt.plot(wave_numbers, tke_spectrum, label='StylES at t=' + str(t))

    # plt.yscale("log")
    # plt.xscale("log")
    # plt.xlabel(r'k [$\rho_i^{-1}$]')
    # plt.ylabel(r'$\mathcal{F}(E)$')
    # plt.legend(frameon=False)
    # plt.savefig("./results_comparison/enstrophy_t" + str(t) + ".png", dpi=300)
    # plt.close()



# energy vs time
print("plot evergy vs time")
i = 0
for lrun in listRUN:
    if (lrun=='DNS'):
        label = r"DNS"
    elif (lrun==1):
        #label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
        label = r"StylES with $\epsilon_{REC}$=1"
    elif (lrun==2):
        #label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
        label = r"StylES with $\epsilon_{REC}$=0.5"
    elif (lrun==3):
        # label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-3}$"
        label = r"StylES with $\epsilon_{REC}$=0.25"
    elif (lrun==4):
        # label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-4}$"
        label = r"StylES with $\epsilon_{REC}$=0.125"

    if (lrun=='DNS'):
        plt.plot(time_DNS, Energy_DNS, color='k', linewidth=0.5, linestyle='solid', label=label)
    else:
        i1 = cst[i]
        i2 = cst[i+1]
        plt.plot(time_StylES[i1:i2], Energy_StylES[i1:i2], color=cl[i], linewidth=0.5, linestyle='dashed', label=label)
        i=i+1

        np.savez("./results_comparison/energy_vs_time", tD=time_DNS, eD=Energy_DNS, tS=time_StylES[i1:i2], eS=Energy_StylES[i1:i2])

#plt.ylim(1e6,1e8)
#plt.xlim(0,10)
#plt.yscale("log")
#plt.xlabel("time steps [-]")
plt.xlabel("time [$\omega_{ci}^{-1}$]")
plt.ylabel("energy")
plt.legend(fontsize="10", frameon=False)
plt.savefig('./results_comparison/energy_vs_time.png', dpi=300)
plt.close()



# enstrophy vs time
print("plot enstrophy vs time")

i = 0
for lrun in listRUN:
    if (lrun=='DNS'):
        label = r"DNS"
    elif (lrun==1):
        #label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
        label = r"StylES with $\epsilon_{REC}$=1"
    elif (lrun==2):
        #label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
        label = r"StylES with $\epsilon_{REC}$=0.5"
    elif (lrun==3):
        # label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-3}$"
        label = r"StylES with $\epsilon_{REC}$=0.25"
    elif (lrun==4):
#        label = r"StylES with $\epsilon_{REC}=$" + r"$10^{-4}$"
        label = r"StylES with $\epsilon_{REC}$=0.125"

    if (lrun=='DNS'):
        plt.plot(time_DNS, enstrophy_DNS, color='k', linewidth=0.5, linestyle='solid', label=label)
    else:
        i1 = cst[i]
        i2 = cst[i+1]
        plt.plot(time_StylES[i1:i2], enstrophy_StylES[i1:i2], color=cl[i], linewidth=0.5, linestyle='dashed', label=label)
        i=i+1

        np.savez("./results_comparison/enstrophy_vs_time", tD=time_DNS, eD=enstrophy_DNS, tS=time_StylES[i1:i2], eS=enstrophy_StylES[i1:i2])
        
#plt.ylim(1e3,1e5)
#plt.xlim(0,10)
#plt.yscale("log")
plt.xlabel("time [$\omega_{ci}^{-1}$]")
plt.ylabel("enstrophy")
plt.legend(fontsize="10", frameon=False)
plt.savefig('./results_comparison/enstrophy_vs_time.png', dpi=300)
plt.close()



# radial flux vs time
print("plot radial flux vs time")

i = 0
for lrun in listRUN:
    if (lrun=='DNS'):
        label = r"DNS"
    elif (lrun==1):
        label = r"StylES with $\epsilon_{REC}$=1"
    elif (lrun==2):
        label = r"StylES with $\epsilon_{REC}$=0.5"
    elif (lrun==3):
        label = r"StylES with $\epsilon_{REC}$=0.25"
    elif (lrun==4):
        label = r"StylES with $\epsilon_{REC}$=0.125"

    if (lrun=='DNS'):
        plt.plot(time_DNS, rflux_DNS, color='k', linewidth=0.5, linestyle='solid', label=label)
    else:
        i1 = cst[i]
        i2 = cst[i+1]
        plt.plot(time_StylES[i1:i2], rflux_StylES[i1:i2], color=cl[i], linewidth=0.5, linestyle='dashed', label=label)
        i=i+1
        
        np.savez("./results_comparison/radialFlux_vs_time", tD=time_DNS, eD=rflux_DNS, tS=time_StylES[i1:i2], eS=rflux_StylES[i1:i2])

#plt.ylim(0,3)
#plt.xlim(0,10)
plt.xlabel("time [$\omega_{ci}^{-1}$]")
plt.ylabel("radial flux")
plt.legend(fontsize="10", frameon=False)
plt.savefig('./results_comparison/radialFlux_vs_time.png', dpi=300)
plt.close()





# poloidal flux vs time
print("plot poloidal flux vs time")

i = 0
for lrun in listRUN:
    if (lrun=='DNS'):
        label = r"DNS"
    elif (lrun==1):
        label = r"StylES with $\epsilon_{REC}$=1"
    elif (lrun==2):
        label = r"StylES with $\epsilon_{REC}$=0.5"
    elif (lrun==3):
        label = r"StylES with $\epsilon_{REC}$=0.25"
    elif (lrun==4):
        label = r"StylES with $\epsilon_{REC}$=0.125"

    if (lrun=='DNS'):
        plt.plot(time_DNS, pflux_DNS, color='k', linewidth=0.5, linestyle='solid', label=label)
    else:
        i1 = cst[i]
        i2 = cst[i+1]
        plt.plot(time_StylES[i1:i2], pflux_StylES[i1:i2], color=cl[i], linewidth=0.5, linestyle='dashed', label=label)
        i=i+1

        np.savez("./results_comparison/poloidalFlux_vs_time", tD=time_DNS, eD=pflux_DNS, tS=time_StylES[i1:i2], eS=pflux_StylES[i1:i2])

#plt.ylim(-5,5)
#plt.xlim(0,10)
plt.xlabel("time [$\omega_{ci}^{-1}$]")
plt.ylabel("poloidal flux")
plt.legend(fontsize="10", frameon=False)
plt.savefig('./results_comparison/poloidalFlux_vs_time.png', dpi=300)
plt.close()



# fields
print("center domain trajectories of each field vs time")
for nf in range(3):
    i = 0
    for lrun in listRUN:
        if (lrun=='DNS'):
            if (nf==0):
                plt.plot(time_DNS, n_cDNS, color='k', linewidth=1.0, linestyle='solid', label=r"$n$ DNS")
            elif (nf==1):
                plt.plot(time_DNS, p_cDNS, color='k', linewidth=1.0, linestyle='solid', label=r"$\phi$ DNS")
            else:
                plt.plot(time_DNS, v_cDNS, color='k', linewidth=1.0, linestyle='solid', label=r"$\zeta$ DNS")
        else:
            tail = str(lrun)
            i1 = cst[i]
            i2 = cst[i+1]
            
            if (lrun==1):
                # n_label = r"$n$ StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
                # p_label = r"$\phi$ StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
                # v_label = r"$\psi$ StylES with $\epsilon_{REC}=$" + r"$10^{-1}$"
                n_label = r"$n$ StylES with $\epsilon_{REC}=$1"
                p_label = r"$\phi$ StylES with $\epsilon_{REC}=$1"
                v_label = r"$\zeta$ StylES with $\epsilon_{REC}=$1"
            elif (lrun==2):
                # n_label = r"$n$ StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
                # p_label = r"$\phi$ StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
                # v_label = r"$\zeta$ StylES with $\epsilon_{REC}=$" + r"$10^{-2}$"
                n_label = r"$n$ StylES with $\epsilon_{REC}=$0.5"
                p_label = r"$\phi$ StylES with $\epsilon_{REC}=$0.5"
                v_label = r"$\zeta$ StylES with $\epsilon_{REC}=$0.5"
            elif (lrun==3):
                # n_label = r"$n$ StylES  with $\epsilon_{REC}=$" + r"$10^{-3}$"
                # p_label = r"$\phi$ StylES  with $\epsilon_{REC}=$" + r"$10^{-3}$"
                # v_label = r"$\zeta$ StylES  with $\epsilon_{REC}=$" + r"$10^{-3}$"
                n_label = r"$n$ StylES with $\epsilon_{REC}=$0.25"
                p_label = r"$\phi$ StylES with $\epsilon_{REC}=$0.25"
                v_label = r"$\zeta$ StylES with $\epsilon_{REC}=$0.25"
            elif (lrun==4):
                # n_label = r"$n$ StylES  with $\epsilon_{REC}=$" + r"$10^{-4}$"
                # p_label = r"$\phi$ StylES  with $\epsilon_{REC}=$" + r"$10^{-4}$"
                # v_label = r"$\zeta$ StylES with $\epsilon_{REC}=$" + r"$10^{-4}$"
                n_label = r"$n$ StylES with $\epsilon_{REC}=$0.125"
                p_label = r"$\phi$ StylES with $\epsilon_{REC}=$0.125"
                v_label = r"$\zeta$ StylES with $\epsilon_{REC}=$0.125"

            if (nf==0):                        
                plt.plot(time_StylES[i1:i2], n_cStylES[i1:i2], color=cl[i], linewidth=1.0, linestyle='dashed', label=n_label)
                plt.ylabel("$n$")
            elif (nf==1):
                plt.plot(time_StylES[i1:i2], p_cStylES[i1:i2], color=cl[i], linewidth=1.0, linestyle='dashed', label=p_label)
                plt.ylabel("$\phi$")
            else:
                plt.plot(time_StylES[i1:i2], v_cStylES[i1:i2], color=cl[i], linewidth=1.0, linestyle='dashed', label=v_label)
                plt.ylabel("$\zeta$")

            i = i+1

    plt.legend(fontsize="10", frameon=False)
    plt.xlabel("time [$\omega_{ci}^{-1}$]")
    #plt.xlim(0,10)    
    if (nf==0):
        plt.savefig('./results_comparison/DNS_vs_StylES_n.png', dpi=300)
    elif (nf==1):
        plt.savefig('./results_comparison/DNS_vs_StylES_phi.png', dpi=300)
    elif (nf==2):
        plt.savefig('./results_comparison/DNS_vs_StylES_vort.png', dpi=300)
    plt.close()






#----------------------------------- MSE on images
print("MSE full trajectories of each field vs time")

for nplot in range(3):
    sumcst = 0
    cst.append(0)
    for j in range(len(listRUN)-1):

        id = 0
        MSE_n = []
        MSE_p = []
        MSE_v = []
        for i in range (cst[j+1]-cst[j]):
            ii = sumcst + i
            if (time_StylES[ii]>=time_DNS[id]):
                MSE_n.append(np.mean((n_tDNS[id] - n_tStylES[ii])**2))
                MSE_p.append(np.mean((p_tDNS[id] - p_tStylES[ii])**2))
                MSE_v.append(np.mean((v_tDNS[id] - v_tStylES[ii])**2))

                id = id+1

        sumcst = sumcst + (cst[j+1]-cst[j])

        if (j==0):
            if (nplot==0):
                n_label = r"$n$ with $\epsilon_{REC}=1$"
            elif (nplot==1):
                p_label = r"$\phi$ with $\epsilon_{REC}=1$"
            elif (nplot==2):
                v_label = r"$\zeta$ with $\epsilon_{REC}=1$"
        elif (j==1):
            if (nplot==0):
                n_label = r"$n$ with $\epsilon_{REC}=0.5$"
            elif (nplot==1):
                p_label = r"$\phi$ with $\epsilon_{REC}=0.5$"
            elif (nplot==2):
                v_label = r"$\zeta$ with $\epsilon_{REC}=0.5$"
        elif (j==2):
            if (nplot==0):
                n_label = r"$n$ with $\epsilon_{REC}=0.25$"
            elif (nplot==1):
                p_label = r"$\phi$ with $\epsilon_{REC}=0.25$"
            elif (nplot==2):
                v_label = r"$\zeta$ with $\epsilon_{REC}=0.25$"
        elif (j==3):
            if (nplot==0):
                n_label = r"$n$ with $\epsilon_{REC}=0.125$"
            elif (nplot==1):
                p_label = r"$\phi$ with $\epsilon_{REC}=0.125$"
            elif (nplot==2):
                v_label = r"$\zeta$ with $\epsilon_{REC}=0.125$"
        
        if (j==0):
            if (nplot==0):
                # plt.scatter(time_DNS[0:id], MSE_n, color='k', marker='.', s=5, label=n_label)
                plt.plot(     time_DNS[0:id], MSE_n, color='k', linestyle='solid', label=n_label)
            elif (nplot==1):
                # plt.scatter(time_DNS[0:id], MSE_p, color='k', marker='.', s=5, label=p_label)
                plt.plot(     time_DNS[0:id], MSE_p, color='k', linestyle='solid', label=p_label)
            elif (nplot==2):
                # plt.scatter(time_DNS[0:id], MSE_v, color='k', marker='.', s=5, label=v_label)
                plt.plot(     time_DNS[0:id], MSE_v, color='k', linestyle='solid', label=v_label)
        elif (j==1):
            if (nplot==0):
                # plt.scatter(time_DNS[0:id], MSE_n, color='k', marker='v', s=5, label=n_label)
                plt.plot(     time_DNS[0:id], MSE_n, color='k', linestyle='dashed', label=n_label)
            elif (nplot==1):
                # plt.scatter(time_DNS[0:id], MSE_p, color='k', marker='v', s=5, label=p_label)
                plt.plot(     time_DNS[0:id], MSE_p, color='k', linestyle='dashed', label=p_label)
            elif (nplot==2):
                # plt.scatter(time_DNS[0:id], MSE_v, color='k', marker='v', s=5, label=v_label)
                plt.plot(     time_DNS[0:id], MSE_v, color='k', linestyle='dashed', label=v_label)
        elif (j==2):
            if (nplot==0):
                # plt.scatter(time_DNS[0:id], MSE_n, color='k', marker='^', s=5, label=n_label)
                plt.plot(     time_DNS[0:id], MSE_n, color='k', linestyle='dotted', label=n_label)
            elif (nplot==1):
                # plt.scatter(time_DNS[0:id], MSE_p, color='k', marker='^', s=5, label=p_label)
                plt.plot(     time_DNS[0:id], MSE_p, color='k', linestyle='dotted', label=p_label)
            elif (nplot==2):
                # plt.scatter(time_DNS[0:id], MSE_v, color='k', marker='^', s=5, label=v_label)
                plt.plot(     time_DNS[0:id], MSE_v, color='k', linestyle='dotted', label=v_label)
        elif (j==3):
            if (nplot==0):
                # plt.scatter(time_DNS[0:id], MSE_n, color='k', marker='x', s=5, label=n_label)
                plt.plot(     time_DNS[0:id], MSE_n, color='k', linestyle='dashdot', label=n_label)
            elif (nplot==1):
                # plt.scatter(time_DNS[0:id], MSE_p, color='k', marker='x', s=5, label=p_label)
                plt.plot(     time_DNS[0:id], MSE_p, color='k', linestyle='dashdot', label=p_label)
            elif (nplot==2):
                # plt.scatter(time_DNS[0:id], MSE_v, color='k', marker='x', s=5, label=v_label)
                plt.plot(     time_DNS[0:id], MSE_v, color='k', linestyle='dashdot', label=v_label)
            

    # if (nplot==1):
    #     plt.ylim(0,2.0)
    # else:
    #     plt.ylim(0,0.1)
    #plt.xlim(0,10)
    plt.legend(fontsize="10", loc ="upper left", frameon=False)
    plt.xlabel("time [$\omega_{ci}^{-1}$]")
    plt.ylabel("MSE")
    plt.savefig('./results_comparison/MSE_fields_' + str(nplot) + '.png', dpi=300)
    plt.close()






#----------------------------------- plot performance
print("plot performance")

# Using readlines()
file1 = open('data_performance.txt', 'r')
Lines = file1.readlines()
 
A = []
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    if count>2:
        vals = line.split()
        for i in vals:
            A.append(float(i))
            

B = np.asarray(A)
B = np.reshape(B, [5,8])
n_DNS = B[:,0]

plt.plot(n_DNS, B[:,4], label="BOUT++",  color='k', linewidth=0.5, linestyle='solid')
plt.plot(n_DNS, B[:,5], label="StylES",  color='r', linewidth=0.5, linestyle='solid')
plt.plot(n_DNS, B[:,6], label="$N^2$",   color='k', linewidth=0.5, linestyle='dashed')
plt.plot(n_DNS, B[:,7], label="$NlogN$", color='r', linewidth=0.5, linestyle='dashed')

#plt.xlim(256,4096)
xrange=[512,1024,2048,4096]
plt.xticks(xrange)
plt.xlabel("N")
plt.ylabel("time per time step [s]")
plt.legend(frameon=False)
#plt.grid(visible=True)
plt.savefig("./results_comparison/performance_BOUT_vs_StylES.png")






# # fields
# colDNS = np.linspace(0,1,num=len(time_DNS))
# for f in range(3):

#     if (f==0):
#         fDNS    = n_cDNS
#         fStylES = n_cStylES
#         cmap    = 'jet'
#         label   = r"$n$"
#     elif (f==1):
#         fDNS    = p_cDNS
#         fStylES = p_cStylES
#         cmap    = 'jet'
#         label   = r"$\phi$"
#     elif (f==2):
#         fDNS    = v_cDNS
#         fStylES = v_cStylES
#         cmap    = 'jet'
#         label   = r"$\zeta$"

#     i = 0
#     for lrun in listRUN:
            
#         if (lrun=='DNS'):
#             marker  = '^'
#             plt.scatter(time_DNS, fDNS, c=colDNS, vmax=2, cmap=cmap, marker=marker, s=5, label=label + " DNS")
#         else:
#             tail = str(lrun)
#             i1 = cst[i]
#             i2 = cst[i]+cst[i+1]
            
#             if (tail=="2"):
#                 marker  = 'o'
#                 n_label   = r"$n$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
#                 p_label   = r"$\phi$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
#                 v_label   = r"$\zeta$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
#                 colStylES = np.linspace(0,0.5,num=len(time_StylES[i1:i2]))
#                 vmax      = 1.5
#             else:
#                 marker  = '*'
#                 n_label   = r"$n$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#                 p_label   = r"$\phi$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#                 v_label   = r"$\zeta$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#                 colStylES = np.linspace(0,1,num=len(time_StylES[i1:i2]))
#                 vmax      = 1
            
#             plt.scatter(time_StylES[i1:i2], fStylES[i1:i2], c=colStylES, vmax=vmax, cmap=cmap, marker=marker, s=5, label=label + " StylES")
                    
#             i = i+1

#     plt.legend(fontsize="10", loc ="lower left", frameon=False)
#     plt.xlabel("time [$\omega_{ci}^{-1}$]")
#     plt.ylabel(label)
#     plt.savefig('./results_comparison/DNS_vs_StylES_UVP_' + str(f) + '.png', dpi=300)
#     plt.close()



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
#         v_label   = r"$\zeta$ with $\epsilon_{REC}=$" + r"$10^{-2}$"
#         colStylES = np.linspace(0,0.5,num=len(MSE_n))
#         vmax      = 1.5        
#     else:
#         n_label   = r"$n$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#         p_label   = r"$\phi$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#         v_label   = r"$\zeta$ with $\epsilon_{REC}=$" + r"$10^{-3}$"
#         colStylES = np.linspace(0,1,num=len(MSE_n))
#         vmax      = 1        

#     plt.scatter(time_DNS[0:id], MSE_n, c=colStylES, vmax=vmax, cmap=cmap, marker='^', s=5, label=n_label)
#     plt.scatter(time_DNS[0:id], MSE_p, c=colStylES, vmax=vmax, cmap=cmap, marker='o', s=5, label=p_label)
#     plt.scatter(time_DNS[0:id], MSE_v, c=colStylES, vmax=vmax, cmap=cmap, marker='*', s=5, label=v_label)
            
# plt.legend(fontsize="10", loc ="upper right", frameon=False)
# plt.xlabel("time [$\omega_{ci}^{-1}$]")
# plt.ylabel("MSE")
# plt.savefig('./results_comparison/MSE_fields.png', dpi=300)
# plt.close()