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

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../../codes/TurboGenPY/')

from tkespec import compute_tke_spectrum2d
from isoturb import generate_isotropic_turbulence_2d

#----------------------------- parameters
MODE        = 'MAKE_ANIMATION'   #'READ_NUMPY', 'MAKE_ANIMATION', 'READ_NETCDF'
PATH_NUMPY  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/results_bout/fields/"
PATH_NETCDF = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/data/"
PATH_ANIMAT = "./results_bout/plots/*.png"
#PATH_ANIMAT = "../utilities/results_reconstruction/plots/*.png"
FIND_MIXMAX = False
DTYPE       = 'float32'
DIR         = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
STIME       = 0 # starting time to take as first image
FTIME       = 455 # starting time to take as last image
ITIME       = 1  # skip between STIME, FTIME, ITIME
SKIP        = 1  # skip between time steps when reading NUMPY arrays
min_U       = None
max_U       = None
min_V       = None
max_V       = None
min_P       = None
max_P       = None

# plot spectrum vars
useLogSca = True
xLogLim   = [1.0e-2, 100]   # to do: to make nmore general
yLogLim   = [1.e-10, 10.]
xLinLim   = [0.0e0, 600]
yLinLim   = [0.0e0, 1.0]
time      = np.linspace(STIME, FTIME, FTIME)
Energy    = np.zeros((FTIME), dtype=DTYPE)
L         = 50.176 
N         = 512
delx      = L/N
dely      = L/N

# delete folders
if (MODE=='READ_NUMPY' or MODE=='READ_NETCDF'):
    os.system("rm -rf results_bout/plots/*")
    os.system("rm -rf results_bout/fields/*")
    os.system("rm -rf results_bout/energy/*")
    os.system("mkdir -p results_bout/plots/")
    os.system("mkdir -p results_bout/fields/")
    os.system("mkdir -p results_bout/energy/")


#----------------------------- functions
def convert(x):
    return x


def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))


def save_fields(totTime, U, V, P, filename="restart.npz"):

    # save restart file
    np.savez(filename, t=totTime, U=U, V=V, P=P)


def plot_spectrum(U, V, L, filename, close=True, label=None):
    U_cpu = convert(U)
    V_cpu = convert(V)

    knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d(U_cpu, V_cpu, L, L, True)

    if useLogSca:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(xLogLim)
        plt.ylim(yLogLim)        
    else:
        plt.xlim(xLinLim)
        plt.ylim(yLinLim) 

    if (label is not None):
        plt.plot(wave_numbers, tke_spectrum, '-', linewidth=0.5, label=label)
        plt.legend()
    else:    
        plt.plot(wave_numbers, tke_spectrum, '-', linewidth=0.5)
   

    if (close):
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    filename = filename.replace(".png",".txt")
    np.savetxt(filename, np.c_[wave_numbers, tke_spectrum], fmt='%1.4e')   # use exponential notation





#----------------------------- select MODE
if (MODE=='READ_NETCDF'):

    # create folders fields and paths
    path = "results_bout"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    path = "results_bout/fields"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm results_bout/fields/*"
        os.system(cmd)
        
    path = "results_bout/plots"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm results_bout/plots/*"
        os.system(cmd)
        
    path = "results_bout/energy"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm results_bout/energy/*"
        os.system(cmd)


    #------------ dry-run to find min-max
    if (FIND_MIXMAX):
        os.chdir(PATH_NETCDF)

        print("Finding min/max accross all data...")

        min_U = 1e10
        max_U = 1e-10
        min_V = 1e10
        max_V = 1e-10
        min_P = 1e10
        max_P = 1e-10
        
        n    = collect("n",    xguards=False, info=False)
        phi  = collect("phi",  xguards=False, info=False)
        vort = collect("vort", xguards=False, info=False)

        for t in range(STIME,FTIME,ITIME):
            Img_n = n[t,:,0,:]
            min_U = min(np.min(Img_n), min_U)
            max_U = max(np.max(Img_n), max_U)

            Img_phi = phi[t,:,0,:]
            min_V = min(np.min(Img_phi), min_V)
            max_V = max(np.max(Img_phi), max_V)

            Img_vort = vort[t,:,0,:]
            min_P = min(np.min(Img_vort), min_P)
            max_P = max(np.max(Img_vort), max_P)
            
        os.chdir("../../../../../StylES/bout_interfaces/")

    print(min_U, max_U, min_V, max_V, min_P, max_P)

    #------------ run on data
    print("reading " + PATH_NETCDF)

    os.chdir(PATH_NETCDF)
    
    n    = collect("n",    xguards=False, info=False)
    phi  = collect("phi",  xguards=False, info=False)
    vort = collect("vort", xguards=False, info=False)

    for t in range(STIME,FTIME,ITIME):
        Img_n = n[t,:,0,:]
        # maxv = np.max(Img_n)
        # minv = np.min(Img_n)
        # Img_n = (Img_n - minv)/(maxv - minv)

        Img_phi = phi[t,:,0,:]
        # maxv = np.max(Img_phi)
        # minv = np.min(Img_phi)
        # Img_phi = (Img_phi - minv)/(maxv - minv)

        Img_vort = vort[t,:,0,:]
        # maxv = np.max(Img_vort)
        # minv = np.min(Img_vort)
        # Img_vort = (Img_vort - minv)/(maxv - minv)
        
        filename = "../../../../../StylES/bout_interfaces/results_bout/fields/fields_time" + str(t).zfill(5) + ".npz"
        save_fields(t, Img_n, Img_phi, Img_vort, filename=filename)

        if (t%1==0):
            filename = "../../../../../StylES/bout_interfaces/results_bout/plots/plots_time" + str(t).zfill(5) + ".png"
            print_fields_3(Img_n, Img_phi, Img_vort, filename=filename, Umin=min_U, Umax=max_U, Vmin=min_V, Vmax=max_V, Pmin=min_P, Pmax=max_P)

        gradV_phi = np.sqrt(((cr(Img_phi, 1, 0) - cr(Img_phi, -1, 0))/(2.0*delx))**2 + ((cr(Img_phi, 0, 1) - cr(Img_phi, 0, -1))/(2.0*dely))**2)

        time[t] = t
        Energy[t] = 0.5*L**2*np.sum(Img_n**2 + gradV_phi**2)
        
        closePlot=False
        if (t%1==0):
            if (t==FTIME-1):
                closePlot=True
            filename = "../../../../../StylES/bout_interfaces/results_bout/energy/Spectrum_" + str(t).zfill(4) + ".png"
            plot_spectrum(Img_n, gradV_phi, L, filename, close=closePlot)                
        
        print("min/max", np.min(Img_n), np.max(Img_n), np.min(Img_phi), np.max(Img_phi), np.min(Img_vort), np.max(Img_vort))
        # print("average", t, np.mean(Img_n), np.mean(Img_phi), np.mean(Img_vort))
        print("done for file time step", t)

    os.chdir("../../../../../StylES/bout_interfaces/")
    
elif (MODE=='READ_NUMPY'):

    #------------ dry-run to find min/max
    if (FIND_MIXMAX):
        print("Finding min/max accross all data...")
        
        min_U = 1e10
        max_U = 1e-10
        min_V = 1e10
        max_V = 1e-10
        min_P = 1e10
        max_P = 1e-10
        
        files = os.listdir(PATH_NUMPY)
        nfiles = len(files)
        for i,file in enumerate(sorted(files)):
            if (i%SKIP==0):
                filename = PATH_NUMPY + file
                data     = np.load(filename)
                Img_n    = np.cast[DTYPE](data['U'])
                Img_phi    = np.cast[DTYPE](data['V'])
                Img_vort    = np.cast[DTYPE](data['P'])
                
                min_U = min(np.min(Img_n), min_U)
                max_U = max(np.max(Img_n), max_U)
                min_V = min(np.min(Img_phi), min_V)
                max_V = max(np.max(Img_phi), max_V)
                min_P = min(np.min(Img_vort), min_P)
                max_P = max(np.max(Img_vort), max_P)            
                
                
        print("Found min/max = ", min_U, max_U, min_V, max_V, min_P, max_P)
    
    #------------ run on data
    files = os.listdir(PATH_NUMPY)
    nfiles = len(files)
    closePlot=False
    for i,file in enumerate(sorted(files)):
        if (i%SKIP==0):
            filename  = PATH_NUMPY + file
            data      = np.load(filename)
            Img_n     = np.cast[DTYPE](data['U'])
            Img_phi   = np.cast[DTYPE](data['V'])
            Img_vort  = np.cast[DTYPE](data['P'])
            
            # if (i>0 and i<301):
            #     Img_phi = Img_phi*0
            # if (i>400 and i<601):
            #     Img_n = Img_n*0

            file_dest = file.replace("fields","plots")
            file_dest = file.replace(".npz",".png")
            filename  = "./results_bout/plots/" + file_dest
            print_fields_3(Img_n, Img_phi, Img_vort, filename=filename, diff=False, \
                Umin=min_U, Umax=max_U, Vmin=min_V, Vmax=max_V, Pmin=min_P, Pmax=max_P)

            gradV_phi = np.sqrt(((cr(Img_phi, 1, 0) - cr(Img_phi, -1, 0))/(2.0*delx))**2 \
                + ((cr(Img_phi, 0, 1) - cr(Img_phi, 0, -1))/(2.0*dely))**2)

            time[i] = i
            Energy[i] = 0.5*L**2*np.sum(Img_n**2 + gradV_phi**2)
            
            if (i%1==0 and i!=nfiles-1):
                filename = "./results_bout/energy/Spectrum_" + str(i).zfill(4) + ".png"
                plot_spectrum(Img_n, gradV_phi, L, filename, close=closePlot)                

            if (i==nfiles-1):
                closePlot=True
                filename = "./results_bout/energy/Spectrum_" + str(i).zfill(4) + ".png"
                plot_spectrum(Img_n, gradV_phi, L, filename, close=closePlot)                

            print ("done for file " + file_dest)



# plot energy
if (MODE=='READ_NETCDF' or MODE=='READ_NUMPY'):
    filename="./results_bout/energy/energy_vs_time"
    np.savez(filename, time=time, Energy=Energy)
    plt.plot(time, Energy)
    plt.savefig('./results_bout/energy/energy_vs_time.png')
    plt.close()



# #----------------------------- make animation
anim_file = 'animation.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(PATH_ANIMAT)
    filenames = sorted(filenames)
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)
