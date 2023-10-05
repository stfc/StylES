import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import sys

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect

sys.path.insert(0, '../LES_Solvers/')

from LES_plot import *
from LES_functions import *

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../../codes/TurboGenPY/')

from tkespec import compute_tke_spectrum2d
from isoturb import generate_isotropic_turbulence_2d

#----------------------------- parameters
MODE        = 'MAKE_ANIMATION'   #'READ_NUMPY', 'MAKE_ANIMATION', 'READ_NETCDF'
# PATH_NUMPY  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/results_StylES/fields/"
PATH_NUMPY  = "../utilities/results_checkStyles/fields/"
PATH_NETCDF = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/data/"
PATH_ANIMAT_ENERGY = "./analysis/energy/"
# PATH_ANIMAT_PLOTS = "./analysis/plots/"
# PATH_ANIMAT_PLOTS = "../utilities/results_checkStyles/plots/"
# PATH_ANIMAT_PLOTS = "../utilities/results_reconstruction/plots/"
PATH_ANIMAT_PLOTS = "../../StylES/utilities/results_reconstruction/plots/"
FIND_MIXMAX = True
DTYPE       = 'float32'
DIR         = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
STIME       = 0 # starting time to take as first image

if (MODE=='READ_NUMPY'):
    files = os.listdir(PATH_NUMPY)
    FTIME = len(files)
elif (MODE=='MAKE_ANIMATION'):
    files = os.listdir(PATH_ANIMAT_PLOTS)
    FTIME = len(files)
else:
    FTIME = 1 # final time from netCFD

print("There are " + str(FTIME) + " files")

ITIME       = 1    # skip between STIME, FTIME, ITIME
SKIP        = 1   # cd - skip between time steps when reading NUMPY arrays
DELT        = 1.0  # delt equal to timestep in BOUT++ input file
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
time      = []
Energy    = []
L         = 50.176 
N         = 512
delx      = L/N
dely      = L/N

# delete folders
if (MODE=='READ_NUMPY' or MODE=='READ_NETCDF'):
    os.system("rm -rf analysis/plots/*")
    os.system("rm -rf analysis/fields/*")
    os.system("rm -rf analysis/energy/*")
    os.system("mkdir -p analysis/plots/")
    os.system("mkdir -p analysis/fields/")
    os.system("mkdir -p analysis/energy/")


#----------------------------- functions
def convert(x):
    return x


def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))


def save_fields(totTime, U, V, P, filename="restart.npz"):

    # save restart file
    np.savez(filename, t=totTime, U=U, V=V, P=P)



#----------------------------- select MODE
if (MODE=='READ_NETCDF'):

    # create folders fields and paths
    path = "analysis"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    path = "analysis/fields"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm analysis/fields/*"
        os.system(cmd)
        
    path = "analysis/plots"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm analysis/plots/*"
        os.system(cmd)
        
    path = "analysis/energy"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm analysis/energy/*"
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
        
        filename = "../../../../../StylES/bout_interfaces/analysis/fields/fields_time" + str(t).zfill(5) + ".npz"
        save_fields(t, Img_n, Img_phi, Img_vort, filename=filename)

        if (t%1==0):
            filename = "../../../../../StylES/bout_interfaces/analysis/plots/plots_time" + str(t).zfill(5) + ".png"
            if (FIND_MIXMAX):
                print_fields_3(Img_n, Img_phi, Img_vort, filename=filename, diff=False, \
                    Umin=min_U, Umax=max_U, Vmin=min_V, Vmax=max_V, Pmin=min_P, Pmax=max_P)
            else:
                print_fields_3(Img_n, Img_phi, Img_vort, filename=filename, diff=False)
                    # Umin=-10.0, Umax=10.0, Vmin=-10.0, Vmax=10.0, Pmin=-10.0, Pmax=10.0)

        gradV_phi = np.sqrt(((cr(Img_phi, 1, 0) - cr(Img_phi, -1, 0))/(2.0*delx))**2 + ((cr(Img_phi, 0, 1) - cr(Img_phi, 0, -1))/(2.0*dely))**2)

        time.append(t*DELT)
        E = 0.5*L**2*np.sum(Img_n**2 + gradV_phi**2)
        Energy.append(E)
        
        closePlot=False
        if (t%100==0 or (t+ITIME>FTIME-1)):
            if (t+ITIME>FTIME-1):
                closePlot=True
            filename = "../../../../../StylES/bout_interfaces/analysis/energy/Spectrum_" + str(t).zfill(4) + ".png"
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
            simtime   = i #np.cast[DTYPE](data['simtime'])
            Img_n     = np.cast[DTYPE](data['U'])
            Img_phi   = np.cast[DTYPE](data['V'])
            Img_vort  = np.cast[DTYPE](data['P'])
            
            file_dest = file.replace("fields","plots")
            file_dest = file.replace(".npz",".png")
            filename  = "./analysis/plots/" + file_dest
            if (FIND_MIXMAX):
                print_fields_3(Img_n, Img_phi, Img_vort, filename=filename, diff=False, \
                    Umin=min_U, Umax=max_U, Vmin=min_V, Vmax=max_V, Pmin=min_P, Pmax=max_P)
            else:
                print_fields_3(Img_n, Img_phi, Img_vort, filename=filename, diff=False)
                    # Umin=-10.0, Umax=10.0, Vmin=-10.0, Vmax=10.0, Pmin=-10.0, Pmax=10.0)

            gradV_phi = np.sqrt(((cr(Img_phi, 1, 0) - cr(Img_phi, -1, 0))/(2.0*delx))**2 \
                + ((cr(Img_phi, 0, 1) - cr(Img_phi, 0, -1))/(2.0*dely))**2)

            time.append(simtime)
            E = 0.5*L**2*np.sum(Img_n**2 + gradV_phi**2)
            Energy.append(E)
            
            if (i%1==0 and i!=nfiles-1):
                filename = "./analysis/energy/Spectrum_" + str(i).zfill(4) + ".png"
                plot_spectrum(Img_n, gradV_phi, L, filename, close=closePlot)                

            if (i+SKIP>nfiles-1):
                closePlot=True
                filename = "./analysis/energy/Spectrum_" + str(i).zfill(4) + ".png"
                plot_spectrum(Img_n, gradV_phi, L, filename, close=closePlot)                

            print ("done for file " + file_dest)



# plot energy
if (MODE=='READ_NETCDF' or MODE=='READ_NUMPY'):
    filename="./analysis/energy/energy_vs_time"
    np.savez(filename, time=time, Energy=Energy)
    plt.plot(time, Energy)
    plt.savefig('./analysis/energy_vs_time.png')
    plt.close()



#----------------------------- make animation fields
anim_file = 'animation_plots.gif'
filenames = glob.glob(PATH_ANIMAT_PLOTS + "/*.png")
filenames = sorted(filenames)

if (MODE!="MAKE_ANIMATION"):
    ftime = np.diff(time)
    ftime = np.insert(ftime, 0, time[0])
    ftime = np.min(ftime) + ftime
    ftime = ftime*DELT
    ftime = ftime.tolist()

with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)



#----------------------------- make animation energy
anim_file = 'animation_energy.gif'
filenames = glob.glob(PATH_ANIMAT_ENERGY + "/*.png")
filenames = sorted(filenames)

if (MODE!="MAKE_ANIMATION"):
    ftime = np.diff(time)
    ftime = np.insert(ftime, 0, time[0])
    ftime = np.min(ftime) + ftime
    ftime = ftime*DELT
    ftime = ftime.tolist()

with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)