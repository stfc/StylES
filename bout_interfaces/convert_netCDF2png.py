import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import sys

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect
from pyevtk.hl import gridToVTK

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
MODE        = 'READ_NETCDF'   # 'READ_NETCDF', 'READ_NUMPY', 'MAKE_ANIMATION'
PATH        = "../../BOUT-dev/build_release/examples/hasegawa-wakatani-3d/"
PATH_NUMPY  = PATH + "results_StylES/fields/"
# PATH_NUMPY  = "../utilities/results_checkStyles/fields/"
PATH_NETCDF = PATH + "data/"
PATH_ANIMAT_ENERGY = "./results/energy/"
PATH_ANIMAT_PLOTS = "./results/plots/"
# PATH_ANIMAT_PLOTS = "../utilities/results_checkStyles/plots/"
# PATH_ANIMAT_PLOTS = "../utilities/results_reconstruction/plots/"
# PATH_ANIMAT_PLOTS = "../../StylES/utilities/results_reconstruction/plots/"
FIND_MIXMAX = 1   # " 0) yes, 1) use INIT_SCA, 2) use None "
DTYPE       = 'float32'
DIR         = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
STIME       = 0  # starting time to take as first image
ITIME       = 1 # skip between STIME, FTIME, ITIME
PLOT_2D     = True
PLOT_VTK    = True
SAVE_FIELDS = False

useLogSca   = True
xLogLim     = [1.0e-2, 100]   # to do: to make nmore general
yLogLim     = [1.e-10, 10.]
xLinLim     = [0.0e0, 600] 
yLinLim     = [0.0e0, 1.0]
time        = []
Energy      = []
L           = 50.176 
N           = OUTPUT_DIM
delx        = L/N
dely        = L/N

if (DIMS_3D):
    file = open(PATH_NETCDF + "/BOUT.inp", 'r')
    for line in file:
        if "nx =" in line:
            NX = int(line.split()[2]) - 4
        if "ny =" in line:
            NY = int(line.split()[2])
        if "nz =" in line:
            NZ = int(line.split()[2])
        if "Lx =" in line:
            LX = float(line.split()[2])
        if "Ly =" in line:
            LY = float(line.split()[2])
        if "Lz =" in line:
            LZ = float(line.split()[2])

    if (MODE=='READ_NUMPY'):
        NX = NX*RS
        NZ = NZ*RS
    
    DX  = LX/NX
    DY  = LY/NY
    DZ  = LZ/NZ
    
    NY2 = int(NY/2)-1

    print("System sizes are: Lx,Ly,Lz,dx,dy,dz,nx,ny,nz =", LX,LY,LZ,DX,DY,DZ,NX,NY,NZ)


#----------------------------- initialize
if (MODE=='READ_NETCDF'):

    # find number of timesteps
    CWD = os.getcwd()
    os.chdir(PATH_NETCDF)
    n = collect("n", xguards=False, info=False)
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
            print("DELT is ", DELT)
            break

elif (MODE=='READ_NUMPY'):

    # find timestep
    file = open(PATH_NETCDF + "BOUT.inp", 'r')
    for line in file:
        if "timestep" in line:
            DELT = float(line.split()[2])
            print("DELT is ", DELT)
            break

    files = os.listdir(PATH_NUMPY)
    FTIME = len(files)
    TGAP = 0.0
    print("starting time ", TGAP)    

elif (MODE=='MAKE_ANIMATION'):

    files = os.listdir(PATH_ANIMAT_PLOTS)
    FTIME = len(files)

print("There are " + str(FTIME) + " files")

if (FIND_MIXMAX==1):
    min_U = -INIT_SCA
    max_U =  INIT_SCA
    min_V = -INIT_SCA
    max_V =  INIT_SCA
    min_P = -INIT_SCA
    max_P =  INIT_SCA
else:
    min_U = None
    max_U = None
    min_V = None
    max_V = None
    min_P = None
    max_P = None
    
    
# delete folders
if (MODE=='READ_NUMPY' or MODE=='READ_NETCDF'):
    os.system("rm -rf results/plots/*")
    os.system("rm -rf results/fields/*")
    os.system("rm -rf results/energy/*")
    os.system("mkdir -p results/plots/")
    os.system("mkdir -p results/fields/")
    os.system("mkdir -p results/energy/")



#----------------------------- functions
def convert(x):
    return x


def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))


def save_fields(totTime, U, V, P, filename="restart.npz"):

    # save restart file
    np.savez(filename, t=totTime, U=U, V=V, P=P)


if (DIMS_3D):
    x = np.linspace(0,LX,NX)
    y = np.linspace(0,LY,NY)
    z = np.linspace(0,LZ,NZ)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

#----------------------------- select MODE
if (MODE=='READ_NETCDF'):

    # create folders fields and paths
    path = "results"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    path = "results/fields"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm results/fields/*"
        os.system(cmd)
        
    path = "results/plots"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm results/plots/*"
        os.system(cmd)
        
    path = "results/energy"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        cmd = "rm results/energy/*"
        os.system(cmd)


    #------------ run on data
    print("reading " + PATH_NETCDF)

    os.chdir(PATH_NETCDF)
    
    if (FIND_MIXMAX==0):
        t_n    = collect("n",    tind=0, xguards=False, info=False)
        t_phi  = collect("phi",  tind=0, xguards=False, info=False)
        t_vort = collect("vort", tind=0, xguards=False, info=False)
        min_U = np.min(t_n)
        max_U = np.max(t_n)
        min_V = np.min(t_phi)
        max_V = np.max(t_phi)
        min_P = np.min(t_vort)
        max_P = np.max(t_vort)

        for t in range(STIME,FTIME,ITIME):
            t_n    = collect("n",    tind=t, xguards=False, info=False)
            t_phi  = collect("phi",  tind=t, xguards=False, info=False)
            t_vort = collect("vort", tind=t, xguards=False, info=False)

            min_U = min(np.min(t_n), min_U)
            max_U = max(np.max(t_n), max_U)
            min_V = min(np.min(t_phi), min_V)
            max_V = max(np.max(t_phi), max_V)
            min_P = min(np.min(t_vort), min_P)
            max_P = max(np.max(t_vort), max_P)
            
    
    print(min_U, max_U, min_V, max_V, min_P, max_P)

    for t in range(STIME,FTIME,ITIME):
        t_n    = collect("n",    tind=t, xguards=False, info=False)
        t_phi  = collect("phi",  tind=t, xguards=False, info=False)
        t_vort = collect("vort", tind=t, xguards=False, info=False)

        n = t_n[0,:,:,:]
        phi = t_phi[0,:,:,:]
        vort = t_vort[0,:,:,:]
        
        dest = "../../../../../StylES/bout_interfaces/results/" 
        tail = str(t).zfill(5)
        if (SAVE_FIELDS):
            filename = dest + "/fields/fields_time" + tail + ".npz"
            save_fields(t, n, phi, vort, filename=filename)

        if (PLOT_VTK):
            filename = dest + "/fields/fields_time" + tail
            gridToVTK(filename, X, Y, Z, pointData={"n": n, "phi": phi, "vort": vort})

        # plot, energy and spectra
        n = n[:,0,:]
        phi = phi[:,0,:]
        vort = vort[:,0,:]

        # plot
        if (PLOT_2D):
            filename = dest + "/plots/plots_time" + tail + ".png"
            print_fields_3(n, phi, vort, filename=filename, transpose=True, \
                Umin=min_U, Umax=max_U, Vmin=min_V, Vmax=max_V, Pmin=min_P, Pmax=max_P)

        # energy
        dVdx = (-cr(phi, 2, 0) + 8*cr(phi, 1, 0) - 8*cr(phi, -1,  0) + cr(phi, -2,  0))/(12.0*DELX_LES)
        dVdy = (-cr(phi, 0, 2) + 8*cr(phi, 0, 1) - 8*cr(phi,  0, -1) + cr(phi,  0, -2))/(12.0*DELY_LES)

        time.append(TGAP + t*DELT)
        E = 0.5*L**2*np.sum(n**2 + dVdx**2 + dVdy**2)
        Energy.append(E)
        
       
        # spectra
        closePlot=False
        if (t%1==0 or (t+ITIME>FTIME-1)):
            if (t+ITIME>FTIME-1):
                closePlot=True
            filename = "../../../../../StylES/bout_interfaces/results/energy/Spectrum_" + tail + ".png"
            plot_spectrum_2d_3v(n, dVdx, dVdy, L, filename, label="StylES", close=closePlot)
        
        print("done for file time step", t)

    os.chdir("../../../../../StylES/bout_interfaces/")
    
elif (MODE=='READ_NUMPY'):

    #------------ dry-run to find min/max
    if (FIND_MIXMAX==0):
        print("Finding min/max accross all data...")
        
        min_U = 1e10
        max_U = 1e-10
        min_V = 1e10
        max_V = 1e-10
        min_P = 1e10
        max_P = 1e-10
        
        files = os.listdir(PATH_NUMPY)
        nfiles = len(files)
        for t,file in enumerate(sorted(files)):
            if (t%ITIME==0):
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
    for t,file in enumerate(sorted(files)):
        if (t%ITIME==0):
            filename  = PATH_NUMPY + file
            data      = np.load(filename)
            simtime   = np.cast[DTYPE](data['simtime'])
            n    = np.cast[DTYPE](data['U'])
            phi  = np.cast[DTYPE](data['V'])
            vort = np.cast[DTYPE](data['P'])

            # adjust this is due to the use of BATCH_SIZE for y dimension...
            if (DIMS_3D):
                n    = np.transpose(n, (1,0,2))
                phi  = np.transpose(phi, (1,0,2))
                vort = np.transpose(vort, (1,0,2))
                n    = np.ascontiguousarray(n)
                phi  = np.ascontiguousarray(phi)
                vort = np.ascontiguousarray(vort)

            file_dest = file.replace("fields","plots")
            file_dest = file.replace("fields_DNS_DNS","plots")
            if (PLOT_VTK):
                file_dest = file.replace(".npz","")
                filename = "./results/fields/" + file_dest
                gridToVTK(filename, X, Y, Z, pointData={"n": n, "phi": phi, "vort": vort})

            # plot, energy and spectra
            n = n[:,0,:]
            phi = phi[:,0,:]
            vort = vort[:,0,:]
        
            # plot
            if (PLOT_2D):
                file_dest = file.replace(".npz",".png")
                filename  = "./results/plots/" + file_dest
                print_fields_3(n, phi, vort, filename=filename, transpose=True, \
                    Umin=min_U, Umax=max_U, Vmin=min_V, Vmax=max_V, Pmin=min_P, Pmax=max_P)

            # energy
            dVdx = (-cr(phi, 2, 0) + 8*cr(phi, 1, 0) - 8*cr(phi, -1,  0) + cr(phi, -2,  0))/(12.0*DELX)
            dVdy = (-cr(phi, 0, 2) + 8*cr(phi, 0, 1) - 8*cr(phi,  0, -1) + cr(phi,  0, -2))/(12.0*DELY)

            time.append(simtime)
            E = 0.5*L**2*np.sum(n**2 + dVdx**2 + dVdy**2)
            Energy.append(E)
            
            # spectra
            if (t%1==0 and t!=nfiles-1):
                closePlot=False

            if (t+ITIME>nfiles-1):
                closePlot=True

            filename = "./results/energy/Spectrum_" + str(t).zfill(4) + ".png"
            plot_spectrum_2d_3v(n, dVdx, dVdy, L, filename, label="StylES", close=closePlot)
                
            print ("done step " + str(t) + "/"+ str(nfiles) + " for file " + file_dest)



# plot energy
if (MODE=='READ_NETCDF' or MODE=='READ_NUMPY'):
    filename="./results/energy/energy_vs_time"
    np.savez(filename, time=time, Energy=Energy)
    plt.plot(time, Energy, 'o-')
    plt.savefig('./results/energy_vs_time.png')
    plt.close()



#----------------------------- make animation fields
if (PLOT_2D):
    anim_file = './results/animation_plots.gif'
    filenames = glob.glob(PATH_ANIMAT_PLOTS + "/*.png")
    filenames = sorted(filenames)

    with imageio.get_writer(anim_file, mode='I', fps=2) as writer:
        for filename in filenames:
            print(filename)
            image = imageio.v2.imread(filename)
            writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)




#----------------------------- make animation energy
anim_file = './results/animation_energy.gif'
filenames = glob.glob(PATH_ANIMAT_ENERGY + "/*.png")
filenames = sorted(filenames)

with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)

