import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import sys

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../../codes/TurboGenPY/')

from tkespec import compute_tke_spectrum2d
from isoturb import generate_isotropic_turbulence_2d

#----------------------------- parameters
MODE        = 'READ_NETCDF'   #'READ_NUMPY', 'MAKE_ANIMATION', 'READ_NETCDF'
PATH_NUMPY  = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/results_bout/plots/DNS/"
PATH_NETCDF = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/data/"
FIND_MIXMAX = True
DTYPE       = 'float32'
DIR         = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
STIME       = 0 # starting time to take as first image
FTIME       = 101 # starting time to take as last image
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


def print_fields(U_, V_, P_, filename, diff=False, \
    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None):
    # Umin=-0.3, Umax=0.3, Vmin=-0.3, Vmax=0.3, Pmin=-0.3, Pmax=0.3):
    # Umin=-5.0, Umax=5.0, Vmin=-5.0, Vmax=5.0, Pmin=-5.0, Pmax=5.0):
    # Umin=-10.0, Umax=10.0, Vmin=-10.0, Vmax=10.0, Pmin=-10.0, Pmax=10.0):

    #-------- convert to numpy arrays
    U = convert(U_)
    V = convert(V_)
    P = convert(P_)

    if (DIR==1):    # plot along vertical
        U = np.transpose(U, axes=[1,0])
        V = np.transpose(V, axes=[1,0])
        P = np.transpose(P, axes=[1,0])

    N = len(U[0,:])

    #--------- plot surfaces
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    fig.subplots_adjust(hspace=0.25)

    ax1 = axs[0,0]
    ax2 = axs[1,0]
    ax3 = axs[0,1]
    ax4 = axs[1,1]
    ax5 = axs[0,2]
    ax6 = axs[1,2]

    if (diff):
        cmap0 = 'hot'
        cmap1 = 'hot'
        cmap2 = 'jet'
        ax2title = (r'diff $n$')
        ax4title = (r'diff $\phi$')
        ax6title = (r'diff $\omega$')
    else:
        cmap0 = 'Blues'
        cmap1 = 'Reds_r'
        cmap2 = 'hot'
        ax2title = (r'$n$')
        ax4title = (r'$\phi$')
        ax6title = (r'$\omega$')
                
    velx = ax1.pcolormesh(U, cmap=cmap0, edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Umin, vmax=Umax)
    fig.colorbar(velx, ax=ax1)
    ax1.title.set_text(ax2title)
    ax1.set_aspect(1)

    vely = ax3.pcolormesh(V, cmap=cmap1, edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Vmin, vmax=Vmax)
    fig.colorbar(vely, ax=ax3)
    ax3.title.set_text(ax4title)
    ax3.set_aspect(1)

    pres = ax5.pcolormesh(P, cmap=cmap2, edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Pmin, vmax=Pmax)
    fig.colorbar(pres, ax=ax5)
    ax5.title.set_text(ax6title)
    ax5.set_aspect(1)



    colors = plt.cm.jet(np.linspace(10,1,21))
    lineColor = colors[0]
    if ("4" in filename):
        lineColor = colors[0]
    if ("8" in filename):
        lineColor = colors[1]
    if ("16" in filename):
        lineColor = colors[2]
    if ("32" in filename):
        lineColor = colors[3]
    if ("64" in filename):
        lineColor = colors[4]
    if ("128" in filename):
        lineColor = colors[5]
    if ("256" in filename):
        lineColor = colors[6]
    if ("512" in filename):
        lineColor = colors[7]
    if ("1024" in filename):
        lineColor = colors[8]
    if ("2048" in filename):
        lineColor = colors[9]
    if ("4096" in filename):
        lineColor = colors[10]


    #-------- plot centerlines
    x = list(range(N))
    hdim = N//2
    yU = U[hdim,:]
    yV = V[hdim,:]
    yP = P[hdim,:]

    velx = ax2.plot(x, yU, color=lineColor)
    ax2.set_ylim([Umin, Umax])
    ax2.title.set_text(ax2title)

    vely = ax4.plot(x, yV, color=lineColor)
    ax4.set_ylim([Vmin, Vmax])
    ax4.title.set_text(ax4title)

    pres = ax6.plot(x, yP, color=lineColor)
    ax6.set_ylim([Pmin, Pmax])
    ax6.title.set_text(ax6title)


    # save images
    plt.suptitle(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


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

        filename = "../../../../../StylES/bout_interfaces/results_bout/plots/plots_time" + str(t).zfill(5) + ".png"
        print_fields(Img_n, Img_phi, Img_vort, filename, Umin=min_U, Umax=max_U, Vmin=min_V, Vmax=max_V, Pmin=min_P, Pmax=max_P)

        gradV_phi = np.sqrt(((cr(Img_phi, 1, 0) - cr(Img_phi, -1, 0))/(2.0*delx))**2 + ((cr(Img_phi, 0, 1) - cr(Img_phi, 0, -1))/(2.0*dely))**2)

        time[t] = t
        Energy[t] = 0.5*L**2*np.sum(Img_n**2 + gradV_phi**2)
        
        closePlot=False
        if (t%50==0):
            if (t==FTIME-1):
                closePlot=True
            filename = "../../../../../StylES/bout_interfaces/results_bout/energy/Spectrum_" + str(t).zfill(4) + ".png"
            plot_spectrum(Img_n, gradV_phi, L, filename, close=closePlot)                
        
        # print("min/max", np.min(Img_n), np.max(Img_n), np.min(Img_phi), np.max(Img_phi), np.min(Img_vort), np.max(Img_vort))
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
    for i,file in enumerate(sorted(files)):
        if (i%SKIP==0):
            filename  = PATH_NUMPY + file
            data      = np.load(filename)
            Img_n     = np.cast[DTYPE](data['U'])
            Img_phi     = np.cast[DTYPE](data['V'])
            Img_vort     = np.cast[DTYPE](data['P'])
            file_dest = file.replace(".npz",".png")
            filename  = "./results_bout/plots/" + file_dest
            print_fields(Img_n, Img_phi, Img_vort, filename, diff=False, Umin=min_U, Umax=max_U, Vmin=min_V, Vmax=max_V, Pmin=min_P, Pmax=max_P)
            print ("done for file " + file_dest)



# plot energy
if (MODE=='READ_NETCDF'):
    plt.plot(time, Energy)
    plt.savefig('./results_bout/energy/energy_vs_time.png')
    plt.close()



#----------------------------- make animation
anim_file = 'animation.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('results_bout/plots/*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)
