import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import imageio

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../../../codes/TurboGenPY/')

from tkespec import compute_tke_spectrum2d_3v
from isoturb import generate_isotropic_turbulence_2d



SAVE_UVW = False
DTYPE    = 'float32'
DIR      = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
STIME    = 202 # starting time to save fields
FTIME    = 300 # starting time to take as last image
ITIME    = 2  # skip between STIME, FTIME, ITIME
NDNS     = 100
DELT     = 1.0 
TSTART   = 0

file = open("HW_data/HW_0/data/BOUT.inp", 'r')
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

DX  = LX/NX
DY  = LY/NY
DZ  = LZ/NZ
NY2 = int(NY/2)-1

print("System sizes are: Lx,Ly,Lz,dx,dy,dz,nx,ny,nz =", LX,LY,LZ,DX,DY,DZ,NX,NY,NZ)


useLogSca = True
xLogLim    = [1.0e-2, 100]   # to do: to make nmore general
yLogLim    = [1.e-10, 10.]
xLinLim    = [0.0e0, 600]
yLinLim    = [0.0e0, 1.0]

ttime  = []
Energy = []


def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def convert(x):
    return x


def print_fields(U_, V_, P_, N, filename, \
    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, \
    Wmin=None, Wmax=None, C_=None, Cmin=None, Cmax=None):

    #---------------------------------- convert to numpy arrays
    U = convert(U_)
    V = convert(V_)
    P = convert(P_)

    if (DIR==1):    # plot along vertical
        U = np.transpose(U, axes=[1,0])
        V = np.transpose(V, axes=[1,0])
        P = np.transpose(P, axes=[1,0])

    N = len(U[0,:])

    #---------------------------------- plot surfaces
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    fig.subplots_adjust(hspace=0.25)

    ax1 = axs[0,0]
    ax2 = axs[1,0]
    ax3 = axs[0,1]
    ax4 = axs[1,1]
    ax5 = axs[0,2]
    ax6 = axs[1,2]


    velx = ax1.pcolormesh(U, cmap='Blues', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Umin, vmax=Umax)
    fig.colorbar(velx, ax=ax1)
    ax1.title.set_text('n')
    ax1.set_aspect(0.5)

    vely = ax3.pcolormesh(V, cmap='Reds_r', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Vmin, vmax=Vmax)
    fig.colorbar(vely, ax=ax3)
    ax3.title.set_text('phi')
    ax3.set_aspect(0.5)

    pres = ax5.pcolormesh(P, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Pmin, vmax=Pmax)
    fig.colorbar(pres, ax=ax5)
    ax5.title.set_text('vorticity')
    ax5.set_aspect(0.5)



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


    #---------------------------------- plot centerlines
    x = list(range(N))
    hdim = N//2
    yU = U[hdim,:]
    yV = V[hdim,:]
    yP = P[hdim,:]

    velx = ax2.plot(x, yU, color=lineColor)
    ax2.set_ylim([Umin, Umax])
    ax2.title.set_text('n')

    vely = ax4.plot(x, yV, color=lineColor)
    ax4.set_ylim([Vmin, Vmax])
    ax4.title.set_text('phi')

    pres = ax6.plot(x, yP, color=lineColor)
    ax6.set_ylim([Pmin, Pmax])
    ax6.title.set_text('vorticity')


    # save images
    plt.suptitle(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()





def plot_spectrum_2d_3v(U, V, L, filename, close=True, label=None):
    U_cpu = convert(U)
    V_cpu = convert(V)

    knyquist, wave_numbers, tke_spectrum = compute_tke_spectrum2d_3v(U_cpu, V_cpu, L, L, True)

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
    
    


def save_fields(totTime, U, V, P, filename="restart.npz"):

    # save restart file
    np.savez(filename, t=totTime, U=U, V=V, P=P)


# create folders fields and paths
path = "energy"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
else:
    cmd = "rm energy/*"
    os.system(cmd)
    
path = "plots"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
else:
    cmd = "rm plots/*"
    os.system(cmd)


# run on data
nMax = []
nMin = []
phiMax = []
phiMin = []
vortMax = []
vortMin = []
for nrun in range(NDNS):
    newfolder = "HW_data/HW_" + str(nrun) + "/data/"
    file_path = newfolder + "BOUT.log.0"
    with open(file_path, 'r') as file:
        content = file.read()
        if "finished" in content:
            
            os.chdir(newfolder)
            print("reading " + newfolder)
            
            for t in range(STIME,FTIME,ITIME):

#                Img_n    = collect("n",    yind=Y_COORD, tind=t, xguards=False, info=False)
#                Img_phi  = collect("phi",  yind=Y_COORD, tind=t, xguards=False, info=False)
#                Img_vort = collect("vort", yind=Y_COORD, tind=t, xguards=False, info=False)

                Img_n    = collect("n",    tind=t, xguards=False, info=False)
                Img_phi  = collect("phi",  tind=t, xguards=False, info=False)
                Img_vort = collect("vort", tind=t, xguards=False, info=False)

                Img_n = Img_n[0,:,8,:]
                nMax.append(np.max(Img_n))
                nMin.append(np.min(Img_n))

                Img_phi = Img_phi[0,:,8,:]
                phiMax.append(np.max(Img_phi))
                phiMin.append(np.min(Img_phi))

                Img_vort = Img_vort[0,:,8,:]
                vortMax.append(np.max(Img_vort))
                vortMin.append(np.min(Img_vort))

                # print_fields(Img_n, Img_phi, Img_vort, N, "../../../plots/plots_run" + str(nrun) + "_time" + str(t).zfill(4) + ".png", \
                #                Umin=-1, Umax=1, Vmin=-1, Vmax=1, Pmin=-1, Pmax=1)

                gradV_phi = np.sqrt(((cr(Img_phi, 1, 0) - cr(Img_phi, -1, 0))/(2.0*DX))**2 + ((cr(Img_phi, 0, 1) - cr(Img_phi, 0, -1))/(2.0*DY))**2)

                ttime.append(t*DELT + TSTART)
                E = 0.5*LX**2*np.sum(Img_n**2 + gradV_phi**2)
                Energy.append(E)
                
                #closePlot=False
                #if (t%100==0 or (t+ITIME>FTIME-1)):
                #    if (t+ITIME>FTIME-1):
                #        closePlot=True
                #    filename = "../../../energy/Spectrum_" + str(t).zfill(4) + ".png"
                #    plot_spectrum_2d_3v(Img_n, gradV_phi, LX, filename, close=closePlot)                
                
                print("done for file time step", t)
                # print("min/max", minN, minPhi, minVort, maxN, maxPhi, maxVort)
                # print("average", t, np.mean(Img_n), np.mean(Img_phi), np.mean(Img_vort))

            os.chdir("../../../")


# plot min/max
np.savetxt("energy_vs_time.txt", np.c_[ttime, Energy], fmt='%1.4e')
plt.plot(ttime, Energy)
plt.savefig('energy_vs_time.png')
plt.close()

np.savetxt("minmax_vs_time.txt", np.c_[ttime, nMax, nMin, phiMax, phiMin, vortMax, vortMin], fmt='%1.4e')
plt.plot(ttime, nMax,    label='nMax')
plt.plot(ttime, nMin,    label='nMax')
plt.plot(ttime, phiMax,  label='phiMax')
plt.plot(ttime, phiMin,  label='phiMax')
plt.plot(ttime, vortMax, label='vortMax')
plt.plot(ttime, vortMin, label='vortMax')
plt.legend()
plt.savefig('minmax_vs_time.png')


# make animation
anim_file = './animation.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./plots/*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        print(filename)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)
