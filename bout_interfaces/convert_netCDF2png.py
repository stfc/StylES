import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect


MODE       = 'READ_NUMPY'   #'READ_NUMPY', 'MAKE_ANIMATION', 'READ_NETCDF'
PATH_NUMPY = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/results_bout/plots/DNS/"
SAVE_UVW   = False
DTYPE      = 'float32'
DIR        = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
STIME      = 0 # starting time to take as first image
FTIME      = 101 # starting time to take as last image
ITIME      = 1  # skip between STIME, FTIME, ITIME
NDNS       = 1

os.system("rm -rf results_bout/plots/*")
os.system("rm -rf results_bout/fields/*")


def convert(x):
    return x


def print_fields(U_, V_, P_, filename, \
    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, \
    Wmin=None, Wmax=None, C_=None, Cmin=None, Cmax=None):
    # Umin=-10.0, Umax=10.0, Vmin=-10.0, Vmax=10.0, Pmin=-10.0, Pmax=10.0, \
    # Wmin=None, Wmax=None, C_=None, Cmin=None, Cmax=None):

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
    ax1.set_aspect(1)

    vely = ax3.pcolormesh(V, cmap='Reds_r', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Vmin, vmax=Vmax)
    fig.colorbar(vely, ax=ax3)
    ax3.title.set_text('phi')
    ax3.set_aspect(1)

    pres = ax5.pcolormesh(P, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Pmin, vmax=Pmax)
    fig.colorbar(pres, ax=ax5)
    ax5.title.set_text('vorticity')
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



    #--------------------------------------- save combined images uvw
    if (SAVE_UVW):
        img = np.zeros([N, N, 3], dtype=DTYPE)
        img[:,:,0] = convert(U[:,:])
        img[:,:,1] = convert(V[:,:])
        img[:,:,2] = convert(P[:,:])

        # normalize velocity
        maxU = np.max(img[:,:,0])
        maxV = np.max(img[:,:,1])
        minU = np.min(img[:,:,0])
        minV = np.min(img[:,:,1])
        maxVel = max(maxU, maxV)
        minVel = min(minU, minV)
        if (maxVel!=minVel):
            img[:,:,0] = (img[:,:,0] - minVel)/(maxVel - minVel)
            img[:,:,1] = (img[:,:,1] - minVel)/(maxVel - minVel)

        # normalize vorticity
        maxW = np.max(img[:,:,2])
        minW = np.min(img[:,:,2])
        if (maxW!=minW):
            img[:,:,2] = (img[:,:,2] - minW)/(maxW - minW)

        img = Image.fromarray(np.uint8(img*255), 'RGB')
        size = N, N
        img.thumbnail(size)
        filename = filename.replace("plots","uvw")
        filename = filename.replace("Plots","Uvw")
        img.save(filename)




def save_fields(totTime, U, V, P, filename="restart.npz"):

    # save restart file
    np.savez(filename, t=totTime, U=U, V=V, P=P)


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

    # run on data
    for nrun in range(NDNS):
        newfolder = "../../BOUT-dev/build_release/examples/hasegawa-wakatani/data/"

        os.chdir(newfolder)
        print("reading " + newfolder)
        
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
            
            # save_fields(t, Img_n, Img_phi, Img_vort, "../../../../../StylES/bout_interfaces/results_bout/fields/fields_run" + str(nrun) + "_time" + str(t).zfill(3) + ".npz")

            print_fields(Img_n, Img_phi, Img_vort, "../../../../../StylES/bout_interfaces/results_bout/plots/plots_run" + str(nrun) + "_time" + str(t).zfill(3) + ".png")

            print("done for file time step", t)

        os.chdir("../../../../../StylES/bout_interfaces/")
    
elif (MODE=='READ_NUMPY'):

    files = os.listdir(PATH_NUMPY)
    nfiles = len(files)
    for i,file in enumerate(sorted(files)):
        filename  = PATH_NUMPY + file
        data      = np.load(filename)
        Img_n     = np.cast[DTYPE](data['U'])
        Img_phi   = np.cast[DTYPE](data['V'])
        Img_vort  = np.cast[DTYPE](data['P'])
        file_dest = file.replace(".npz",".png")
        filename = "./results_bout/plots/" + file_dest
        print_fields(Img_n, Img_phi, Img_vort, filename)
        print ("done for file " + file_dest)



# make animation
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
