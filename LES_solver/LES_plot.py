import numpy as np
import matplotlib.pyplot as plt

from LES_parameters import *

def save_fields(U, V, P, C, it, dir=0):

    # plot surfaces
    fig, axs = plt.subplots(2, 4, figsize=(20,5))
    fig.subplots_adjust(hspace=1)

    ax1 = axs[0][0]
    ax2 = axs[0][1]
    ax3 = axs[0][2]
    ax4 = axs[0][3]
    ax5 = axs[1][0]
    ax6 = axs[1][1]
    ax7 = axs[1][2]
    ax8 = axs[1][3]

    U = np.transpose(U)
    velx = ax1.pcolor(U, cmap='Blues', edgecolors='k', linewidths=0.1)
    fig.colorbar(velx, ax=ax1)
    ax1.title.set_text('X-vel')
    ax1.set_aspect(1)

    V = np.transpose(V)
    vely = ax2.pcolor(V, cmap='Reds_r', edgecolors='k', linewidths=0.1)
    fig.colorbar(vely, ax=ax2)
    ax2.title.set_text('Y-vel')
    ax2.set_aspect(1)

    P = np.transpose(P)
    pres = ax3.pcolor(P, cmap='RdBu', edgecolors='k', linewidths=0.1)
    fig.colorbar(pres, ax=ax3)
    ax3.title.set_text('pressure')
    ax3.set_aspect(1)

    C = np.transpose(C)
    scal = ax4.pcolor(C, cmap='BuPu', edgecolors='k', linewidths=0.1)
    fig.colorbar(scal, ax=ax4)
    ax4.title.set_text('scalar')
    ax4.set_aspect(1)


    # plot centerlines
    U = np.transpose(U)
    V = np.transpose(V)
    P = np.transpose(P)
    C = np.transpose(C)

    if (dir==0):    # x-direction
        x = list(range(-1, Ny+1))
        hdim = int((Nx+2)/2)
        yU = U[hdim,:]
        yV = V[hdim,:]
        yP = P[hdim,:]
        yC = C[hdim,:]
    elif (dir==1):  # y-direction 
        x = list(range(-1, Nx+1))
        hdim = int((Ny+2)/2)
        yU = U[:,hdim]
        yV = V[:,hdim]
        yP = P[:,hdim]
        yC = C[:,hdim]

    velx = ax5.plot(x,yU)
    ax5.title.set_text('X-vel')

    vely = ax6.plot(x,yV)
    ax6.title.set_text('Y-vel')

    pres = ax7.plot(x,yP)
    ax7.title.set_text('pressure')

    scal = ax8.plot(x,yC)
    ax8.title.set_text('scalar')


    # save images
    plt.show()
    plt.savefig("it_{0:d}_fields.png".format(it))
