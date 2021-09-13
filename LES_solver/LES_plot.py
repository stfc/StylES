import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from LES_parameters import *

def save_fields(U_, V_, P_, C_, it, dir=0):

    # find vorticity
    W_ = np.zeros([Nx+2,Ny+2], dtype=DTYPE)  # passive scalar
    W_ = (cr(V_, 1, 0) - cr(V_, -1, 0))/dXY - (cr(U_, 0, 1) - cr(U_, 0, -1))/dXY

    U = cp.asnumpy(U_)
    V = cp.asnumpy(V_)
    P = cp.asnumpy(P_)
    C = cp.asnumpy(C_)
    W = cp.asnumpy(W_)



    # plot surfaces
    fig, axs = plt.subplots(2, 4, figsize=(20,10))
    fig.subplots_adjust(hspace=0.25)

    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[0,2]
    ax4 = axs[0,3]
    ax5 = axs[1,0]
    ax6 = axs[1,1]
    ax7 = axs[1,2]
    ax8 = axs[1,3]


    U = np.transpose(U)
    velx = ax1.pcolormesh(U, cmap='Blues', edgecolors='k', linewidths=0.1, shading='gouraud')
    fig.colorbar(velx, ax=ax1)
    ax1.title.set_text('X-vel')
    ax1.set_aspect(1)

    V = np.transpose(V)
    vely = ax2.pcolormesh(V, cmap='Reds_r', edgecolors='k', linewidths=0.1, shading='gouraud')
    fig.colorbar(vely, ax=ax2)
    ax2.title.set_text('Y-vel')
    ax2.set_aspect(1)

    P = np.transpose(P)
    pres = ax3.pcolormesh(P, cmap='RdBu', edgecolors='k', linewidths=0.1, shading='gouraud')
    fig.colorbar(pres, ax=ax3)
    ax3.title.set_text('pressure')
    ax3.set_aspect(1)

    if (PASSIVE):
        C = np.transpose(C)
        scal = ax4.pcolormesh(C, cmap='BuPu', edgecolors='k', linewidths=0.1, shading='gouraud')
        fig.colorbar(scal, ax=ax4)
        ax4.title.set_text('scalar')
        ax4.set_aspect(1)
    else:
        W = np.transpose(W)
        vort = ax4.pcolormesh(W, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud')
        fig.colorbar(vort, ax=ax4)
        ax4.title.set_text('vorticity')
        ax4.set_aspect(1)



    # plot centerlines
    U = np.transpose(U)
    V = np.transpose(V)
    P = np.transpose(P)
    C = np.transpose(C)
    W = np.transpose(W)

    if (dir==0):    # x-direction
        x = list(range(Ny))
        hdim = Nx//2
        yU = U[hdim,:]
        yV = V[hdim,:]
        yP = P[hdim,:]
        yC = C[hdim,:]
        yW = W[hdim,:]
    elif (dir==1):  # y-direction 
        x = list(range(Nx))
        hdim = Ny//2
        yU = U[:,hdim]
        yV = V[:,hdim]
        yP = P[:,hdim]
        yC = C[:,hdim]
        yW = W[:,hdim]

    velx = ax5.plot(x,yU)
    ax5.title.set_text('X-vel')

    vely = ax6.plot(x,yV)
    ax6.title.set_text('Y-vel')

    pres = ax7.plot(x,yP)
    ax7.title.set_text('pressure')

    if (PASSIVE):
        scal = ax8.plot(x,yC)
        ax8.title.set_text('scalar')
    else:
        vort = ax8.plot(x,yW)
        ax8.title.set_text('vorticity')

    # save images
    plt.show()
    plt.savefig("it_{0:d}_fields.png".format(it), bbox_inches='tight', pad_inches=0)
