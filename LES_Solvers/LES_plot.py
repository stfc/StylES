import matplotlib.pyplot as plt

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

from LES_functions  import *



def print_fields(U_, V_, P_, C_, it, dir=0):

    #---------------------------------- find vorticity
    W_ = nc.zeros([N,N], dtype=DTYPE)
    W_ = (cr(V_, 1, 0) - cr(V_, -1, 0))/dl - (cr(U_, 0, 1) - cr(U_, 0, -1))/dl

    U = convert(U_)
    V = convert(V_)
    P = convert(P_)
    C = convert(C_)
    W = convert(W_)



    #---------------------------------- plot surfaces
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


    velx = ax1.pcolormesh(U, cmap='Blues', edgecolors='k', linewidths=0.1, shading='gouraud')
    fig.colorbar(velx, ax=ax1)
    ax1.title.set_text('X-vel')
    ax1.set_aspect(1)

    vely = ax2.pcolormesh(V, cmap='Reds_r', edgecolors='k', linewidths=0.1, shading='gouraud')
    fig.colorbar(vely, ax=ax2)
    ax2.title.set_text('Y-vel')
    ax2.set_aspect(1)

    pres = ax3.pcolormesh(P, cmap='RdBu', edgecolors='k', linewidths=0.1, shading='gouraud')
    fig.colorbar(pres, ax=ax3)
    ax3.title.set_text('pressure')
    ax3.set_aspect(1)

    if (PASSIVE):
        scal = ax4.pcolormesh(C, cmap='BuPu', edgecolors='k', linewidths=0.1, shading='gouraud')
        fig.colorbar(scal, ax=ax4)
        ax4.title.set_text('scalar')
        ax4.set_aspect(1)
    else:
        vort = ax4.pcolormesh(W, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud')
        fig.colorbar(vort, ax=ax4)
        ax4.title.set_text('vorticity')
        ax4.set_aspect(1)



    #---------------------------------- plot centerlines
    if (dir==0):    # x-direction
        x = list(range(N))
        hdim = N//2
        yU = U[hdim,:]
        yV = V[hdim,:]
        yP = P[hdim,:]
        yC = C[hdim,:]
        yW = W[hdim,:]
    elif (dir==1):  # y-direction 
        x = list(range(N))
        hdim = N//2
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
    plt.savefig("fields_it_{0:d}.png".format(it), bbox_inches='tight', pad_inches=0)    
    plt.close()



    #--------------------------------------- save combined images uvw
    if (SAVE_UVW):
        img = np.zeros([N, N, 3], dtype=DTYPE)
        img[:,:,0] = convert(U[:,:])
        img[:,:,1] = convert(V[:,:])
        img[:,:,2] = convert(W[:,:])

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
        filename = "uvw_" + str(it) + ".png"
        size = N, N
        img.thumbnail(size)
        img.save(filename)
