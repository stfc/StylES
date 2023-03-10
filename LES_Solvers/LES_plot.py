import matplotlib.pyplot as plt

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

from LES_functions  import *

from testcases.HIT_2D.HIT_2D import *


def print_fields(U_, V_, P_, W_, N, filename, \
    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None, \
    Wmin=None, Wmax=None, C_=None, Cmin=None, Cmax=None):

    #---------------------------------- convert to numpy arrays
    U = convert(U_)
    V = convert(V_)
    P = convert(P_)
    W = convert(W_)

    if (PASSIVE):
        C = convert(C_)

    N = len(U[0,:])

    #---------------------------------- plot surfaces
    if (PASSIVE):
        fig, axs = plt.subplots(2, 5, figsize=(20,10))
        fig.subplots_adjust(hspace=0.25)
    else:
        fig, axs = plt.subplots(2, 4, figsize=(20,10))
        fig.subplots_adjust(hspace=0.25)

    ax1 = axs[0,0]
    ax2 = axs[1,0]
    ax3 = axs[0,1]
    ax4 = axs[1,1]
    ax5 = axs[0,2]
    ax6 = axs[1,2]
    ax7 = axs[0,3]
    ax8 = axs[1,3]
    if (PASSIVE):
        ax9  = axs[0,4]
        ax10 = axs[1,4]


    velx = ax1.pcolormesh(U, cmap='Blues', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Umin, vmax=Umax)
    fig.colorbar(velx, ax=ax1)
    ax1.title.set_text('X-vel')
    ax1.set_aspect(1)

    vely = ax3.pcolormesh(V, cmap='Reds_r', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Vmin, vmax=Vmax)
    fig.colorbar(vely, ax=ax3)
    ax3.title.set_text('Y-vel')
    ax3.set_aspect(1)

    pres = ax5.pcolormesh(P, cmap='RdBu', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Pmin, vmax=Pmax)
    fig.colorbar(pres, ax=ax5)
    ax5.title.set_text('pressure')
    ax5.set_aspect(1)

    vort = ax7.pcolormesh(W, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Wmin, vmax=Wmax)
    fig.colorbar(vort, ax=ax7)
    ax7.title.set_text('vorticity')
    ax7.set_aspect(1)

    if (PASSIVE):
        scal = ax9.pcolormesh(C, cmap='BuPu', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Cmin, vmax=Cmax)
        fig.colorbar(scal, ax=ax9)
        ax9.title.set_text('scalar')
        ax9.set_aspect(1)


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
    if (dir==0):    # x-direction
        x = list(range(N))
        hdim = N//2
        yU = U[hdim,:]
        yV = V[hdim,:]
        yP = P[hdim,:]
        yW = W[hdim,:]
    elif (dir==1):  # y-direction 
        x = list(range(N))
        hdim = N//2
        yU = U[:,hdim]
        yV = V[:,hdim]
        yP = P[:,hdim]
        yW = W[:,hdim]

    if (PASSIVE):
        if (dir==0):    # x-direction
            yC = C[hdim,:]
        elif (dir==1):  # y-direction 
            yC = C[:,hdim]


    velx = ax2.plot(x, yU, color=lineColor)
    ax2.set_ylim([Umin, Umax])
    ax2.title.set_text('X-vel')

    vely = ax4.plot(x, yV, color=lineColor)
    ax4.set_ylim([Vmin, Vmax])
    ax4.title.set_text('Y-vel')

    pres = ax6.plot(x, yP, color=lineColor)
    ax6.set_ylim([Pmin, Pmax])
    ax6.title.set_text('pressure')

    vort = ax8.plot(x, yW, color=lineColor)
    ax8.set_ylim([Wmin, Wmax])
    ax8.title.set_text('vorticity')

    if (PASSIVE):
        scal = ax10.plot(x, yC, color=lineColor)
        ax10.set_ylim([Cmin, Cmax])
        ax10.title.set_text('scalar')

    # save images
    plt.suptitle(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)    
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
            img[:,:,2] = (img[:,:,2] - minW)/(maxW - minW + small)

        img = Image.fromarray(np.uint8(img*255), 'RGB')
        size = N, N
        img.thumbnail(size)
        filename = filename.replace("plots","uvw")
        filename = filename.replace("Plots","Uvw")
        img.save(filename)







def print_fields_3(U_, V_, P_, N, filename, testcase='HIT_2D', \
    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None):

    if (testcase=='HIT_2D'):
        labelR = r'$u$'
        labelG = r'$v$'
        labelB = r'$\omega$'

    if (testcase=='HW' or testcase=='mHW'):
        labelR = r'$n$'
        labelG = r'$\phi$'
        labelB = r'$\zeta$'

    #---------------------------------- convert to numpy arrays
    U = convert(U_)
    V = convert(V_)
    P = convert(P_)

    N = len(U[0,:])

    #---------------------------------- plot surfaces
    fig, axs = plt.subplots(2, 3, figsize=(20,10))
    fig.subplots_adjust(hspace=0.25)

    ax1 = axs[0,0]
    ax2 = axs[1,0]
    ax3 = axs[0,1]
    ax4 = axs[1,1]
    ax5 = axs[0,2]
    ax6 = axs[1,2]

    velx = ax1.pcolormesh(U, cmap='Blues', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Umin, vmax=Umax)
    fig.colorbar(velx, ax=ax1)
    ax1.title.set_text(labelR)
    ax1.set_aspect(1)

    vely = ax3.pcolormesh(V, cmap='Reds_r', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Vmin, vmax=Vmax)
    fig.colorbar(vely, ax=ax3)
    ax3.title.set_text(labelG)
    ax3.set_aspect(1)

    pres = ax5.pcolormesh(P, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Pmin, vmax=Pmax)
    fig.colorbar(pres, ax=ax5)
    ax5.title.set_text(labelB)
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
    if (dir==0):    # x-direction
        x = list(range(N))
        hdim = N//2
        yU = U[hdim,:]
        yV = V[hdim,:]
        yP = P[hdim,:]
    elif (dir==1):  # y-direction 
        x = list(range(N))
        hdim = N//2
        yU = U[:,hdim]
        yV = V[:,hdim]
        yP = P[:,hdim]

    velx = ax2.plot(x, yU, color=lineColor)
    ax2.set_ylim([Umin, Umax])
    ax2.title.set_text(labelR)

    vely = ax4.plot(x, yV, color=lineColor)
    ax4.set_ylim([Vmin, Vmax])
    ax4.title.set_text(labelG)

    pres = ax6.plot(x, yP, color=lineColor)
    ax6.set_ylim([Pmin, Pmax])
    ax6.title.set_text(labelB)

    # save images
    plt.suptitle(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)    
    plt.close()









def print_fields_3_diff(U_, V_, P_, N, filename, testcase='HIT_2D', \
    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None):

    
    labelR = r'DNS'
    labelG = r'StyLES'
    labelB = r'differences'

    #---------------------------------- convert to numpy arrays
    U = convert(U_)
    V = convert(V_)
    P = convert(P_)

    N = len(U[0,:])

    #---------------------------------- plot surfaces
    fig, axs = plt.subplots(1, 3, figsize=(20,10))
    fig.subplots_adjust(hspace=0.25)

    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]


    velx = ax1.pcolormesh(U, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Umin, vmax=Umax)
    ax1x = ax1.get_position().x0
    ax1y = ax1.get_position().y0
    ax1h = ax1.get_position().height    
    cax1 = fig.add_axes([ax1x-0.055, ax1y*2.4 , 0.02, ax1h*0.6])
    fig.colorbar(velx, cax=cax1, ax=ax1)
    ax1.title.set_text(labelR)
    ax1.set_aspect(1)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)


    vely = ax2.pcolormesh(V, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Vmin, vmax=Vmax)
    ax2.title.set_text(labelG)
    ax2.set_aspect(1)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    pres = ax3.pcolormesh(P, cmap='jet', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Pmin, vmax=Pmax)
    ax3x = ax3.get_position().x1
    ax3y = ax3.get_position().y0
    ax3h = ax3.get_position().height    
    cax3 = fig.add_axes([ax3x+0.005, ax3y*2.4 , 0.02, ax3h*0.6])
    fig.colorbar(pres, cax=cax3, ax=ax3)
    ax3.title.set_text(labelB)
    ax3.set_aspect(1)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax3.autoscale_view()

    # save images
    plt.suptitle(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)    
    plt.close()






def print_fields_4_diff(U_DNS_, U_LES_, U_, diff_, N, filename, testcase='HIT_2D', \
    Umin=None, Umax=None, Vmin=None, Vmax=None, Pmin=None, Pmax=None):

   
    labelR = r'DNS'
    labelL = r'LES'
    labelG = r'StyLES'
    labelB = r'differences'

    #---------------------------------- convert to numpy arrays
    U_DNS = convert(U_DNS_)
    U_LES = convert(U_LES_)
    U     = convert(U_)
    diff  = convert(diff_)    

    N = len(U[0,:])
    N_LES = len(U_LES[0,:])
    r = int(N/N_LES)

    #---------------------------------- plot surfaces
    fig, axs = plt.subplots(1, 4, figsize=(20,10), gridspec_kw={'width_ratios': [r, 1, r, r]})
    fig.subplots_adjust(hspace=0.25)

    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax4 = axs[3]

    velx = ax1.pcolormesh(U_DNS, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Umin, vmax=Umax)
    #fig.colorbar(velx, ax=ax1)
    ax1.title.set_text(labelR)
    ax1.set_aspect(1)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax1.axis('off')

    vely = ax2.pcolormesh(U_LES, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Vmin, vmax=Vmax)
    #fig.colorbar(vely, ax=ax2)
    ax2.title.set_text(labelL)
    ax2.set_aspect(1)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax2.axis('off')

    vely = ax3.pcolormesh(U, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Vmin, vmax=Vmax)
    #fig.colorbar(vely, ax=ax3)
    ax3.title.set_text(labelG)
    ax3.set_aspect(1)
    ax3.axes.xaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax3.axis('off')

    pres = ax4.pcolormesh(diff, cmap='jet', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Pmin, vmax=Pmax)
    #fig.colorbar(pres, ax=ax4)
    ax4.title.set_text(labelB)
    ax4.set_aspect(1)
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    ax4.axis('off')

    # save images
    plt.suptitle(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)    
    plt.close()






def print_fields_2(U_LES_, U_, N, filename, testcase='HIT_2D', \
    Umin=None, Umax=None, Vmin=None, Vmax=None):

   
    labelL = r'LES'
    labelG = r'StyLES'

    #---------------------------------- convert to numpy arrays
    U_LES = convert(U_LES_)
    U     = convert(U_)

    N = len(U[0,:])
    N_LES = len(U_LES[0,:])
    r = int(N/N_LES)

    #---------------------------------- plot surfaces
    fig, axs = plt.subplots(1, 2, figsize=(20,10), gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.25)

    ax1 = axs[0]
    ax2 = axs[1]

    velx = ax1.pcolormesh(U_LES, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud')
    #fig.colorbar(velx, ax=ax1)
    ax1.title.set_text(labelL)
    ax1.set_aspect(1)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax1.axis('off')

    vely = ax2.pcolormesh(U, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud')
    #fig.colorbar(vely, ax=ax2)
    ax2.title.set_text(labelG)
    ax2.set_aspect(1)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax2.axis('off')

    # save images
    plt.suptitle(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)    
    plt.close()




def print_fields_1(W_, filename, Wmin=None, Wmax=None, legend=True):
    
    #---------------------------------- find vorticity
    W = convert(W_)


    #---------------------------------- plot surfaces
    fig, ax1 = plt.subplots(1, 1, figsize=(20,10))
    fig.subplots_adjust(hspace=0.25)

    vort = ax1.pcolormesh(W, cmap='hot', edgecolors='k', linewidths=0.1, shading='gouraud', vmin=Wmin, vmax=Wmax)
    ax1.set_aspect(1)
    if (legend):
        fig.colorbar(vort, ax=ax1)
        ax1.title.set_text(r'\omega')
        plt.suptitle(filename)
    else:
            ax1.axis("off")

    # save images
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)    
    plt.close()



