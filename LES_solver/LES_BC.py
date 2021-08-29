from LES_parameters import *

def apply_BCs(phi, vel_field=False, BCs=[0, 0, 0, 0]):


    #------------------------- west side
    if (BCs[0] == 0):    # default: periodic conditions
        for j in range(Ny+2):
            phi[0][j] = phi[Nx][j]
    elif ((BCs[0] == 1) and vel_field):    # no-slip condition
        for j in range(Ny+2):
            phi[1][j] = zero


    #------------------------- east side
    if (BCs[1] == 0):    # default: periodic conditions
        for j in range(Ny+2):
            phi[Nx+1][j] = phi[1][j]
    elif ((BCs[1] == 1) and vel_field):    # no-slip condition
        for j in range(Ny+2):
            phi[Nx][j] = zero


    #------------------------- south side
    if (BCs[2] == 0):    # default: periodic conditions
        for i in range(Nx+2):
            phi[i][0] = phi[i][Ny]
    elif ((BCs[2] == 1) and vel_field):    # no-slip condition
        for i in range(Nx+2):
            phi[i][1] = zero


    #------------------------- north side
    if (BCs[3] == 0):    # default: periodic conditions
        for i in range(Nx+2):
            phi[i][Ny+1] = phi[i][1]
    elif ((BCs[3] == 1) and vel_field):    # no-slip condition
        for i in range(Nx+2):
            phi[i][Ny] = zero



