from LES_parameters import *

def apply_BCs(U, V, P, BCs=[0, 0, 0, 0]):


    #------------------------- west side
    if (BCs[0] == 0):    # default: periodic conditions
        for j in range(Ny+2):
            U[0][j] = U[Nx][j]
            V[0][j] = V[Nx][j]
            P[0][j] = P[Nx][j]
    elif (BCs[0] == 1):    # no-slip condition
        for j in range(Ny+2):
            U[1][j] = zero
            V[1][j] = zero


    #------------------------- east side
    if (BCs[1] == 0):    # default: periodic conditions
        for j in range(Ny+2):
            U[Nx+1][j] = U[1][j]
            V[Nx+1][j] = V[1][j]
            P[Nx+1][j] = P[1][j]
    elif (BCs[1] == 1):    # no-slip condition
        for j in range(Ny+2):
            U[Nx][j] = zero
            V[Nx][j] = zero


    #------------------------- south side
    if (BCs[2] == 0):    # default: periodic conditions
        for i in range(Nx+2):
            U[i][0] = U[i][Ny]
            V[i][0] = V[i][Ny]
            P[i][0] = P[i][Ny]
    elif (BCs[2] == 1):    # no-slip condition
        for i in range(Nx+2):
            U[i][1] = zero
            V[i][1] = zero


    #------------------------- north side
    if (BCs[3] == 0):    # default: periodic conditions
        for i in range(Nx+2):
            U[i][Ny+1] = U[i][1]
            V[i][Ny+1] = V[i][1]
            P[i][Ny+1] = P[i][1]
    elif (BCs[3] == 1):    # no-slip condition
        for i in range(Nx+2):
            U[i][Ny] = zero
            V[i][Ny] = zero



