from LES_constants import *
from LES_parameters import *
from input import BCs, Uin, Vin, Pin, Pout



def apply_BCs(U, V, P, C, Pc, Ue, Vn):


    #------------------------- west side
    if (BCs[0] == 0):              # default: periodic conditions
        for j in range(Ny+2):
            U[0][j] = U[Nx][j]
            V[0][j] = V[Nx][j]
            P[0][j] = P[Nx][j]
            C[0][j] = C[Nx][j]
            Pc[0][j] = Pc[Nx][j]
            Ue[0][j] = Ue[Nx][j]
            Vn[0][j] = Vn[Nx][j]


    #------------------------- east side
    if (BCs[1] == 0):              # default: periodic conditions
        for j in range(Ny+2):
            U[Nx+1][j] = U[1][j]
            V[Nx+1][j] = V[1][j]
            P[Nx+1][j] = P[1][j]
            C[Nx+1][j] = C[1][j]
            Pc[Nx+1][j] = Pc[1][j]
            Ue[Nx+1][j] = Ue[1][j]
            Vn[Nx+1][j] = Vn[1][j]


    #------------------------- south side
    if (BCs[2] == 0):              # default: periodic conditions
        for i in range(Nx+2):
            U[i][0] = U[i][Ny]
            V[i][0] = V[i][Ny]
            P[i][0] = P[i][Ny]
            C[i][0] = C[i][Ny]
            Pc[i][0] = Pc[i][Ny]
            Ue[i][0] = Ue[i][Ny]
            Vn[i][0] = Vn[i][Ny]


    #------------------------- north side
    if (BCs[3] == 0):              # default: periodic conditions
        for i in range(Nx+2):
            U[i][Ny+1] = U[i][1]
            V[i][Ny+1] = V[i][1]
            P[i][Ny+1] = P[i][1]
            C[i][Ny+1] = C[i][1]
            Pc[i][Ny+1] = Pc[i][1]
            Ue[i][Ny+1] = Ue[i][1]
            Vn[i][Ny+1] = Vn[i][1]



