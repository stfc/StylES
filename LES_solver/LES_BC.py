from LES_constants import *
from LES_parameters import *

def apply_BCs(U, V, P, Ue, Vn, BCs, Uin, Vin, Pin, Pout):


    #------------------------- west side
    if   (BCs[0] == 0):              # default: periodic conditions
        for j in range(Ny+2):
            U[0][j] = U[Nx][j]
            V[0][j] = V[Nx][j]
            P[0][j] = P[Nx][j]
            Ue[0][j] = Ue[Nx][j]
            Vn[0][j] = Vn[Nx][j]

    elif (BCs[0] == 1): # no-slip (wall) condition
        for j in range(Ny+2):
            U[0][j] = zero
            V[0][j] = zero
            P[0][j] = P[1][j]
            Ue[0][j] = zero
            Vn[0][j] = zero

    elif (BCs[0] == 2):              # fixed inlet velocity
        for j in range(Ny+2):
            U[0][j] = Uin
            V[0][j] = Vin
            P[0][j] = P[1][j]
            Ue[0][j] = Uin
            Vn[0][j] = Vin

    elif (BCs[0] == 3):              # fixed inlet pressure
        for j in range(Ny+2):
            U[0][j] = U[1][j]
            V[0][j] = V[1][j]
            P[0][j] = Pin
            Ue[0][j] = Ue[1][j]
            Vn[0][j] = Vn[1][j]

    elif (BCs[0] == 4):              # fixed outlet pressure
        for j in range(Ny+2):
            U[0][j] = U[1][j]
            V[0][j] = V[1][j]
            P[0][j] = Pout
            Ue[0][j] = Ue[1][j]
            Vn[0][j] = Vn[1][j]


    #------------------------- east side
    if   (BCs[1] == 0):              # default: periodic conditions
        for j in range(Ny+2):
            U[Nx+1][j] = U[1][j]
            V[Nx+1][j] = V[1][j]
            P[Nx+1][j] = P[1][j]
            Ue[Nx+1][j] = Ue[1][j]
            Vn[Nx+1][j] = Vn[1][j]

    elif (BCs[1] == 1): # no-slip (wall) condition
        for j in range(Ny+2):
            U[Nx+1][j] = zero
            V[Nx+1][j] = zero
            P[Nx+1][j] = P[Nx][j]
            Ue[Nx+1][j] = zero
            Vn[Nx+1][j] = zero

    elif (BCs[1] == 2):              # fixed inlet velocity
        for j in range(Ny+2):
            U[Nx+1][j] = Uin
            V[Nx+1][j] = Vin
            P[Nx+1][j] = P[Nx][j]
            Ue[Nx+1][j] = Uin
            Vn[Nx+1][j] = Vin

    elif (BCs[1] == 3):              # fixed inlet pressure
        for j in range(Ny+2):
            U[Nx+1][j] = U[Nx][j]
            V[Nx+1][j] = V[Nx][j]
            P[Nx+1][j] = Pin
            Ue[Nx+1][j] = Ue[Nx][j]
            Vn[Nx+1][j] = Vn[Nx][j]

    elif (BCs[1] == 4):              # fixed outlet pressure
        for j in range(Ny+2):
            U[Nx+1][j] = U[Nx][j]
            V[Nx+1][j] = V[Nx][j]
            P[Nx+1][j] = Pout
            Ue[Nx+1][j] = Ue[Nx][j]
            Vn[Nx+1][j] = Vn[Nx][j]



    #------------------------- south side
    if   (BCs[2] == 0):              # default: periodic conditions
        for i in range(Nx+2):
            U[i][0] = U[i][Ny]
            V[i][0] = V[i][Ny]
            P[i][0] = P[i][Ny]
            Ue[i][0] = Ue[i][Ny]
            Vn[i][0] = Vn[i][Ny]

    elif (BCs[2] == 1): # no-slip (wall) condition
        for i in range(Nx+2):
            U[i][0] = zero
            V[i][0] = zero
            P[i][0] = P[i][1]
            Ue[i][0] = zero
            Vn[i][0] = zero

    elif (BCs[2] == 2):              # fixed inlet velocity
        for i in range(Nx+2):
            U[i][0] = Uin
            V[i][0] = Vin
            P[i][0] = P[i][1]
            Ue[i][0] = Uin
            Vn[i][0] = Vin

    elif (BCs[2] == 3):              # fixed inlet pressure
        for i in range(Nx+2):
            U[i][0] = U[i][1]
            V[i][0] = V[i][1]
            P[i][0] = P[i][1]
            Ue[i][0] = Ue[i][1]
            Vn[i][0] = Vn[i][1]

    elif (BCs[2] == 4):              # fixed outlet pressure
        for i in range(Nx+2):
            U[i][0] = U[i][1]
            V[i][0] = V[i][1]
            P[i][0] = Pout
            Ue[i][0] = Ue[i][1]
            Vn[i][0] = Vn[i][1]




    #------------------------- north side
    if   (BCs[3] == 0):              # default: periodic conditions
        for i in range(Nx+2):
            U[i][Ny+1] = U[i][1]
            V[i][Ny+1] = V[i][1]
            P[i][Ny+1] = P[i][1]
            Ue[i][Ny+1] = Ue[i][1]
            Vn[i][Ny+1] = Vn[i][1]

    elif (BCs[3] == 1): # no-slip (wall) condition
        for i in range(Nx+2):
            U[i][Ny+1] = zero
            V[i][Ny+1] = zero
            P[i][Ny+1] = P[i][Ny]
            Ue[i][Ny+1] = zero
            Vn[i][Ny+1] = zero

    elif (BCs[3] == 2):              # fixed inlet velocity
        for i in range(Nx+2):
            U[i][Ny+1] = Uin
            V[i][Ny+1] = Vin
            P[i][Ny+1] = P[i][Ny]
            Ue[i][Ny+1] = Uin
            Vn[i][Ny+1] = Vin

    elif (BCs[3] == 3):              # fixed inlet pressure
        for i in range(Nx+2):
            U[i][Ny+1] = U[i][Ny]
            V[i][Ny+1] = V[i][Ny]
            P[i][Ny+1] = Pin
            Ue[i][Ny+1] = Ue[i][Ny]
            Vn[i][Ny+1] = Vn[i][Ny]

    elif (BCs[3] == 4):              # fixed outlet pressure
        for i in range(Nx+2):
            U[i][Ny+1] = U[i][Ny]
            V[i][Ny+1] = V[i][Ny]
            P[i][Ny+1] = Pout
            Ue[i][Ny+1] = Ue[i][Ny]
            Vn[i][Ny+1] = Vn[i][Ny]




def findBCMomCoeff(i, j, U, V, P, Uin, Vin, Pin, Pout, rhoRef):

    ww = one
    ee = one
    ss = one
    nn = one
    rhsU = zero
    rhsV = zero

    # west side
    if (i==1):
        if   (BCs[0]==0):     # periodic
            ww = one
            rhsU = zero

        elif (BCs[0]==1):     # no-slip (wall)
            ww = zero
            rhsU = rhsU + hf*P[i,j]*deltaX   # Pw=Po

        elif (BCs[0]==2):     # fixed inlet velocity
            ww = zero
            rhsU = rhsU + (DX + max(Uin*rhoRef*hf,zero))*Uin + hf*P[i,j]*deltaY  # Uw=Uin and Pw=Po

        elif (BCs[0]==3):     # fixed inlet pressure
            ww = zero
            rhsU = rhsU + (DX + max(U[i][j]*rhoRef*hf,zero))*U[i][j] + hf*Pin*deltaY # Uw=Uo and Pw=Pin

        elif (BCs[0]==4):     # fixed outlet pressure
            ww = zero
            rhsU = rhsU + (max(U[i][j]*rhoRef*hf,zero))*U[i][j] + hf*Pout*deltaY  # Uw=Uo and Pw=Pout


    # east side
    if (i==Nx):
        if   (BCs[1]==0):     # periodic
            ee = one
            rhsU = zero

        elif (BCs[1]==1):     # no-slip (wall)
            ee = zero
            rhsU = rhsU - hf*P[i,j]*deltaX   # Pw=Po

        elif (BCs[1]==2):     # fixed inlet velocity
            ee = zero
            rhsU = rhsU + (DX + max(zero, -Uin*rhoRef*hf))*Uin - hf*P[i,j]*deltaY  # Uw=Uin and Pw=Po

        elif (BCs[1]==3):     # fixed inlet pressure
            ee = zero
            rhsU = rhsU + (DX + max(zero, -U[i][j]*rhoRef*hf))*U[i][j] - hf*Pin*deltaY # Uw=Uo and Pw=Pin

        elif (BCs[1]==4):     # fixed outlet pressure
            ee = zero
            rhsU = rhsU + (max(zero, -U[i][j]*rhoRef*hf))*U[i][j] - hf*Pout*deltaY  # Uw=Uo and Pw=Pout


    # south side
    if (j==1):
        if   (BCs[2]==0):     # periodic
            ss = one
            rhsV = zero

        elif (BCs[2]==1):     # no-slip (wall)
            ss = zero
            rhsV = rhsV + hf*P[i,j]*deltaX   # Pw=Po

        elif (BCs[2]==2):     # fixed inlet velocity
            ss = zero
            rhsV = rhsV + (DY + max(Vin*rhoRef*hf,zero))*Vin + hf*P[i,j]*deltaX  # Vs=Vin and Pw=Po

        elif (BCs[2]==3):     # fixed inlet pressure
            ss = zero
            rhsV = rhsV + (DY + max(V[i][j]*rhoRef*hf,zero))*V[i][j] + hf*Pin*deltaX # Vs=Vo and Pw=Pin

        elif (BCs[2]==4):     # fixed outlet pressure
            ss = zero
            rhsV = rhsV + (max(V[i][j]*rhoRef*hf,zero))*V[i][j] + hf*Pout*deltaX  # Vs=Vo and Pw=Pout


    # north side
    if (j==Ny):
        if   (BCs[3]==0):     # periodic
            nn = one
            rhsV = zero

        elif (BCs[3]==1):     # no-slip (wall)
            nn = zero
            rhsV = rhsV - hf*P[i,j]*deltaX   # Pw=Po

        elif (BCs[3]==2):     # fixed inlet velocity
            nn = zero
            rhsV = rhsV + (DY + max(Uin*rhoRef*hf,zero))*Vin - hf*P[i,j]*deltaX  # Vn=Vin and Pw=Po

        elif (BCs[3]==3):     # fixed inlet pressure
            nn = zero
            rhsV = rhsV + (DY + max(V[i][j]*rhoRef*hf,zero))*V[i][j] - hf*Pin*deltaX # Vn=Vo and Pw=Pin

        elif (BCs[3]==4):     # fixed outlet pressure
            nn = zero
            rhsV = rhsV + (max(V[i][j]*rhoRef*hf,zero))*V[i][j] - hf*Pout*deltaX  # Vn=Vo and Pw=Pout


    return ww, ee, ss, nn, rhsU, rhsV