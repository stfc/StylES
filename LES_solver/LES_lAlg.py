import numpy as np

from LES_parameters import toll, maxIt, Nx, Ny


def TDMAsolver(a, b, c, d, N):

    for i in range(1, N):
        m = a[i]/b[i-1]
        b[i] = b[i] - m*c[i-1] 
        d[i] = d[i] - m*d[i-1]
        	    
    x = b
    x[N-1] = d[N-1]/b[N-1]

    for i in range(N-2, -1, -1):
        x[i] = (d[i]-c[i]*x[i+1])/b[i]

    return x


def solve_2D(phi, Aw, Ae, As, An, Ap, b):
     
    # solve full system as sequence of tridiagonal matrixs
    for i in range(Nx):
        for j in range(Ny):
            prevPhi[i][j] = phi[i][j]

    it=0
    res=0.0e0
    while (res>toll and it<maxIt):

        #  alternate according to the iteration
        if (it%2 == 0):
            for i in range(Nx):
                newPhi[i][:] = TDMA(As[i][:], Ap[i][:], An[i][:], b[i][:])
        else:
            for j in range(Ny):
                newPhi[:][j] = TDMA(Aw[:][j], Ap[:][j], Ae[:][j], b[:][j])


        # calculate residuals
        res=0.0e0
        for i in range(Nx):
            for j in range(Ny):
                res = res + abs(newPhi[i][j] - prevPhi[i][j])
                prevPhi[i][j] = newPhi[i][j]


        # adjust boundary source terms for next iteration
        if (BCs[0] == 0):   #periodicity in x-direction
            for j in range(Ny):
                b[0 ][j] = b[0 ][j] - Aw[0 ][j]*prevPhi[Nx][j] + Aw[0 ][j]*newPhi[Nx][j]
                b[Nx][j] = b[Nx][j] - Aw[Nx][j]*prevPhi[0 ][j] + Aw[Nx][j]*newPhi[0 ][j]

        if (BCs[2] == 0):   #periodicity in y-direction
            for i in range(Nx):
                b[i][0 ] = b[i][0 ] - Aw[i][0 ]*prevPhi[i][Ny] + Aw[i][0 ]*newPhi[i][Ny]
                b[i][Ny] = b[i][Ny] - Aw[i][Ny]*prevPhi[i][0 ] + Aw[i][Ny]*newPhi[i][0 ]

        it = it+1

    return newPhi