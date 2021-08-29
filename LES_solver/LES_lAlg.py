from LES_parameters import toll, maxIt, Nx, Ny


def TDMAsolver(a, b, c, d):
    #TDMA solver, a b c d can be NumPy array type or Python list type.
    #refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    #and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in xrange(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in xrange(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc


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