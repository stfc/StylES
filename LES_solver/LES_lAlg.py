from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

from LES_functions  import *



def solver_TDMA(a, bb, c, dd, N):

    b = nc.zeros([N,N], dtype=DTYPE)  # aw coefficient for TDMA
    d = nc.zeros([N,N], dtype=DTYPE)  # aw coefficient for TDMA

    for i in range(N):
        b[:,i] = bb[:,i]
        d[:,i] = dd[:,i]

    for i in range(1, N):
        m = a[:,i]/b[:,i-1]
        b[:,i] = b[:,i] - m*c[:,i-1] 
        d[:,i] = d[:,i] - m*d[:,i-1]
        	    
    x = b
    x[:,N-1] = d[:,N-1]/b[:,N-1]

    for i in range(N-2, -1, -1):
        x[:,i] = (d[:,i]-c[:,i]*x[:,i+1])/b[:,i]

    return x



def solver_TDMAcyclic(a, b, c, r, n):
    
    bb = nc.zeros([n, n], dtype=DTYPE)  # aw coefficient for TDMA
    u  = nc.zeros([n, n], dtype=DTYPE)  # aw coefficient for TDMA

    alpha = c[:,n-1]
    beta  = a[:,0]

    gamma = -b[:,0]   # avoid substraction error in forming bb[0]
    bb[:,0] = b[:,0] - gamma   # set up diagonal of the modified tridiagonal system
    bb[:,n-1] = b[:,n-1] - alpha*beta/gamma
    for i in range(1, n-1):
        bb[:,i] = b[:,i]

    # solve A*x = r
    x = solver_TDMA(a, bb, c, r, n)

    # setup vector u
    u[:,0] = gamma
    u[:,n-1] = alpha
    for i in range(1, n-1):
        u[:,i] = zero

    # solve A*z = u
    z = solver_TDMA(a, bb, c, u, n)

    # form v*x/(1+v*z)
    fact = (x[:,0]+beta*x[:,n-1]/gamma)/(one + z[:,0] + beta*z[:,n-1]/gamma)

    # get solution x
    for i in range(n):
        x[:,i] = x[:,i] - fact*z[:,i]

    return x





