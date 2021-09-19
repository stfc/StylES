import os

from time import time
from PIL import Image
from math import sqrt

from LES_modules    import *
from LES_constants  import *
from LES_parameters import *

from LES_functions  import *
from LES_plot       import *
from LES_lAlg       import *



# start timing
tstart = time()



#---------------------------- define arrays
Uo = nc.zeros([N,N], dtype=DTYPE)   # old x-velocity
Vo = nc.zeros([N,N], dtype=DTYPE)   # old y-velocity
Po = nc.zeros([N,N], dtype=DTYPE)   # old pressure field
Co = nc.zeros([N,N], dtype=DTYPE)   # old passive scalar
Ue = nc.zeros([N,N], dtype=DTYPE)  # face x-velocities
Vn = nc.zeros([N,N], dtype=DTYPE)  # face y-velocities  
pc = nc.zeros([N,N], dtype=DTYPE)  # pressure correction
Z  = nc.zeros([N,N], dtype=DTYPE)




#---------------------------- set flow pressure, velocity fields and BCs
os.system("rm fields*.png")
os.system("rm Energy_spectrum.png")

# initial flow
totTime = zero
if (RESTART):
    U, V, P, C, B, totTime = load_fields()
else:
    U, V, P, C, B, totTime = init_fields()

print_fields(U, V, P, C, 0, dir)



# find face velocities first guess as forward difference (i.e. on side east and north)
Ue = hf*(cr(U, 1, 0) + U)
Vn = hf*(cr(V, 1, 0) + V)


#---------------------------- main time step loop
tstep    = 0
resM_cpu = zero
resP_cpu = zero
resC_cpu = zero
res_cpu  = zero
firstIt  = True
its      = 0

# check divergence
div = rho*dl*nc.sum(nc.abs(cr(Ue, -1, 0) - Ue + cr(Vn, 0, -1) - Vn))
div = div*iNN
div_cpu = convert(div)

# print values
tend = time()
if (tstep%print_res == 0):
    wtime = (tend-tstart)
    print("Wall time [s] {0:6.1f}  steps {1:3d}  time {2:5.2e}  delt {3:5.2e}  resM {4:5.2e}  "\
        "resP {5:5.2e}  resC {6:5.2e}  res {7:5.2e}  its {8:3d}  div {9:5.2e}"       \
    .format(wtime, tstep, totTime, delt, resM_cpu, resP_cpu, \
    resC_cpu, res_cpu, its, div_cpu))

# plot spectrum
plot_spectrum(U, V, L, tstep)


while (tstep<totSteps):


    # save old values of U, V and P
    Uo[:,:] = U[:,:]
    Vo[:,:] = V[:,:]
    Po[:,:] = P[:,:]
    if (PASSIVE):
        Co[:,:] = C[:,:]


    # start outer loop on SIMPLE convergence
    it = 0
    res = large
    while (res>toll and it<maxIt):



        #---------------------------- find Rhie-Chow interpolation (PWIM)
        if (not firstIt):
            deltpX1 = hf*(cr(P, 2, 0) - P)    
            deltpX2 = hf*(cr(P, 1, 0) - cr(P, -1, 0))
            deltpX3 = (P - cr(P,  1, 0))

            deltpY1 = hf*(cr(P, 0, 2) - P)
            deltpY2 = hf*(cr(P, 0, 1) - cr(P, 0, -1))
            deltpY3 = (P - cr(P, 0,  1))

            Ue = hf*(cr(U, 1, 0) + U)             \
               + hf*deltpX1*cr(iApM, 1, 0)*dl  \
               + hf*deltpX2*iApM*dl            \
               + hf*deltpX3*(cr(iApM, 1, 0) + iApM)*dl

            Vn = hf*(cr(V, 0, 1) + V)              \
               + hf*deltpY1*cr(iApM, 0, 1)*dl  \
               + hf*deltpY2*iApM*dl            \
               + hf*deltpY3*(cr(iApM, 0, 1) + iApM)*dl
    
        firstIt = False
               


        #---------------------------- solve momentum equations
        Fw = rho*cr(Ue, -1, 0)
        Fe = rho*Ue
        Fs = rho*cr(Vn, 0, -1)
        Fn = rho*Vn

        Aw = A*(Dc + hf*(nc.abs(Fw) + Fw))
        Ae = A*(Dc + hf*(nc.abs(Fe) - Fe))
        As = A*(Dc + hf*(nc.abs(Fs) + Fs))
        An = A*(Dc + hf*(nc.abs(Fn) - Fn))
        Ao = rho*A*dl/delt

        Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
        iApM = one/Ap

        itM  = 0
        resM = large
        while (resM>tollM and itM<maxIt):

            # x-direction
            sU = Ao*Uo - hf*(cr(P, 1, 0) - cr(P, -1, 0))*A + hf*(cr(B, 1, 0) + cr(B, -1, 0))
            dd = sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0)
            U = solver_TDMAcyclic(-As, Ap, -An, dd, N)
            U = (sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0) + As*cr(U, 0, -1) + An*cr(U, 0, 1))*iApM
            resMU = nc.sum(nc.abs(Ap*U - sU - Aw*cr(U, -1, 0) - Ae*cr(U, 1, 0) - As*cr(U, 0, -1) - An*cr(U, 0, 1)))

            # y-direction
            sV = Ao*Vo - hf*(cr(P, 0, 1) - cr(P, 0, -1))*A + hf*(cr(B, 0, 1) + cr(B, 0, -1))
            dd = sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0)
            V = solver_TDMAcyclic(-As, Ap, -An, dd, N)
            V = (sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0) + As*cr(V, 0, -1) + An*cr(V, 0, 1))*iApM
            resMV = nc.sum(nc.abs(Ap*V - sV - Aw*cr(V, -1, 0) - Ae*cr(V, 1, 0) - As*cr(V, 0, -1) - An*cr(V, 0, 1)))

            resM = hf*(resMU+resMV)*iNN
            resM_cpu = convert(resM)
            # print("Momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
            itM = itM+1




        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = large
        Aw = rho*A*A*hf*(cr(iApM, -1,  0) + iApM)
        Ae = rho*A*A*hf*(cr(iApM,  1,  0) + iApM)
        As = rho*A*A*hf*(cr(iApM,  0, -1) + iApM)
        An = rho*A*A*hf*(cr(iApM,  0,  1) + iApM)
        Ap = Aw+Ae+As+An
        iApP = one/Ap
        So = -rho*A*(Ue-cr(Ue, -1, 0) + Vn-cr(Vn, 0, -1))
        pc[:,:] = Z[:,:]

        itP  = 0
        resP = large
        while (resP>tollP and itP<maxIt):

            dd = So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0)
            pc = solver_TDMAcyclic(-As, Ap, -An, dd, N)
            pc = (So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0) + As*cr(pc, 0, -1) + An*cr(pc, 0, 1))*iApP

            resP = nc.sum(nc.abs(Ap*pc - So - Aw*cr(pc, -1, 0) - Ae*cr(pc, 1, 0) - As*cr(pc, 0, -1) - An*cr(pc, 0, 1)))
            resP = resP*iNN

            resP_cpu = convert(resP)
            # print("Pressure correction:  it {0:3d}  residuals {1:3e}".format(itP, resP_cpu))
            itP = itP+1




        #---------------------------- update values using under relaxation factors
        deltpX1 = cr(pc, -1, 0) - cr(pc, 1, 0)
        deltpX2 = pc            - cr(pc, 1, 0)

        deltpY1 = cr(pc, 0, -1) - cr(pc, 0, 1)
        deltpY2 = pc            - cr(pc, 0, 1)

        P = P + alphaP*pc
        U  = alphaUV*(U  + A*iApM*hf*deltpX1)                    + (one-alphaUV)*U
        V  = alphaUV*(V  + A*iApM*hf*deltpY1)                    + (one-alphaUV)*V
        Ue = alphaUV*(Ue + A*hf*(cr(iApM, 1, 0) + iApM)*deltpX2) + (one-alphaUV)*Ue
        Vn = alphaUV*(Vn + A*hf*(cr(iApM, 0, 1) + iApM)*deltpY2) + (one-alphaUV)*Vn

        res = nc.sum(nc.abs(So))
        res = res*iNN
        res_cpu = convert(res)
        # print("SIMPLE iterations:  it {0:3d}  residuals {1:3e}".format(it, res_cpu))




        #---------------------------- solve transport equation for passive scalar
        if (PASSIVE):

            # solve iteratively
            Fw = rho*cr(Ue, -1, 0)
            Fe = rho*Ue
            Fs = rho*cr(Vn, 0, -1)
            Fn = rho*Vn

            Aw = A*(Dc + hf*(nc.abs(Fw) + Fw))
            Ae = A*(Dc + hf*(nc.abs(Fe) - Fe))
            As = A*(Dc + hf*(nc.abs(Fs) + Fs))
            An = A*(Dc + hf*(nc.abs(Fn) - Fn))
            Ao = rho*A*dl/delt

            Ap = Ao + (Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs))
            iApC = one/Ap

            itC  = 0
            resC = large
            while (resC>tollC and itC<maxIt):
                dd = Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0)
                C = solver_TDMAcyclic(-As, Ap, -An, dd, N)
                C = (Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0) + As*cr(C, 0, -1) + An*cr(C, 0, 1))*iApC

                resC = nc.sum(nc.abs(Ap*C - Ao*Co - Aw*cr(C, -1, 0) - Ae*cr(C, 1, 0) - As*cr(C, 0, -1) - An*cr(C, 0, 1)))
                resC = resC*iNN
                resC_cpu = convert(resC)
                # print("Passive scalar:  it {0:3d}  residuals {1:3e}".format(itC, resC_cpu))
                itC = itC+1

            # find integral of passive scalar
            totSca = convert(nc.sum(C))
            maxSca = convert(nc.max(C))
            print("Tot scalar {0:.8e}  max scalar {1:3e}".format(totSca, maxSca))

        it = it+1




    #---------------------------- print update and save fields
    if (it==maxIt):
        print("Attention: SIMPLE solver not converged!!!")
        exit()

    else:
        # find new delt based on Courant number
        cdelt = CNum*dl/(sqrt(nc.max(U)*nc.max(U) + nc.max(V)*nc.max(V))+small)
        delt = convert(cdelt)
        delt = min(delt, maxDelt)
        totTime = totTime + delt
        tstep = tstep+1
        its = it

        # check divergence
        div = rho*dl*nc.sum(nc.abs(cr(Ue, -1, 0) - Ue + cr(Vn, 0, -1) - Vn))
        div = div*iNN
        div_cpu = convert(div)  

        # print values
        tend = time()
        if (tstep%print_res == 0):
            wtime = (tend-tstart)
            print("Wall time [s] {0:6.1f}  steps {1:3d}  time {2:5.2e}  delt {3:5.2e}  resM {4:5.2e}  "\
                "resP {5:5.2e}  resC {6:5.2e}  res {7:5.2e}  its {8:3d}  div {9:5.2e}"       \
            .format(wtime, tstep, totTime, delt, resM_cpu, resP_cpu, \
            resC_cpu, res_cpu, its, div_cpu))

        # save images
        if (tstep%print_img == 0):
            print_fields(U, V, P, C, tstep, dir)

        # write checkpoint
        if (tstep%print_ckp == 0):
            save_fields(totTime, U, V, P, C, B)


        # print spectrum
        if (tstep%print_spe == 0):
            plot_spectrum(U, V, L, tstep)



# write checkpoint always at the end
print("End of the run. Saving latest results.")

save_fields(totTime, U, V, P, C, B)
