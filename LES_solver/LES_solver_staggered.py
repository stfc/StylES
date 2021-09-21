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


        #---------------------------- solve momentum equations
        # x-direction
        Fw = A*rho*hf*(Uo            + cr(Uo, -1, 0))
        Fe = A*rho*hf*(cr(Uo,  1, 0) + Uo           )
        Fs = A*rho*hf*(Vo            + cr(Vo, -1, 0))
        Fn = A*rho*hf*(cr(Vo,  0, 1) + cr(Vo, -1, 1))

        Aw = Dc + hf*(nc.abs(Fw) + Fw)
        Ae = Dc + hf*(nc.abs(Fe) - Fe)
        As = Dc + hf*(nc.abs(Fs) + Fs)
        An = Dc + hf*(nc.abs(Fn) - Fn)
        Ao = rho*A*dl/delt

        Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
        iApU = one/Ap
        sU   = Ao*Uo -(P - cr(P, -1, 0))*A + hf*(B + cr(B, -1, 0))

        itM  = 0
        resM = large
        while (resM>tollM and itM<maxIt):

            dd = sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0)
            U = solver_TDMAcyclic(-As, Ap, -An, dd, N)
            U = (sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0) + As*cr(U, 0, -1) + An*cr(U, 0, 1))*iApU
            resM = nc.sum(nc.abs(Ap*U - sU - Aw*cr(U, -1, 0) - Ae*cr(U, 1, 0) - As*cr(U, 0, -1) - An*cr(U, 0, 1)))
            resM = resM*iNN
            resM_cpu = convert(resM)
            # print("x-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
            itM = itM+1


        # y-direction
        Fw = A*rho*hf*(Uo             + cr(Uo, 0, -1))
        Fe = A*rho*hf*(cr(Uo,  1,  0) + cr(Uo, 1, -1))
        Fs = A*rho*hf*(cr(Vo,  0, -1) + Vo           )
        Fn = A*rho*hf*(Vo             + cr(Vo, 0,  1))

        Aw = Dc + hf*(nc.abs(Fw) + Fw)
        Ae = Dc + hf*(nc.abs(Fe) - Fe)
        As = Dc + hf*(nc.abs(Fs) + Fs)
        An = Dc + hf*(nc.abs(Fn) - Fn)
        Ao = rho*A*dl/delt

        Ap  = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
        iApV = one/Ap
        sV   = Ao*Vo -(P - cr(P, 0, -1))*A + hf*(B + cr(B, 0, -1))

        itM  = 0
        resM = one
        while (resM>tollM and itM<maxIt):

            dd = sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0)
            V = solver_TDMAcyclic(-As, Ap, -An, dd, N)
            V = (sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0) + As*cr(V, 0, -1) + An*cr(V, 0, 1))*iApV
            resM = nc.sum(nc.abs(Ap*V - sV - Aw*cr(V, -1, 0) - Ae*cr(V, 1, 0) - As*cr(V, 0, -1) - An*cr(V, 0, 1)))

            resM = resM*iNN
            resM_cpu = convert(resM)
            # print("y-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(it, resM_cpu))
            itM = itM+1


        #---------------------------- solve pressure correction equation
        itPc  = 0
        resPc = large
        Aw = rho*A*A*cr(iApU, 0,  0)
        Ae = rho*A*A*cr(iApU, 1,  0)
        As = rho*A*A*cr(iApV, 0,  0)
        An = rho*A*A*cr(iApV, 0,  1)
        Ap = Aw+Ae+As+An
        iApP = one/Ap
        So = -rho*A*(cr(U, 1, 0) - U + cr(V, 0, 1) - V)

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
        deltpX1 = cr(pc, -1, 0) - pc
        deltpY1 = cr(pc, 0, -1) - pc

        P  = P + alphaP*pc
        U  = U + A*iApU*deltpX1
        V  = V + A*iApV*deltpY1

        res = nc.sum(nc.abs(So))
        res = res*iNN
        res_cpu = convert(res)
        # print("SIMPLE iterations:  it {0:3d}  residuals {1:3e}".format(it, res_cpu))




        #---------------------------- solve transport equation for passive scalar
        if (PASSIVE):

            # solve iteratively
            Fw = A*rho*cr(U, -1, 0)
            Fe = A*rho*U
            Fs = A*rho*cr(V, 0, -1)
            Fn = A*rho*V

            Aw = Dc + hf*(nc.abs(Fw) + Fw)
            Ae = Dc + hf*(nc.abs(Fe) - Fe)
            As = Dc + hf*(nc.abs(Fs) + Fs)
            An = Dc + hf*(nc.abs(Fn) - Fn)
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
