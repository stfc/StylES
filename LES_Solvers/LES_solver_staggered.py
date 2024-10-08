#----------------------------------------------------------------------------------------------
#
#    Copyright (C): 2022 UKRI-STFC (Hartree Centre)
#
#    Author: Jony Castagna, Francesca Schiavello
#
#    Licence: This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------------------------------
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

runcase = sys.argv[1]
if (runcase=="" or runcase=="HIT_2D"):
    print("Inputs from file HIT_2D")
    from testcases.HIT_2D.HIT_2D import *
elif (runcase=="HIT_2D_reconstruction"):
    print("Inputs from file HIT_2D_reconstruction")
    from testcases.HIT_2D.HIT_2D_reconstruction import *
    


# start timing
tstart = time()



#---------------------------- define arrays
Uo = nc.zeros([N,N], dtype=DTYPE)   # old x-velocity
Vo = nc.zeros([N,N], dtype=DTYPE)   # old y-velocity
Po = nc.zeros([N,N], dtype=DTYPE)   # old pressure field
Co = nc.zeros([N,N], dtype=DTYPE)   # old passive scalar
pc = nc.zeros([N,N], dtype=DTYPE)  # pressure correction
Z  = nc.zeros([N,N], dtype=DTYPE)
DNS_cv = np.zeros([totSteps+1, 4])




#---------------------------- set flow pressure, velocity fields and BCs

# clean up and declarations
#os.system("rm restart.npz")
os.system("rm DNS_center_values.txt")

os.system("rm -rf plots")
os.system("rm -rf fields")
os.system("rm -rf uvp")
os.system("rm -rf energy")
os.system("rm -rf v_viol")

os.system("mkdir plots")
os.system("mkdir fields")
os.system("mkdir uvp")
os.system("mkdir energy")
os.system("mkdir v_viol")

# initial flow
for run in range(NRUNS):
    totTime = zero
    if (RESTART):
        U, V, P, C, B, totTime = load_fields(DNSrun=True)
    else:
        U, V, P, C, B, totTime = init_fields(run)


    #---------------------------- main time step loop
    tstep    = 0
    resM_cpu = zero
    resP_cpu = zero
    resC_cpu = zero
    res_cpu  = zero
    its      = 0

    # check divergence
    div = rho*A*nc.sum(nc.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
    div = div*iNN
    div_cpu = convert(div)

    # find new delt based on Courant number
    cdelt = CNum*dl/(sqrt(nc.max(U)*nc.max(U) + nc.max(V)*nc.max(V))+small)
    delt = convert(cdelt)
    delt = min(delt, maxDelt)

    # start loop
    while (tstep<totSteps and totTime<finalTime):


        # save old values of U, V and P
        Uo[:,:] = U[:,:]
        Vo[:,:] = V[:,:]
        Po[:,:] = P[:,:]
        if (PASSIVE):
            Co[:,:] = C[:,:]


        # start outer loop on SIMPLE convergence
        it = 0
        res = large
        while (res>toll and it<maxItDNS):


            #---------------------------- solve momentum equations
            # x-direction
            Fw = A*rho*hf*(Uo            + cr(Uo, -1, 0))
            Fe = A*rho*hf*(cr(Uo,  1, 0) + Uo           )
            Fs = A*rho*hf*(Vo            + cr(Vo, -1, 0))
            Fn = A*rho*hf*(cr(Vo,  0, 1) + cr(Vo, -1, 1))

            Aw = Dc + hf*Fw  # hf*(nc.abs(Fw) + Fw)
            Ae = Dc - hf*Fe  # hf*(nc.abs(Fe) - Fe)
            As = Dc + hf*Fs  # hf*(nc.abs(Fs) + Fs)
            An = Dc - hf*Fn  # hf*(nc.abs(Fn) - Fn)
            Ao = rho*A*dl/delt

            Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
            iApU = one/Ap
            sU = Ao*Uo -(P - cr(P, -1, 0))*A + hf*(B + cr(B, -1, 0))

            itM  = 0
            resM = large
            while (resM>tollM and itM<maxItDNS):

                dd = sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0)
                U = solver_TDMAcyclic(-As, Ap, -An, dd, N)
                U = (sU + Aw*cr(U, -1, 0) + Ae*cr(U, 1, 0) + As*cr(U, 0, -1) + An*cr(U, 0, 1))*iApU
                resM = nc.sum(nc.abs(Ap*U - sU - Aw*cr(U, -1, 0) - Ae*cr(U, 1, 0) - As*cr(U, 0, -1) - An*cr(U, 0, 1)))
                resM = resM*iNN
                resM_cpu = convert(resM)
                if ((itM+1)%100 == 0):                
                    print("x-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
                itM = itM+1


            # y-direction
            Fw = A*rho*hf*(Uo             + cr(Uo, 0, -1))
            Fe = A*rho*hf*(cr(Uo,  1,  0) + cr(Uo, 1, -1))
            Fs = A*rho*hf*(cr(Vo,  0, -1) + Vo           )
            Fn = A*rho*hf*(Vo             + cr(Vo, 0,  1))

            Aw = Dc + hf*Fw  # hf*(nc.abs(Fw) + Fw)
            Ae = Dc - hf*Fe  # hf*(nc.abs(Fe) - Fe)
            As = Dc + hf*Fs  # hf*(nc.abs(Fs) + Fs)
            An = Dc - hf*Fn  # hf*(nc.abs(Fn) - Fn)
            Ao = rho*A*dl/delt

            Ap  = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
            iApV = one/Ap
            sV = Ao*Vo -(P - cr(P, 0, -1))*A + hf*(B + cr(B, 0, -1))

            itM  = 0
            resM = one
            while (resM>tollM and itM<maxItDNS):

                dd = sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0)
                V = solver_TDMAcyclic(-As, Ap, -An, dd, N)
                V = (sV + Aw*cr(V, -1, 0) + Ae*cr(V, 1, 0) + As*cr(V, 0, -1) + An*cr(V, 0, 1))*iApV
                resM = nc.sum(nc.abs(Ap*V - sV - Aw*cr(V, -1, 0) - Ae*cr(V, 1, 0) - As*cr(V, 0, -1) - An*cr(V, 0, 1)))

                resM = resM*iNN
                resM_cpu = convert(resM)
                if ((itM+1)%100 == 0):             
                    print("y-momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
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
            while (resP>tollP and itP<maxItDNS):

                dd = So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0)
                pc = solver_TDMAcyclic(-As, Ap, -An, dd, N)
                pc = (So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0) + As*cr(pc, 0, -1) + An*cr(pc, 0, 1))*iApP

                resP = nc.sum(nc.abs(Ap*pc - So - Aw*cr(pc, -1, 0) - Ae*cr(pc, 1, 0) - As*cr(pc, 0, -1) - An*cr(pc, 0, 1)))
                resP = resP*iNN

                resP_cpu = convert(resP)
                if ((itP+1)%100 == 0):
                    print("Pressure correction:  it {0:3d}  residuals {1:3e}".format(itP, resP_cpu))
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
            if ((it+1)%100 == 0):
                print("SIMPLE iterations:  it {0:3d}  residuals {1:3e}".format(it, res_cpu))




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
                while (resC>tollC and itC<maxItDNS):
                    dd = Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0)
                    C = solver_TDMAcyclic(-As, Ap, -An, dd, N)
                    C = (Ao*Co + Aw*cr(C, -1, 0) + Ae*cr(C, 1, 0) + As*cr(C, 0, -1) + An*cr(C, 0, 1))*iApC

                    resC = nc.sum(nc.abs(Ap*C - Ao*Co - Aw*cr(C, -1, 0) - Ae*cr(C, 1, 0) - As*cr(C, 0, -1) - An*cr(C, 0, 1)))
                    resC = resC*iNN
                    resC_cpu = convert(resC)
                    if ((itC+1)%100==0):                    
                        print("Passive scalar:  it {0:3d}  residuals {1:3e}".format(itC, resC_cpu))
                    itC = itC+1

                # find integral of passive scalar
                totSca = convert(nc.sum(C))
                maxSca = convert(nc.max(C))
                print("Tot scalar {0:.8e}  max scalar {1:3e}".format(totSca, maxSca))

            it = it+1




        #---------------------------- print update and save fields
        if (it==maxItDNS):
            print("Attention: SIMPLE solver not converged!!!")
            exit()

        else:

            # track center point velocities and pressure
            DNS_cv[tstep,0] = totTime
            DNS_cv[tstep,1] = U[N//2, N//2]
            DNS_cv[tstep,2] = V[N//2, N//2]
            DNS_cv[tstep,3] = P[N//2, N//2]


            # check min and max values
            u_max = nc.max(nc.abs(U))
            v_max = nc.max(nc.abs(V))
            u_max_cpu = convert(u_max)
            v_max_cpu = convert(v_max)
            uv_max = [u_max_cpu, v_max_cpu]
            if uv_max[0] > uRef or uv_max[1] > uRef:
                save_vel_violations("v_viol/v_viol_run"+ str(run) + "txt", uv_max, tstep)


            # plot, save, find spectrum fields
            if (len(te)>0):

                #loop for turnover times(te) and respective time in seconds(te_s)
                for s in range(len(te_s)):
                    if (totTime<te_s[s]+hf*delt and totTime>te_s[s]-hf*delt):
                        W = find_vorticity(U, V)
                        print_fields(U, V, P, W, N,            "plots/plots_run"   + str(run) + "_" + str(te[s]) + "te.png")
                        save_fields(totTime, U, V, P, C, B, W, "fields/fields_run" + str(run) + "_" + str(te[s]) + "te.npz")
                        plot_spectrum_2d_3v(U, V, L,                 "energy/energy_run" + str(run) + "_" + str(te[s]) + "te.png")
            else:

                if (totTime>=statsTime):
                    tail = "run{0:d}_it{1:d}".format(run,tstep)

                    # save images
                    if (tstep%print_img == 0):
                        W = find_vorticity(U, V)
                        print_fields(U, V, P, W, N, "plots/plots_" + tail + ".png")
                        # print_fields_1(W, "plots/vorticity_" + tail + ".png", Wmin=-0.3, Wmax=0.3)

                    # write checkpoint
                    if (tstep%print_ckp == 0):
                        W = find_vorticity(U, V)
                        save_fields(totTime, U, V, P, C, B, W, "fields/fields_" + tail + ".npz")

                    # print spectrum
                    if (tstep%print_spe == 0):
                        plot_spectrum_2d_3v(U, V, L, "energy/energy_spectrum_" + tail + ".png")


            # find new delt based on Courant number
            cdelt = CNum*dl/(sqrt(nc.max(U)*nc.max(U) + nc.max(V)*nc.max(V))+small)
            delt = convert(cdelt)
            delt = min(delt, maxDelt)
            totTime = totTime + delt
            tstep = tstep+1
            its = it

            # check divergence
            div = rho*A*nc.sum(nc.abs(cr(U, 1, 0) - U + cr(V, 0, 1) - V))
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

# end of the simulation




# plot, save, find spectrum fields
if (len(te)==0):
    tail = "run{0:d}_it{1:d}".format(run,tstep)

    # save images
    W = find_vorticity(U, V)
    print_fields(U, V, P, W, N, "plots/plots_" + tail + ".png")
    # print_fields_1(W, "plots/vorticity_" + tail + ".png", Wmin=-0.3, Wmax=0.3)

    # write checkpoint
    save_fields(totTime, U, V, P, C, B, W, "fields/fields_" + tail + ".npz")

    # print spectrum
    plot_spectrum_2d_3v(U, V, L, "energy/energy_spectrum_" + tail + ".png")

# save center values
filename = "DNS_center_values" + ".txt"
np.savetxt(filename, np.c_[DNS_cv[0:tstep+1,0], DNS_cv[0:tstep+1,1], DNS_cv[0:tstep+1,2], DNS_cv[0:tstep+1,3]], fmt='%1.4e')   # use exponential notation

# save restart file
save_fields(totTime, U, V, P, C, B, W)


print("Simulation successfully completed!")
