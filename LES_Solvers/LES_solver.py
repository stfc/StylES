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
Ue = nc.zeros([N,N], dtype=DTYPE)  # face x-velocities
Vn = nc.zeros([N,N], dtype=DTYPE)  # face y-velocities  
pc = nc.zeros([N,N], dtype=DTYPE)  # pressure correction
Z  = nc.zeros([N,N], dtype=DTYPE)
DNS_cv = np.zeros([totSteps+1, 4])




#---------------------------- set flow pressure, velocity fields and BCs

# clean up and declarations
#os.system("rm restart.npz")
os.system("rm DNS_center_values.txt")
os.system("rm Energy_spectrum.png")

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
    div = rho*A*nc.sum(nc.abs(Ue - cr(Ue, -1, 0) + Vn - cr(Vn, 0, -1)))
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
            Fw = A*rho*cr(Ue, -1, 0)
            Fe = A*rho*Ue
            Fs = A*rho*cr(Vn, 0, -1)
            Fn = A*rho*Vn

            Aw = Dc + hf*Fw  # hf*(nc.abs(Fw) + Fw)
            Ae = Dc - hf*Fe  # hf*(nc.abs(Fe) - Fe)
            As = Dc + hf*Fs  # hf*(nc.abs(Fs) + Fs)
            An = Dc - hf*Fn  # hf*(nc.abs(Fn) - Fn)
            Ao = rho*A*dl/delt

            Ap = Ao + Aw + Ae + As + An + (Fe-Fw) + (Fn-Fs)
            iApM = one/Ap

            itM  = 0
            resM = large
            while (resM>tollM and itM<maxItDNS):

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
                if ((itM+1)%100==0):          
                    print("Momemtum iterations:  it {0:3d}  residuals {1:3e}".format(itM, resM_cpu))
                itM = itM+1



            #---------------------------- find Rhie-Chow interpolation (PWIM)
            deltpX1 = hf*(cr(P, 1, 0) - cr(P, -1, 0))
            deltpX2 = hf*(cr(P, 2, 0) - P)    
            deltpX3 = (P - cr(P,  1, 0))

            deltpY1 = hf*(cr(P, 0, 1) - cr(P, 0, -1))
            deltpY2 = hf*(cr(P, 0, 2) - P)
            deltpY3 = (P - cr(P, 0,  1))

            Ue = hf*(cr(U, 1, 0) + U)                 \
                + hf*deltpX1*iApM*A                   \
                + hf*deltpX2*cr(iApM, 1, 0)*A         \
                + hf*deltpX3*(cr(iApM, 1, 0) + iApM)*A

            Vn = hf*(cr(V, 0, 1) + V)                 \
                + hf*deltpY1*iApM*A                   \
                + hf*deltpY2*cr(iApM, 0, 1)*A         \
                + hf*deltpY3*(cr(iApM, 0, 1) + iApM)*A
        


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
            while (resP>tollP and itP<maxItDNS):

                dd = So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0)
                pc = solver_TDMAcyclic(-As, Ap, -An, dd, N)
                pc = (So + Aw*cr(pc, -1, 0) + Ae*cr(pc, 1, 0) + As*cr(pc, 0, -1) + An*cr(pc, 0, 1))*iApP

                resP = nc.sum(nc.abs(Ap*pc - So - Aw*cr(pc, -1, 0) - Ae*cr(pc, 1, 0) - As*cr(pc, 0, -1) - An*cr(pc, 0, 1)))
                resP = resP*iNN

                resP_cpu = convert(resP)
                if ((itP+1)%100==0):
                    print("Pressure correction:  it {0:3d}  residuals {1:3e}".format(itP, resP_cpu))
                itP = itP+1




            #---------------------------- update values using under relaxation factors
            deltpX1 = cr(pc, -1, 0) - cr(pc, 1, 0)
            deltpX2 = pc            - cr(pc, 1, 0)

            deltpY1 = cr(pc, 0, -1) - cr(pc, 0, 1)
            deltpY2 = pc            - cr(pc, 0, 1)

            P  = P  + alphaP*pc
            U  = U  + A*iApM*hf*deltpX1
            V  = V  + A*iApM*hf*deltpY1
            Ue = Ue + A*hf*(cr(iApM, 1, 0) + iApM)*deltpX2
            Vn = Vn + A*hf*(cr(iApM, 0, 1) + iApM)*deltpY2

            res = nc.sum(nc.abs(So))
            res = res*iNN
            res_cpu = convert(res)
            if ((it+1)%100==0):
                print("SIMPLE iterations:  it {0:3d}  residuals {1:3e}".format(it, res_cpu))




            #---------------------------- solve transport equation for passive scalar
            if (PASSIVE):

                # solve iteratively
                Fw = A*rho*cr(Ue, -1, 0)
                Fe = A*rho*Ue
                Fs = A*rho*cr(Vn, 0, -1)
                Fn = A*rho*Vn

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
                    # print("Passive scalar:  it {0:3d}  residuals {1:3e}".format(itC, resC_cpu))
                    itC = itC+1

                # find integral of passive scalar
                totSca = convert(nc.sum(C))
                maxSca = convert(nc.max(C))
                if ((itC+1)%100==0):
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
        
                tail = "run{0:d}_it{1:d}".format(run,tstep)

                # save images
                if (tstep%print_img == 0):
                    W = find_vorticity(U, V)
                    print_fields(U, V, P, W, N, "plots/plots_" + tail + ".png")

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
            div = rho*A*nc.sum(nc.abs(Ue - cr(Ue, -1, 0) + Vn - cr(Vn, 0, -1)))
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
