import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import imageio

from PIL import Image
from boututils.datafile import DataFile
from boutdata.collect import collect
from pyevtk.hl import gridToVTK, writeParallelVTKGrid

# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, '../../../../codes/TurboGenPY/')

from tkespec import compute_tke_spectrum2d_3v
from isoturb import generate_isotropic_turbulence_2d



SAVE_UVW = False
DTYPE    = 'float32'
DIR      = 0  # orientation plot (0=> x==horizontal; 1=> z==horizontal). In BOUT++ z is always periodic!
STIME    = 200 # starting time to save fields
FTIME    = 301 # starting time to take as last image
ITIME    = 2  # skip between STIME, FTIME, ITIME
NDNS     = 100
SAVE_NPZ = True

file = open("HW_data/HW_0/data/BOUT.inp", 'r')
for line in file:
    if "nx =" in line:
        NX = int(line.split()[2]) - 4
    if "ny =" in line:
        NY = int(line.split()[2])
    if "nz =" in line:
        NZ = int(line.split()[2])
    if "Lx =" in line:
        LX = float(line.split()[2])
    if "Ly =" in line:
        LY = float(line.split()[2])
    if "Lz =" in line:
        LZ = float(line.split()[2])

DX  = LX/NX
DY  = LY/NY
DZ  = LZ/NZ
NY2 = int(NY/2)-1

print("System sizes are: Lx,Ly,Lz,dx,dy,dz,nx,ny,nz =", LX,LY,LZ,DX,DY,DZ,NX,NY,NZ)

useLogSca = True
xLogLim    = [1.0e-2, 100]   # to do: to make nmore general
yLogLim    = [1.e-10, 10.]
xLinLim    = [0.0e0, 600]
yLinLim    = [0.0e0, 1.0]

ttime  = []
Energy = []


def cr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))

def convert(x):
    return x

    


def save_fields(totTime, U, V, P, filename="restart.npz"):

    # save restart file
    np.savez(filename, t=totTime, U=U, V=V, P=P)


# create folders fields and paths
if (SAVE_NPZ):
   isExist = os.path.exists("fields_npz")
   if not isExist:
       os.makedirs("fields_npz")
else:
    isExist = os.path.exists("fields_vts")
    if not isExist:
        os.makedirs("fields_vts")

x = np.linspace(0,LX,NX)
y = np.linspace(0,LY,NY)
z = np.linspace(0,LZ,NZ)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


# run on data
nMax = []
nMin = []
phiMax = []
phiMin = []
vortMax = []
vortMin = []
for nrun in range(NDNS):
    newfolder = "HW_data/HW_" + str(nrun) + "/data/"
    file_path = newfolder + "BOUT.log.0"
    with open(file_path, 'r') as file:
        content = file.read()
        if "finished" in content:
        #if (True): 
            
            os.chdir(newfolder)
            print("reading " + newfolder)
            
            for t in range(STIME,FTIME,ITIME):

#                Img_n = collect("n",    yind=0, tind=t, xguards=False, info=False)
#                Img_p = collect("phi",  yind=0, tind=t, xguards=False, info=False)
#                Img_v = collect("vort", yind=0, tind=t, xguards=False, info=False)

                Img_n = collect("n",    tind=t, xguards=False, info=False)
                Img_p = collect("phi",  tind=t, xguards=False, info=False)
                Img_v = collect("vort", tind=t, xguards=False, info=False)


                # save numpy files for training NN
                if (SAVE_NPZ):
                    Img_n = Img_n[0,:,NY2,:]
                    Img_p = Img_p[0,:,NY2,:]
                    Img_v = Img_v[0,:,NY2,:]
                    save_fields(t, Img_n, Img_p, Img_v, "../../../fields_npz/fields_run" + str(nrun) + "_time" + str(t).zfill(3) + ".npz")
                else:
                    Img_n = Img_n[0,:,:,:]
                    Img_p = Img_p[0,:,:,:]
                    Img_v = Img_v[0,:,:,:]

#                    # save csv files for table
#                    csv_filename = "../../../fields_vts/fields_run" + str(nrun) + "_time" + str(t).zfill(3) + ".csv"
#                    f = open(csv_filename, "w")
#                    f.write("x,  y,  z,  n,  phi,  vor\n")
#                    for kk in range(NZ):
#                        for jj in range(NY):
#                            for ii in range(NX):
#                                f.write("%.8f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f\n" % (ii*DX, jj*DY, kk*DZ, Img_n[ii,jj,kk], Img_p[ii,jj,kk], Img_v[ii,jj,kk]))
#                    f.close()


                    # save vts files for paraview visualization
                    vtk_filename = "../../../fields_vts/fields_run" + str(nrun) + "_time" + str(t).zfill(3)
                    gridToVTK(vtk_filename, X, Y, Z, pointData={"n": Img_n, "phi": Img_p, "vort": Img_v})

                print("done for file time step", t)

            os.chdir("../../../")
