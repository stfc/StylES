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
import numpy as np
import os

from PIL import Image
from matplotlib import cm

OUTPUT_DIM  = 128
DTYPE       = "float64"
PATH        = "./../testloop/data/HIT_2D_ReT1000/"
CORRECT_DIV = True



#------------------------- Procedure
#   1- load single fields U, V and Pressure
#   2- convert single fields to black and white colours
#   3- resize fields, if needed
#   4- scale fields. Note: U and V must have been produced using same Min and Max values!!
#   5- correct divergence, if set CORRECT_DIV is True
#   6- check divergence
#   7- find vorticity
#   8- find momentum
#   9- combine U,V and P in the single image "uvw.png"


# clean up
os.system("rm " + PATH + "velx_BW.png")
os.system("rm " + PATH + "vely_BW.png")
os.system("rm " + PATH + "pres_BW.png")
os.system("rm " + PATH + "divergence.png")
os.system("rm " + PATH + "vorticity.png")
os.system("rm " + PATH + "momentum.png")
os.system("rm " + PATH + "uvw.png")


#============================== 1) load single fields
velx = Image.open(PATH + "velx.png")
vely = Image.open(PATH + "vely.png")
pres = Image.open(PATH + "pres.png")


#============================== 2) convert single field to black and white
velx = velx.convert('L')
vely = vely.convert('L')
pres = pres.convert('L')



#============================== 3) resize, if needed
nc, nr = velx.size
if (nc != OUTPUT_DIM):
    velx = velx.resize((OUTPUT_DIM,OUTPUT_DIM))
    vely = vely.resize((OUTPUT_DIM,OUTPUT_DIM))
    pres = pres.resize((OUTPUT_DIM,OUTPUT_DIM))


# save images in black and white
velx.save(PATH + "BW_velx.png")
vely.save(PATH + "BW_vely.png")
pres.save(PATH + "BW_pres.png")



#============================== 4) scale fields

# convert to numpy arrays
velx = convert(velx, dtype=DTYPE)
vely = convert(vely, dtype=DTYPE)
pres = convert(pres, dtype=DTYPE)

# scale fields to [0-1] range. Note: U and V fields are supposed to be obtained from same Min and Max values!!
velx = velx/255.0
vely = vely/255.0
pres = pres/255.0



#============================== 5) correct divergence
if (CORRECT_DIV):

    for j in range(0,nr,2):
        jm=j-1
        jp=j+1
        if (j==0):
            jm=nr-1
        if (j==nr-1):
            jp=0
        for i in range(0,nc-1):
            im=i-1
            ip=i+1
            if (i==0):
                im=nc-1
            velx[ip,j] = velx[im,j] - (vely[i,jp] - vely[i,jm])

    for j in range(1,nr,2):
        jm=j-1
        jp=j+1
        if (j==0):
            jm=nr-1
        if (j==nr-1):
            jp=0
        for i in range(0,nc-1):
            im=i-1
            ip=i+1
            if (i==0):
                im=nc-1
            velx[ip,j] = velx[im,j] - (vely[i,jp] - vely[i,jm])



#============================== 6) check divergence
cont=0
totDiv_o=0
totDiv_e=0
div = np.zeros([nc,nr])
for j in range(1,nr-1):
    for i in range(1,nc-1):
        div[i,j] = (velx[i+1,j] - velx[i-1,j]) + (vely[i,j+1] - vely[i,j-1])
        if ((cont/2) % 2 == 0):
            totDiv_o = totDiv_o + div[i,j]
        else:
            totDiv_e = totDiv_e + div[i,j]
        cont=cont+1

minDiv = np.min(div)
maxDiv = np.max(div)

div = Image.fromarray(np.uint8(cm.gist_earth(div)*255))
div.save(PATH + "divergence.png")

print("")
print("Odd, even and total divergencies are", totDiv_o, totDiv_e, totDiv_o+totDiv_e)
print("Minx and max divergencies are:      ", minDiv, maxDiv)



#============================== 7) find vorticity
vor = np.zeros([nc,nr])
vor = find_vorticity(U, V)
vor = (vor - np.min(vor))/(np.max(vor) - np.min(vor))
vor = Image.fromarray(np.uint8(cm.gist_earth(vor)*255))
vor.save(PATH + "vorticity.png")



#============================== 8) find momentum
totMomx = 0.0e0
totMomy = 0.0e0
momx = np.zeros([nc,nr])
momy = np.zeros([nc,nr])
for jj in range(0,nr):
    j = jj
    jm = j-1
    jp = j+1
    if (jj==0):
        jm = nr-1
    if (jj==nr-1):
        jp = 0
    for ii in range(0,nc):
        i  = ii
        im = i-1
        ip = i+1
        if (ii==0):
            im = nc-1
        if (ii==nc-1):
            ip = 0

        momx[i,j] = -velx[i,j]*0.5e0*(velx[ip,j] - velx[im,j])   \
                     -vely[i,j]*0.5e0*(velx[i,jp] - velx[i,jm])   \
                     -0.5e0*(pres[ip,j]-pres[im,j])                \
                     +(velx[ip,j] - 2.0e0*velx[i,j] + velx[im,j]) \
                     +(velx[i,jp] - 2.0e0*velx[i,j] + velx[i,jm])

        momy[i,j] = -velx[i,j]*0.5e0*(vely[ip,j] - vely[im,j])   \
                     -vely[i,j]*0.5e0*(vely[i,jp] - vely[i,jm])   \
                     -0.5e0*(pres[i,jp]-pres[i,jm])                \
                     +(vely[ip,j] - 2.0e0*vely[i,j] + vely[im,j]) \
                     +(vely[i,jp] - 2.0e0*vely[i,j] + vely[i,jm])

        totMomx = totMomx + momx[i,j]
        totMomy = totMomy + momy[i,j]

print("")
print("Total momentum in x-direction is ", totMomx)
print("Total momentum in y-direction is ", totMomy)

mom = np.zeros([nc,nr,3])
momx = (momx - np.min(momx))/(np.max(momx) - np.min(momx) + 1e-10)
momy = (momy - np.min(momy))/(np.max(momy) - np.min(momy) + 1e-10)

mom[:,:,0] = momx
mom[:,:,1] = momy

mom = Image.fromarray(np.uint8(mom*255.0e0))
mom.save(PATH + "momentum.png")



#============================== 9) find combined image uvw
uvw = np.zeros([nc,nr,3])
uvw[:,:,0] = velx
uvw[:,:,1] = vely
uvw[:,:,2] = pres

uvw = Image.fromarray(np.uint8(uvw*255))
uvw.save(PATH + "uvw.png")

