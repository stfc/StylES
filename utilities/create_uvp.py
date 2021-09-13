import numpy as np
import os

from PIL import Image
from matplotlib import cm

OUTPUT_DIM  = 128
DTYPE       = "float64"
PATH        = "./../testloop/data/2D_HIT_ReT1000/"
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
#   9- combine U,V and P in the single image "uvp.png"


# clean up
os.system("rm " + PATH + "velx_BW.png")
os.system("rm " + PATH + "vely_BW.png")
os.system("rm " + PATH + "pres_BW.png")
os.system("rm " + PATH + "divergence.png")
os.system("rm " + PATH + "vorticity.png")
os.system("rm " + PATH + "momentum.png")
os.system("rm " + PATH + "uvp.png")


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
div = nc.zeros([nc,nr])
for j in range(1,nr-1):
    for i in range(1,nc-1):
        div[i,j] = (velx[i+1,j] - velx[i-1,j]) + (vely[i,j+1] - vely[i,j-1])
        if ((cont/2) % 2 == 0):
            totDiv_o = totDiv_o + div[i,j]
        else:
            totDiv_e = totDiv_e + div[i,j]
        cont=cont+1

minDiv = nc.min(div)
maxDiv = nc.max(div)

div = Image.fromarray(nc.uint8(cm.gist_earth(div)*255))
div.save(PATH + "divergence.png")

print("")
print("Odd, even and total divergencies are", totDiv_o, totDiv_e, totDiv_o+totDiv_e)
print("Minx and max divergencies are:      ", minDiv, maxDiv)



#============================== 7) find vorticity
vor = nc.zeros([nc,nr])
for j in range(1,nr-1):
    for i in range(1,nc-1):
        vor[i,j] = (velx[i,j+1] - velx[i,j-1]) - (vely[i+1,j] - vely[i-1,j])

vor = (vor - nc.min(vor))/(nc.max(vor) - nc.min(vor))
vor = Image.fromarray(nc.uint8(cm.gist_earth(vor)*255))
vor.save(PATH + "vorticity.png")



#============================== 8) find momentum
totMomx = 0.0e0
totMomy = 0.0e0
momx = nc.zeros([nc,nr])
momy = nc.zeros([nc,nr])
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

mom = nc.zeros([nc,nr,3])
momx = (momx - nc.min(momx))/(nc.max(momx) - nc.min(momx) + 1e-10)
momy = (momy - nc.min(momy))/(nc.max(momy) - nc.min(momy) + 1e-10)

mom[:,:,0] = momx
mom[:,:,1] = momy

mom = Image.fromarray(nc.uint8(mom*255.0e0))
mom.save(PATH + "momentum.png")



#============================== 9) find combined image uvp
uvp = nc.zeros([nc,nr,3])
uvp[:,:,0] = velx
uvp[:,:,1] = vely
uvp[:,:,2] = pres

uvp = Image.fromarray(nc.uint8(uvp*255))
uvp.save(PATH + "uvp.png")

