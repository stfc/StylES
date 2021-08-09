import tensorflow as tf
import numpy as np
import os

from PIL import Image
from matplotlib import cm

def periodic_padding_flexible(tensor, axis, padding=1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """

    if isinstance(axis,int):
        axis = (axis,)

    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)

    if (padding[0][0]>0):
        for ax,p in zip(axis,padding):
            # create a slice object that selects everything from all axes,
            # except only 0:p for the specified for right, and -p: for left

            ind_right = [slice(-p[0],None) if i == ax else slice(None) for i in range(ndim)]
            ind_left  = [slice(0, p[1])    if i == ax else slice(None) for i in range(ndim)]

            right     = tensor[ind_right]
            left      = tensor[ind_left]
            middle    = tensor
            tensor    = tf.concat([right,middle,left], axis=ax)

    return tensor


# delete old images
os.system("rm -rf *.png")


# save fields in black and white
velx = Image.open("../../../data/last_frame/velocity_x.png")
velx = velx.convert('L') # convert image to black and white
#velx = velx.resize((4,4))
#velx.save('velx.png')

vely = Image.open("../../../data/last_frame/velocity_y.png")
vely = vely.convert('L') # convert image to black and white
#vely = vely.resize((4,4))
#vely.save('vely.png')

pres = Image.open("../../../data/last_frame/pressure.png")
pres = pres.convert('L') # convert image to black and white
#pres.save('pres.png')


# convert to numpy arrays
velx = np.asarray(velx, dtype=np.float64)
vely = np.asarray(vely, dtype=np.float64)
pres = np.asarray(pres, dtype=np.float64)


# adjust fields to satisfy continuity
velx = velx/255.0
vely = vely/255.0
pres = pres/255.0

nc = velx.shape[0]
nr = velx.shape[1]

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
        velx[ip][j] = velx[im][j] - (vely[i][jp] - vely[i][jm])

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
        velx[ip][j] = velx[im][j] - (vely[i][jp] - vely[i][jm])


# check divergency
cont=0
totDiv_o=0
totDiv_e=0
div = np.zeros([nc,nr])
for j in range(1,nr-1):
    for i in range(1,nc-1):
        div[i][j] = (velx[i+1][j] - velx[i-1][j]) + (vely[i][j+1] - vely[i][j-1])
        if ((cont/2) % 2 == 0):
            totDiv_o = totDiv_o + div[i][j]
        else:
            totDiv_e = totDiv_e + div[i][j]
        cont=cont+1

print("Odd, even and total divergencies are", totDiv_o, totDiv_e, totDiv_o+totDiv_e)

minDiv = np.min(div)
maxDiv = np.max(div)
print("Minx and max divergencies are: ", minDiv, maxDiv)

div = Image.fromarray(np.uint8(cm.gist_earth(div)*255))
div.save("div.png")



# normalize fields
maxVx = np.max(velx)
maxVy = np.max(vely)
maxV = max(maxVx, maxVy)

minVx = np.min(velx)
minVy = np.min(vely)
minV = min(minVx, minVy)

velx = (velx - minV)/(maxV - minV)
vely = (vely - minV)/(maxV - minV)
pres = (pres - np.min(pres))/(np.max(pres) - np.min(pres))


# add periodicity
#velx = periodic_padding_flexible(velx, axis=(0,1), padding=([1, 1], [1, 1]))
#vely = periodic_padding_flexible(vely, axis=(0,1), padding=([1, 1], [1, 1]))
#pres = periodic_padding_flexible(pres, axis=(0,1), padding=([1, 1], [1, 1]))


# find combined image uvp
uvp = np.zeros([nc,nr,3])
uvp[:,:,0] = velx
uvp[:,:,1] = vely
uvp[:,:,2] = pres

uvp = Image.fromarray(np.uint8(uvp*255))
uvp.save("uvp.png")



# find vorticity
vor = np.zeros([nc,nr])
for j in range(1,nr-1):
    for i in range(1,nc-1):
        vor[i][j] = (velx[i][j+1] - velx[i][j-1]) - (vely[i+1][j] - vely[i-1][j])

vor = (vor - np.min(vor))/(np.max(vor) - np.min(vor))
vor = Image.fromarray(np.uint8(cm.gist_earth(vor)*255))
vor.save("vor.png")

