import tensorflow as tf
import numpy as np

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

    if (padding[0,0]>0):
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



x_in = np.ones([1, 1, 4, 4])
x_in[:,:, :, 0] = 0
x_in[:,:, 0, :] = 0
x_in[:,:, :,-1] = 0
x_in[:,:,-1, :] = 0
x_in[:,:, :, 1] = 0
x_in[:,:, 1, :] = 0

x = tf.constant(x_in, dtype=DTYPE)

w = tf.constant(1.0, shape=[4, 4, 1, 1], dtype=DTYPE)
y1 = tf.nn.conv2d(x, filters=w, strides=[1, 1, 2, 2], padding="SAME", data_format="NCHW")



if (w.shape[0] % 2 == 0):
    pleft   = np.int((w.shape[0]-1)/2)
    pright  = np.int(w.shape[0]/2)
    ptop    = pleft
    pbottom = pright
else:
    pleft   = np.int(w.shape[0]/2)
    pright  = pleft
    ptop    = pleft
    pbottom = pleft
    

xpadded = periodic_padding_flexible(x, axis=(2,3), padding=([pleft, pright], [ptop, pbottom]))
y2 = tf.nn.conv2d(xpadded, filters=w, strides=[1, 1, 2, 2], padding="VALID", data_format="NCHW")


xpadded = periodic_padding_flexible(x, axis=(2,3), padding=([pleft, pright], [ptop, pbottom]))
y3 = tf.nn.depthwise_conv2d(xpadded, filter=w, strides=[1, 1, 2, 2], padding="VALID", data_format="NCHW")



print(x_in)
print(y1)
print(y2)
print(y3)