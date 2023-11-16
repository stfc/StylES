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
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

from tensorflow.keras import layers
from typing import Union

from parameters import *



#A type that represents a valid Tensorflow expression
TfExpression = Union[tf.Tensor, tf.Variable, tf.Operation]

#A type that can be converted to a valid Tensorflow expression
if (DTYPE=="float64"):
    TfExpressionEx = Union[TfExpression, int, np.float64, np.ndarray]
else:
    TfExpressionEx = Union[TfExpression, int, np.float32, np.ndarray]


#------------------------------ general functions
def tr(phi, i, j):
    return tf.roll(phi, (-i, -j), axis=(0,1))

def nr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))



#------------------------------ functions to build StyleGAN network

#------------- define Gaussian kernel
def gaussian_kernel(size: int, mean: float, std: float):

    """Makes 2D gaussian Kernel for convolution."""

    x = tf.range(start = -size, limit = size + 1, dtype = DTYPE)
    Z = (2 * np.pi * std**2)**0.5
    d = tf.math.exp(-0.5 * (x - mean)**2 / std**2) / Z
    gauss_kernel = tf.einsum('i,j->ij', d, d)

    filtered = gauss_kernel / tf.reduce_sum(gauss_kernel)

    return filtered

#-------------define periodic padding
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

    sp = np.sum(np.abs(padding))
    if (sp>0):
        nax = 0
        for ax,p in zip(axis,padding):
            # create a slice object that selects everything from all axes,
            # except only 0:p for the specified for right, and -p: for left

            ind_right = [slice(-p[0],None) if i == ax else slice(None) for i in range(ndim)]
            ind_left  = [slice(0, p[1])    if i == ax else slice(None) for i in range(ndim)]

            if (TESTCASE=='mHW'):
                if (nax==0):
                    right = tensor[ind_right]*0   # non periodic in z-direction ()
                    left  = tensor[ind_left]*0    # Remember padding happens along x first...
                else:
                    right = tensor[ind_right]
                    left  = tensor[ind_left]
                nax = nax+1
            else:
                right = tensor[ind_right]
                left  = tensor[ind_left]

            middle    = tensor
            tensor    = tf.concat([right,middle,left], axis=ax)

    return tensor

#-------------linear interpolation
class layer_lerp(layers.Layer):
    def __init__(self, **kwargs):
        super(layer_lerp, self).__init__()

    def call(self, a, b, t):
        return a + (b - a) * t
    

#-------------Pixel normalization
def pixel_norm(x, epsilon=1e-8):
    epsilon = tf.constant(epsilon, dtype=DTYPE)
    return x * tf.math.rsqrt(tf.math.reduce_mean(tf.math.square(x), axis=1, keepdims=True) + epsilon)


#-------------Find number of feature maps
def nf(stage):
    return min(int(FMAP_BASE / (2.0 ** (stage * FMAP_DECAY))), FMAP_MAX)


#-------------get_Weight 
class layer_get_Weight(layers.Layer):
    def __init__(self, shape, gain=np.sqrt(2), use_wscale=False, lrmul=1, **kwargs):
        super(layer_get_Weight, self).__init__()

        # Equalized learning rate and custom learning rate multiplier.
        fan_in = np.prod(shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
        he_std = gain / np.sqrt(fan_in)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.runtime_coef = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.runtime_coef = lrmul

        w_init = tf.random_normal_initializer(mean=0.0, stddev=init_std)
        self.w = tf.Variable(
            initial_value=w_init(shape=shape, dtype=DTYPE),
            trainable=True,
            name="weights"
        )

    def call(self, inputs):
        return self.w * self.runtime_coef


#-------------Layer Constant
class layer_const(layers.Layer):
    def __init__(self, x=None, fmaps=512):
        super(layer_const, self).__init__()

        w_init = tf.random_normal_initializer(mean=1.0, stddev=0.01, seed=0)   # add a bit of variation to avoid nan when noise is zero!
        self.w = tf.Variable(
            initial_value=w_init(shape=[fmaps, 4*4], dtype=DTYPE),
            trainable=True,
            name="const_weight"
        )

    def call(self, x):
        return tf.cast(tf.reshape(self.w, [-1, 512, 4, 4]), dtype=DTYPE)


#-------------Layer Dense
class layer_dense(layers.Layer):
    def __init__(self, x=None, fmaps=1, gain=1, use_wscale=False, lrmul=1.0, **kwargs):
        super(layer_dense, self).__init__()

        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
 
        # Equalized learning rate and custom learning rate multiplier.
        shape  = [x.shape[1], fmaps]
        fan_in = np.prod(shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
        he_std = gain / np.sqrt(fan_in)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.runtime_coef = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.runtime_coef = lrmul

        w_init = tf.random_normal_initializer(mean=0.0, stddev=init_std)
        self.w = tf.Variable(
            initial_value=w_init(shape=shape, dtype=DTYPE),
            trainable=True,
        )

    def call(self, x):
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
        return tf.matmul(x, tf.cast(self.w, DTYPE) * self.runtime_coef)


#-------------Layer Bias
class layer_bias(layers.Layer):
    def __init__(self, x=None, lrmul=1.0, **kwargs):
        super(layer_bias, self).__init__()

        b_init = tf.zeros_initializer()
        self.lrmul = lrmul
        self.b = tf.Variable(
            initial_value=b_init(shape=x.shape[1], dtype=DTYPE),
            trainable=True,
        )

    def call(self, x):
        if len(x.shape) == 2:
            return x + self.b*self.lrmul
        return x + tf.reshape(self.b*self.lrmul, [1, -1, 1, 1])



#---------------------------------------------------------------------
class layer_create_noise(layers.Layer):
    def __init__(self, xshape, ldx, randomize_noise, **kwargs):
        super(layer_create_noise, self).__init__(**kwargs)

        self.NSIZE = xshape[-2] * xshape[-1]
        self.N     = NC_NOISE
        self.N2    = int(self.N/2)
        self.T     = self.NSIZE-1
        self.Dt    = self.T/(self.NSIZE-1)
        self.t     = self.Dt*tf.cast(tf.random.uniform([self.NSIZE], maxval=self.NSIZE, dtype="int32"), DTYPE)
        self.t     = self.t[tf.newaxis,:]
        self.t     = tf.tile(self.t, [self.N2, 1])
        self.k     = tf.range(1,int(self.N2+1), dtype=DTYPE)
        self.f     = self.k/self.T
        self.f     = self.f[:,tf.newaxis]

        if (randomize_noise):
            c_init = tf.ones_initializer()
            self.c = tf.Variable(
                initial_value=c_init(shape=[1,self.N2], dtype=DTYPE),
                trainable=False,
                **kwargs
            )
        else:
            c_init = tf.ones_initializer()
            self.c = tf.Variable(
                initial_value=c_init(shape=[1,self.N2], dtype=DTYPE),
                trainable=True,
                **kwargs
            )


    def call(self, x, phi, scalingNoise=1.0):

        freq = self.f * self.t
        argsin = tf.math.sin(2*np.pi*freq + phi)
        noise = tf.matmul(self.c,argsin)
        noise = AMP_NOISE_MAX*scalingNoise*(2.0*(noise - tf.math.reduce_min(noise)) \
            /(tf.math.reduce_max(noise) - tf.math.reduce_min(noise)) - 1.0)
        noise = noise - tf.math.reduce_mean(noise)
        noise = tf.reshape(noise, shape=[x.shape[-2], x.shape[-1]])
        
        return noise





class layer_create_noise_z0(layers.Layer):
    def __init__(self, xshape, ldx, randomize_noise, **kwargs):
        super(layer_create_noise_z0, self).__init__(**kwargs)

        self.NSIZE = xshape[-2] * xshape[-1]
        self.N     = NC_NOISE
        self.N2    = int(self.N/2)
        self.T     = self.NSIZE-1
        self.Dt    = self.T/(self.NSIZE-1)
        self.t     = self.Dt*tf.cast(tf.random.uniform([self.NSIZE], maxval=self.NSIZE, dtype="int32"), DTYPE)
        self.t     = self.t[tf.newaxis,:]
        self.t     = tf.tile(self.t, [self.N2, 1])
        self.k     = tf.range(1,int(self.N2+1), dtype=DTYPE)
        self.f     = self.k/self.T
        self.f     = self.f[:,tf.newaxis]

        c_init = tf.ones_initializer()
        self.c = tf.Variable(
            initial_value=c_init(shape=[self.N2, self.NSIZE], dtype=DTYPE),
            trainable=True,
            **kwargs
        )

    def call(self, x, phi, scalingNoise=1.0):

        freq = self.f * self.t
        argsin = tf.math.sin(2*np.pi*freq + phi)
        noise = self.c * argsin
        noise = tf.math.reduce_sum(noise, axis=1)
        noise = AMP_NOISE_MAX*scalingNoise*(2.0*(noise - tf.math.reduce_min(noise)) \
            /(tf.math.reduce_max(noise) - tf.math.reduce_min(noise)) - 1.0)
        noise = noise - tf.math.reduce_mean(noise)
        noise = tf.reshape(noise, shape=[x.shape[-2], x.shape[-1]])
        
        return noise



class layer_noise(layers.Layer):
    def __init__(self, x, **kwargs):
        super(layer_noise, self).__init__(**kwargs)

        w_init = tf.ones_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=x.shape, dtype=DTYPE),
            trainable=True,
            **kwargs
        )

    def call(self, x):
        return tf.cast(self.w, x.dtype)



def apply_noise(x, ldx, phi_noise_in=None, randomize_noise=True):
    assert len(x.shape) == 4  # NCHW

    if phi_noise_in is None or randomize_noise:
        phi_noise = tf.random.uniform([tf.shape(x)[0], NC2_NOISE, 1], maxval=2.0*np.pi, dtype=x.dtype)
    else:
        phi_noise = tf.cast(phi_noise_in, x.dtype)

    # lcnoise = layer_create_noise([x.shape[-2], x.shape[-1]], ldx, randomize_noise, name="layer_noise_constants%d" % ldx)
    # noise   = lcnoise([x.shape[-2], x.shape[-1]], phi_noise)  # why passing the shape as argument is not working???

    lcnoise      = layer_create_noise([x.shape[-2], x.shape[-1]], ldx, randomize_noise, name="layer_noise_constants%d" % ldx)
    scalingNoise = 1.0 - ldx/(G_LAYERS-1)
    noise        = lcnoise(x, phi_noise, scalingNoise=scalingNoise)

    lnoise = layer_noise(noise, name="layer_noise_weights%d" % ldx)
    nweights = lnoise(x)
    
    return x + noise * nweights



#-------------Instance normalization
def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4  # NCHW
    orig_dtype = DTYPE
    x = tf.cast(x, DTYPE)
    x -= tf.math.reduce_mean(x, axis=[2, 3], keepdims=True)
    epsilon = tf.constant(epsilon, dtype=DTYPE)
    x *= tf.math.rsqrt(tf.math.reduce_mean(tf.math.square(x), axis=[2, 3], keepdims=True) + epsilon)
    x = tf.cast(x, orig_dtype)
    return x


#-------------Style mode layer
def style_mod(x, dlatent, **kwargs):
    dense = layer_dense(dlatent, fmaps=x.shape[1] * 2, gain=1, **kwargs)
    style = dense(dlatent)
    bias  = layer_bias(style, **kwargs)
    style = bias(style)
    style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
    return x * (style[:, 0] + 1) + style[:, 1]   # this is important: we add +1  to avoid to multiply by 0
                                                 # due to the normalization of style to mean 0.
                                                 # Note: style[:,1] = style[:,1,:,:]




#-------------gaussian filter
def gaussian_filter(field, rs=0, rsca=0):

    # separate DNS fields
    if (rs==0):
        rs = int(2**(RES_LOG2 - RES_LOG2-FIL))  # use suggested filter from FIL
    
    field = field[tf.newaxis,:,:,tf.newaxis]

    # prepare Gaussian Kernel
    gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.cast(gauss_kernel, dtype=field.dtype)

    # add padding
    pleft   = 4*rs
    pright  = 4*rs
    ptop    = 4*rs
    pbottom = 4*rs

    field = periodic_padding_flexible(field, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

    # convolve
    field = tf.nn.conv2d(field, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

    # downscale
    if (rsca>0):
        fU = field[0,::rsca,::rsca,0]
    else:
        fU = field[0,:,:,0]
        
    fU = fU[tf.newaxis, tf.newaxis, :, :]

    return fU



def gaussian_filter_withNorm(field, rs=0, rsca=0):

    # separate DNS fields
    if (rs==0):
        rs = int(2**(RES_LOG2 - RES_LOG2-FIL))  # use suggested filter from FIL
    
    field = field[tf.newaxis,:,:,tf.newaxis]

    # prepare Gaussian Kernel
    gauss_kernel = gaussian_kernel(4*rs, 0.0, rs)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.cast(gauss_kernel, dtype=field.dtype)

    # add padding
    pleft   = 4*rs
    pright  = 4*rs
    ptop    = 4*rs
    pbottom = 4*rs

    field = periodic_padding_flexible(field, axis=(1,2), padding=([pleft, pright], [ptop, pbottom]))

    # convolve
    field = tf.nn.conv2d(field, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

    # downscale
    if (rsca>0):
        fU = field[0,::rsca,::rsca,0]
    else:
        fU = field[0,:,:,0]
        
    # normalize
    fU = (fU - tf.math.reduce_min(fU))/(tf.math.reduce_max(fU) - tf.math.reduce_min(fU))*2 - 1

    fU = fU[tf.newaxis, tf.newaxis, :, :]

    return fU



#-------------Bluer filter
def _blur2d(x, f=[1, 2, 1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=DTYPE)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = DTYPE
    x = tf.cast(x, DTYPE)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=DTYPE, name="filter")
    strides = [1, 1, stride, stride]

    if DEVICE_TYPE in ('CPU','IPU'):
        strides = [1, stride, stride, 1]

    if (f.shape[0] % 2 == 0):
        pleft   = np.int((f.shape[0]-1)/2)
        pright  = np.int(f.shape[0]/2)
    else:
        pleft   = np.int((f.shape[0]-1)/2)
        pright  = pleft

    if (f.shape[1] % 2 == 0):
        ptop    = np.int((f.shape[1]-1)/2)
        pbottom = np.int(f.shape[1]/2)
    else:
        ptop    = np.int((f.shape[1]-1)/2)
        pbottom = ptop

    if (pleft*pright*ptop*pbottom>0):
        x = periodic_padding_flexible(x, axis=(2,3), padding=([pleft, pright], [ptop, pbottom]))
        x = tf.transpose(x, TRANSPOSE_FOR_CONV2D)
        x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding="VALID", data_format="NCHW")  # note as the padding here is VALID
        x = tf.transpose(x , TRANSPOSE_FROM_CONV2D)
    else:
        x = tf.transpose(x, TRANSPOSE_FOR_CONV2D)
        x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding="SAME", data_format=data_format)  # no need of padding as f is even
        x = tf.transpose(x , TRANSPOSE_FROM_CONV2D)
    
    x = tf.cast(x, orig_dtype)

    return x


@tf.custom_gradient
def func_blur2d(in_x):
    y = _blur2d(in_x)

    @tf.custom_gradient
    def grad(dy):
        dx = _blur2d(dy, flip=True)
        return dx, lambda ddx: _blur2d(ddx)

    return y, grad


class layer_blur2d(layers.Layer):
    def __init__(self, **kwargs):
        super(layer_blur2d, self).__init__()

    def call(self, x):
        return func_blur2d(x)







#-------------custom functions for upscale and downscale
def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x



def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32, float64 => downscale using _blur2d().
    if factor == 2 and (DTYPE == "float32"  or DTYPE == "float64"):
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW")




@tf.custom_gradient
def func_upscale2d(in_x, factor=2):
    y = _upscale2d(in_x, factor)

    @tf.custom_gradient
    def grad(dy, factor=2):
        dx = _downscale2d(dy, factor, gain=factor ** 2)
        return dx, lambda ddx: _upscale2d(ddx, factor)

    return y, grad


class layer_upscale2d(layers.Layer):
    def __init__(self, factor=2, **kwargs):
        super(layer_upscale2d, self).__init__()
        self.factor = factor

    def call(self, x):
        return func_upscale2d(x)





@tf.custom_gradient
def func_downscale2d(in_x, factor=2):
    y = _downscale2d(in_x, factor)

    @tf.custom_gradient
    def grad(dy, factor=2):
        dx = _upscale2d(dy, factor, gain=1 / factor ** 2)
        return dx, lambda ddx: _downscale2d(ddx, factor)

    return y, grad



class layer_downscale2d(layers.Layer):
    def __init__(self, factor=2, **kwargs):
        super(layer_downscale2d, self).__init__()
        self.factor = factor

    def call(self, x):
        return func_downscale2d(x)







#-------------Layer Conv2D
def conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    get_weight = layer_get_Weight([kernel, kernel, x.shape[1], fmaps], **kwargs)
    w = get_weight(x)
    w = tf.cast(w, DTYPE)
    strides=[1, 1, 1, 1]

    if (w.shape[0] % 2 == 0):
        pleft   = np.int((w.shape[0]-1)/2)
        pright  = np.int(w.shape[0]/2)
    else:
        pleft   = np.int((w.shape[0]-1)/2)
        pright  = pleft

    if (w.shape[1] % 2 == 0):
        ptop    = np.int((w.shape[1]-1)/2)
        pbottom = np.int(w.shape[1]/2)
    else:
        ptop    = np.int((w.shape[1]-1)/2)
        pbottom = ptop

    x = periodic_padding_flexible(x, axis=(2,3), padding=([pleft, pright], [ptop, pbottom]))

    x = tf.transpose(x, TRANSPOSE_FOR_CONV2D)
    
    x = tf.nn.conv2d(x, w, strides=strides, padding="VALID", data_format=data_format)
    
    x = tf.transpose(x , TRANSPOSE_FROM_CONV2D)
    return x


#-------------Upscale conv2d
def upscale2d_conv2d(x, fmaps, kernel, fused_scale="auto", **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, "auto"]
    if fused_scale == "auto":
        fused_scale = min(x.shape[2:]) * 2 >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        upscale2d = layer_upscale2d()
        x = upscale2d(x)
        x = conv2d(x, fmaps, kernel, **kwargs)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    get_weight = layer_get_Weight([kernel, kernel, x.shape[1], fmaps], **kwargs)
    w = get_weight(x)
    w = tf.transpose(w, [0, 1, 3, 2])  # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode="CONSTANT")
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, DTYPE)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(
        x, w, os, strides=[1, 1, 2, 2], padding="SAME", data_format="NCHW")



#-------------conv2d_downscale2d
def conv2d_downscale2d(x, fmaps, kernel, fused_scale="auto", **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, "auto"]
    if fused_scale == "auto":
        fused_scale = min(x.shape[2:]) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        downscale2d = layer_downscale2d()
        x = conv2d(x, fmaps, kernel, **kwargs)
        x = downscale2d(x)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv2d().
    get_weight = layer_get_Weight([kernel, kernel, x.shape[1], fmaps], **kwargs)
    w = get_weight(x)
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode="CONSTANT")
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, DTYPE)
    strides=[1, 1, 2, 2]

    if (w.shape[0] % 2 == 0):
        pleft   = np.int((w.shape[0]-1)/2)
        pright  = np.int(w.shape[0]/2)
    else:
        pleft   = np.int((w.shape[0]-1)/2)
        pright  = pleft

    if (w.shape[1] % 2 == 0):
        ptop    = np.int((w.shape[1]-1)/2)
        pbottom = np.int(w.shape[1]/2)
    else:
        ptop    = np.int((w.shape[1]-1)/2)
        pbottom = ptop

    x = periodic_padding_flexible(x, axis=(2,3), padding=([pleft, pright], [ptop, pbottom]))
    return tf.nn.conv2d(x, w, strides=strides, padding="VALID", data_format="NCHW")




#-------------Minibatch standard deviation.
def minibatch_stddev_layer(x, group_size=4, num_new_features=1):

    # Minibatch must be divisible by (or smaller than) group_size.
    group_size = tf.minimum(group_size, tf.shape(x)[0])
    s          = x.shape  # [NCHW]  Input shape.

    # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y  = tf.reshape(x, [group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[3]])
    y  = tf.cast(y, DTYPE)                            # [GMncHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)          # [GMncHW] Subtract mean over group.
    y  = tf.reduce_mean(tf.square(y), axis=0)              # [MncHW]  Calc variance over group.
    y  = tf.sqrt(y + 1e-8)                                 # [MncHW]  Calc stddev over group.
    y  = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [Mn111]  Take average over fmaps and pixels.
    y  = tf.reduce_mean(y, axis=[2])                       # [Mn11] Split channels into c channel groups
    y  = tf.cast(y, DTYPE)                               # [Mn11]  Cast back to original data type.
    y  = tf.tile(y, [group_size, 1, s[2], s[3]])           # [NnHW]  Replicate over group and pixels.

    return tf.concat([x, y], axis=1)                       # [NCHW]  Append as new fmap.





#-------------VGG loss
def VGG_loss(imgA, imgB, VGG_extractor):

    timgA   = tf.transpose(imgA, [0, 3, 2, 1])
    feaA_3  = VGG_extractor(timgA)[3]
    feaA_6  = VGG_extractor(timgA)[6]
    feaA_10 = VGG_extractor(timgA)[10]
    feaA_14 = VGG_extractor(timgA)[14]
    feaA_18 = VGG_extractor(timgA)[18]

    feaA_3, _  = tf.linalg.normalize(feaA_3,  axis=[-2, -1])  # the [-2, -1] is to make sure they are
    feaA_6, _  = tf.linalg.normalize(feaA_6,  axis=[-2, -1])  # normalized only by 2D matrix
    feaA_10, _ = tf.linalg.normalize(feaA_10, axis=[-2, -1])
    feaA_14, _ = tf.linalg.normalize(feaA_14, axis=[-2, -1])
    feaA_18, _ = tf.linalg.normalize(feaA_18, axis=[-2, -1])

    timgB   = tf.transpose(imgB, [0, 3, 2, 1])
    feaB_3  = VGG_extractor(timgB)[3]
    feaB_6  = VGG_extractor(timgB)[6]
    feaB_10 = VGG_extractor(timgB)[10]
    feaB_14 = VGG_extractor(timgB)[14]
    feaB_18 = VGG_extractor(timgB)[18]

    feaB_3, _  = tf.linalg.normalize(feaB_3,  axis=[-2, -1])  # the [-2, -1] is to make sure they are
    feaB_6, _  = tf.linalg.normalize(feaB_6,  axis=[-2, -1])  # normalized only by 2D matrix
    feaB_10, _ = tf.linalg.normalize(feaB_10, axis=[-2, -1])
    feaB_14, _ = tf.linalg.normalize(feaB_14, axis=[-2, -1])
    feaB_18, _ = tf.linalg.normalize(feaB_18, axis=[-2, -1])

    loss_fea_3  = tf.math.reduce_mean(tf.math.squared_difference(feaA_3,  feaB_3))
    loss_fea_6  = tf.math.reduce_mean(tf.math.squared_difference(feaA_6,  feaB_6))
    loss_fea_10 = tf.math.reduce_mean(tf.math.squared_difference(feaA_10, feaB_10))
    loss_fea_14 = tf.math.reduce_mean(tf.math.squared_difference(feaA_14, feaB_14))
    loss_fea_18 = tf.math.reduce_mean(tf.math.squared_difference(feaA_18, feaB_18))

    loss_pix = tf.math.reduce_mean(tf.math.squared_difference(imgA, imgB))/tf.math.reduce_mean(imgA**2)

    losses = []
    losses.append(loss_pix)
    losses.append(loss_fea_3)
    losses.append(loss_fea_6)
    losses.append(loss_fea_10)
    losses.append(loss_fea_14)
    losses.append(loss_fea_18)

    return losses






def VGG_loss_LES(imgA, imgB, VGG_extractor):

    timgA   = tf.transpose(imgA, [0, 3, 2, 1])
    feaA_3  = VGG_extractor(timgA)[3]
    feaA_6  = VGG_extractor(timgA)[6]
    feaA_10 = VGG_extractor(timgA)[10]
    feaA_14 = VGG_extractor(timgA)[14]
    feaA_18 = VGG_extractor(timgA)[18]

    feaA_3, _  = tf.linalg.normalize(feaA_3,  axis=[-2, -1])  # the [-2, -1] is to make sure they are
    feaA_6, _  = tf.linalg.normalize(feaA_6,  axis=[-2, -1])  # normalized only by 2D matrix
    feaA_10, _ = tf.linalg.normalize(feaA_10, axis=[-2, -1])
    feaA_14, _ = tf.linalg.normalize(feaA_14, axis=[-2, -1])
    feaA_18, _ = tf.linalg.normalize(feaA_18, axis=[-2, -1])

    timgB   = tf.transpose(imgB, [0, 3, 2, 1])
    feaB_3  = VGG_extractor(timgB)[3]
    feaB_6  = VGG_extractor(timgB)[6]
    feaB_10 = VGG_extractor(timgB)[10]
    feaB_14 = VGG_extractor(timgB)[14]
    feaB_18 = VGG_extractor(timgB)[18]

    feaB_3, _  = tf.linalg.normalize(feaB_3,  axis=[-2, -1])  # the [-2, -1] is to make sure they are
    feaB_6, _  = tf.linalg.normalize(feaB_6,  axis=[-2, -1])  # normalized only by 2D matrix
    feaB_10, _ = tf.linalg.normalize(feaB_10, axis=[-2, -1])
    feaB_14, _ = tf.linalg.normalize(feaB_14, axis=[-2, -1])
    feaB_18, _ = tf.linalg.normalize(feaB_18, axis=[-2, -1])

    loss_fea_3  = tf.math.reduce_mean(tf.math.squared_difference(feaA_3,  feaB_3))
    loss_fea_6  = tf.math.reduce_mean(tf.math.squared_difference(feaA_6,  feaB_6))
    loss_fea_10 = tf.math.reduce_mean(tf.math.squared_difference(feaA_10, feaB_10))
    loss_fea_14 = tf.math.reduce_mean(tf.math.squared_difference(feaA_14, feaB_14))
    loss_fea_18 = tf.math.reduce_mean(tf.math.squared_difference(feaA_18, feaB_18))

    loss_pix = tf.math.reduce_mean(tf.math.squared_difference(imgA, imgB))/tf.math.reduce_mean(imgA**2)

    losses = []
    losses.append(loss_pix)
    losses.append(loss_fea_3)
    losses.append(loss_fea_6)
    losses.append(loss_fea_10)
    losses.append(loss_fea_14)
    losses.append(loss_fea_18)

    return losses    



def tf_find_vorticity(U, V):
    W = ((tr(V, 1, 0)-tr(V, -1, 0)) - (tr(U, 0, 1)-tr(U, 0, -1)))
    return W



def normalize_sc(U):
 
    U_min = tf.reduce_min(U)
    U_max = tf.reduce_max(U)
    U = 2.0*(U - U_min)/(U_max - U_min) - 1.0

    return U


def normalize(UVP):

    tU = UVP[0,0,:,:]
    tV = UVP[0,1,:,:]
    tP = UVP[0,2,:,:]

    U_min = tf.reduce_min(tU)
    V_min = tf.reduce_min(tV)
    P_min = tf.reduce_min(tP)

    U_max = tf.reduce_max(tU)
    V_max = tf.reduce_max(tV)
    P_max = tf.reduce_max(tP)

    tU = 2.0*(UVP[0,0,:,:] - U_min)/(U_max - U_min) - 1.0
    tV = 2.0*(UVP[0,1,:,:] - V_min)/(V_max - V_min) - 1.0
    tP = 2.0*(UVP[0,2,:,:] - P_min)/(P_max - P_min) - 1.0

    tU = tU[tf.newaxis,tf.newaxis,:,:]
    tV = tV[tf.newaxis,tf.newaxis,:,:]
    tP = tP[tf.newaxis,tf.newaxis,:,:]   

    UVP = tf.concat([tU, tV, tP], 1)

    return UVP



def rescale(UVP, UVP_minmax):
    
    U_min = UVP_minmax[0]
    U_max = UVP_minmax[1]
    V_min = UVP_minmax[2]
    V_max = UVP_minmax[3]
    P_min = UVP_minmax[4]
    P_max = UVP_minmax[5]
        
    tU = UVP[0,0,:,:]
    tV = UVP[0,1,:,:]
    tP = UVP[0,2,:,:]
    
    u_min =  tf.reduce_min(tU)
    v_min =  tf.reduce_min(tV)
    p_min =  tf.reduce_min(tP)

    tU = (tU - u_min)/(tf.reduce_max(tU) - u_min)
    tV = (tV - v_min)/(tf.reduce_max(tV) - v_min)
    tP = (tP - p_min)/(tf.reduce_max(tP) - p_min)

    tU = (U_max - U_min)*tU + U_min
    tV = (V_max - V_min)*tV + V_min
    tP = (P_max - P_min)*tP + P_min    

    tU = tU[tf.newaxis,tf.newaxis,:,:]
    tV = tV[tf.newaxis,tf.newaxis,:,:]
    tP = tP[tf.newaxis,tf.newaxis,:,:]

    UVP = tf.concat([tU, tV, tP], 1)
    
    return UVP



#---------------------------------------------- Layers search latent space
class layer_zlatent_kDNS(layers.Layer):
    def __init__(self, **kwargs):
        super(layer_zlatent_kDNS, self).__init__()

        k_init = tf.random_normal_initializer(mean=0.6, stddev=0.0)
        self.k = tf.Variable(
            initial_value=k_init(shape=[M_LAYERS, LATENT_SIZE], dtype=DTYPE),
            trainable=True,
            name="zlatent_kDNS"
        )

    def call(self, mapping, z):

        # interpolate latent spaces
        zn = self.k[0,:]*z[:,0,:] + (1.0-self.k[0,:])*z[:,1,:]
        zn = zn[:,tf.newaxis,:]
        for i in range(M_LAYERS):
            zs = self.k[i,:]*z[:,2*i,:] + (1.0-self.k[i,:])*z[:,2*i+1,:]
            zs = zs[:,tf.newaxis,:]
            zn = tf.concat([zn,zs], axis=1)

        zf = z[:,2*M_LAYERS,:]   # latent space for final layers
        zf = zf[:,tf.newaxis,:]
        zn = tf.concat([zn,zf], axis=1)

        # map interpolated latent space z into w
        w0 = mapping(zn[:,M_LAYERS,:], training=False)
        wn = mapping(zn[:,0,:], training=False)
        wn = wn[:,0:1,:]
        for i in range(1,M_LAYERS):
            ws = mapping(zn[:,i,:], training=False)
            ws = ws[:,i:i+1,:]
            wn = tf.concat([wn,ws], axis=1)

        w0 = w0[:,M_LAYERS:G_LAYERS,:]
        wn = tf.concat([wn,w0], axis=1)
        
        return wn



class layer_zlatent_kDNS2(layers.Layer):
    def __init__(self, **kwargs):
        super(layer_zlatent_kDNS2, self).__init__()

    def call(self, mapping, zn):

        # map interpolated latent space z into w
        w0 = mapping(zn[:,M_LAYERS,:], training=False)
        wn = mapping(zn[:,0,:], training=False)
        wn = wn[:,0:1,:]
        for i in range(1,M_LAYERS):
            ws = mapping(zn[:,i,:], training=False)
            ws = ws[:,i:i+1,:]
            wn = tf.concat([wn,ws], axis=1)

        w0 = w0[:,M_LAYERS:G_LAYERS,:]
        wn = tf.concat([wn,w0], axis=1)
        
        return wn



class layer_wlatent_mLES(layers.Layer):
    def __init__(self, **kwargs):
        super(layer_wlatent_mLES, self).__init__()

        m_init = tf.random_normal_initializer(mean=0.5, stddev=0.0)
        self.m = tf.Variable(
            initial_value=m_init(shape=[M_LAYERS, LATENT_SIZE], dtype=DTYPE),
            trainable=True,
            name="wlatent_mDNS"
        )

    def call(self, w0, w1):

        w_LES = self.m*w0[:,0:M_LAYERS,:] + (1.0-self.m)*w1[:,0:M_LAYERS,:]
        w_DNS = w1[:,M_LAYERS:G_LAYERS,:] 
        w = tf.concat([w_LES,w_DNS], axis=1)
        
        return w



@tf.function
def find_centred_fields(UVP):

        # make sure average of each field is zero
        U = UVP[0,0,:,:]
        V = UVP[0,1,:,:]
        P = UVP[0,2,:,:]

        Um = tf.reduce_mean(U)
        Vm = tf.reduce_mean(V)
        Pm = tf.reduce_mean(P)
        
        U = U - Um
        V = V - Vm
        P = P - Pm

        U = U[tf.newaxis,tf.newaxis,:,:]
        V = V[tf.newaxis,tf.newaxis,:,:]
        P = P[tf.newaxis,tf.newaxis,:,:]

        UVP = tf.concat([U, V, P], 1)
        
        return UVP



@tf.function
def find_predictions(synthesis, filter, z):

    # find predictions
    inference = synthesis(z, training=False)
    
    if (len(inference)==2):
        predictions = inference[0]
        wn = inference[1]
    else:
        predictions = inference
        wn = tf.identity(w0)
        
    UVP_DNS = predictions[RES_LOG2-2]
    UVP_LES = predictions[RES_LOG2-FIL-2]


    # adjust DNS fields
    if (TESTCASE=='HIT_2D'):

        # calculate vorticity 
        U_DNS = UVP_DNS[0,0,:,:]
        V_DNS = UVP_DNS[0,1,:,:]
        P_DNS = UVP_DNS[0,2,:,:]

        Pmin  = tf.reduce_min(P_DNS)
        Pmax  = tf.reduce_max(P_DNS)

        Pc_DNS = tf_find_vorticity(U_DNS, V_DNS)
        Pcmin  = tf.reduce_min(Pc_DNS)
        Pcmax  = tf.reduce_max(Pc_DNS)

        P_DNS = (P_DNS - Pmin)/(Pmax - Pmin)*(Pcmax - Pcmin) + Pcmin

        U_DNS = U_DNS[tf.newaxis,tf.newaxis,:,:]
        V_DNS = V_DNS[tf.newaxis,tf.newaxis,:,:]
        P_DNS = P_DNS[tf.newaxis,tf.newaxis,:,:]

        UVP_DNS = tf.concat([U_DNS, V_DNS, P_DNS], 1)

    elif (TESTCASE=='HW' or TESTCASE=='mHW'):

        find_centred_fields
        UVP_DNS = find_centred_fields(UVP_DNS)


    # adjust LES fields
    if (TESTCASE=='HIT_2D'):

        # calculate vorticity 
        U_LES = UVP_LES[0,0,:,:]
        V_LES = UVP_LES[0,1,:,:]
        P_LES = tf_find_vorticity(U_LES, V_LES)

        U_LES = U_LES[tf.newaxis,tf.newaxis,:,:]
        V_LES = V_LES[tf.newaxis,tf.newaxis,:,:]
        P_LES = P_LES[tf.newaxis,tf.newaxis,:,:]

        UVP_LES = tf.concat([U_LES, V_LES, P_LES], 1)    

    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        
        # find_centred_fields
        UVP_LES = find_centred_fields(UVP_LES)


    # find filtered fields
    fU_DNS = UVP_DNS[:,0,:,:]
    fU_DNS = fU_DNS[:,tf.newaxis,:,:]
    fU_DNS = filter(fU_DNS, training = False)

    fV_DNS = UVP_DNS[:,1,:,:]
    fV_DNS = fV_DNS[:,tf.newaxis,:,:]
    fV_DNS = filter(fV_DNS, training = False)

    if (TESTCASE=='HIT_2D'):
        fP_DNS = tf_find_vorticity(fU_DNS[0,0,:,:], fV_DNS[0,0,:,:])
        fP_DNS = fP_DNS[tf.newaxis,tf.newaxis,:,:]
    elif (TESTCASE=='HW' or TESTCASE=='mHW'):
        fP_DNS = UVP_DNS[:,2,:,:]
        fP_DNS = fP_DNS[:,tf.newaxis,:,:]
        fP_DNS = filter(fP_DNS, training = False)
    
    fUVP_DNS = tf.concat([fU_DNS, fV_DNS, fP_DNS], axis=1)


    return UVP_DNS, UVP_LES, fUVP_DNS, wn, predictions



@tf.function
def find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, tDNS, tLES, typeRes=0):

    if (typeRes==0):  # residuals for create_restart
        resDNS   = tf.math.reduce_mean(tf.math.squared_difference(UVP_DNS,  tDNS))
        resLES   = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, tLES))
        resREC   = resDNS + resLES
        loss_fil = resLES
    elif (typeRes==1):  # residuals for create_restart
        resDNS   = tf.math.reduce_mean(tf.math.squared_difference(fUVP_DNS, tLES))
        resLES   = 0.0*tf.math.reduce_mean(tf.math.squared_difference(UVP_LES,  tLES))
        resREC   = resDNS + resLES
        loss_fil = resLES        

    return resREC, resLES, resDNS, loss_fil



@tf.function
def step_find_zlatents_kDNS(synthesis, filter, opt, z, tDNS, tLES, ltv, typeRes):
    with tf.GradientTape() as tape_LES:
        
        # find predictions
        UVP_DNS, UVP_LES, fUVP_DNS, wn, _ = find_predictions(synthesis, filter, z)

        # find residuals        
        resREC, resLES, resDNS, loss_fil = find_residuals(UVP_DNS, UVP_LES, fUVP_DNS, tDNS, tLES, typeRes=typeRes)
        
        # apply gradients
        gradients_LES = tape_LES.gradient(resREC, ltv)
        opt.apply_gradients(zip(gradients_LES, ltv))

        
    return UVP_DNS, UVP_LES, fUVP_DNS, resREC, resLES, resDNS, loss_fil, wn
