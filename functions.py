#----------------------------------------------------------------------------------------------
#
#    Copyright: STFC - Hartree Centre (2021)
#
#    Author: Jony Castagna
#
#    Licence: most of this material is taken from StyleGAN and MSG-StyleGAN. Please use same
#             licence policy
#
#-----------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

from tensorflow.keras import layers
from typing import Union

from parameters import *


# sys.path.insert(n, item) inserts the item at the nth position in the list 
# (0 at the beginning, 1 after the first element, etc ...)
sys.path.insert(0, './LES_Solvers/')
sys.path.insert(0, './LES_Solvers/testcases/HIT_2D')
sys.path.insert(0, '../TurboGenPY/')

from LES_constants import hf
from HIT_2D import L, rho, nu


#A type that represents a valid Tensorflow expression
TfExpression = Union[tf.Tensor, tf.Variable, tf.Operation]

#A type that can be converted to a valid Tensorflow expression
TfExpressionEx = Union[TfExpression, int, float, np.ndarray]


#------------------------------ general functions
def tr(phi, i, j):
    return tf.roll(phi, (-i, -j), axis=(0,1))

def nr(phi, i, j):
    return np.roll(phi, (-i, -j), axis=(0,1))



#------------------------------ functions to build StyleGAN network

# define periodic padding
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

#-------------linear interpolation
def lerp(a: TfExpressionEx, b: TfExpressionEx, t: TfExpressionEx) -> TfExpressionEx:
    with tf.name_scope("Lerp"):
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

        w_init = tf.ones_initializer()
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



#-------------Layer Noise
class layer_noise(layers.Layer):
    def __init__(self, x, shape, **kwargs):
        super(layer_noise, self).__init__()

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=shape, dtype=DTYPE),
            trainable=True,
            name="Noise_init"
        )

    def call(self, x):
        return tf.cast(self.w, DTYPE)


class apply_noise(layers.Layer):
    def __init__(self, x, noise_var=None, randomize_noise=True, **kwargs):
        super(apply_noise, self).__init__()

        w_init = tf.zeros_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=x.shape[1], dtype=DTYPE),
            trainable=True,
            name="Noise_weight"
        )

    def call(self, x, noise):
        return x + noise * tf.reshape(tf.cast(self.w, DTYPE), [1, -1, 1, 1])



#-------------Layer wlatent
class layer_wlatent(layers.Layer):
    def __init__(self, x=None, **kwargs):
        super(layer_wlatent, self).__init__()

        wl_init = tf.ones_initializer()
        self.wl = tf.Variable(
            initial_value=wl_init(shape=x.shape[1:], dtype=DTYPE),
            trainable=True,
            name="wlatent"
        )

    def call(self, x):
        return x*self.wl


#-------------Layer wlatent variable
class layer_wlatent_variable(layers.Layer):
    def __init__(self, x=None, **kwargs):
        super(layer_wlatent, self).__init__()

        wl_init = tf.ones_initializer()
        self.wl = tf.Variable(
            initial_value=wl_init(shape=(7, 512), dtype=DTYPE),
            trainable=True,
            name="wlatent"
        )

        self.nwl = tf.Variable(
            initial_value=wl_init(shape=(7, 512), dtype=DTYPE),
            trainable=False,
            name="nwlatent"
        )

    def call(self, x):
        x = tf.concat([x[:,0:7,:]*self.wl, x[:,7:G_LAYERS,:]*self.nwl], 1) 
        return x


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


#-------------Filter
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

    # if (f.shape[0] % 2 == 0):
    #     pleft   = np.int((f.shape[0]-1)/2)
    #     pright  = np.int(f.shape[0]/2)
    # else:
    #     pleft   = np.int((f.shape[0]-1)/2)
    #     pright  = pleft

    # if (f.shape[1] % 2 == 0):
    #     ptop    = np.int((f.shape[1]-1)/2)
    #     pbottom = np.int(f.shape[1]/2)
    # else:
    #     ptop    = np.int((f.shape[1]-1)/2)
    #     pbottom = ptop

    # x2 = periodic_padding_flexible(x, axis=(2,3), padding=([pleft, pright], [ptop, pbottom]))
    # x2 = tf.nn.depthwise_conv2d(x2, f, strides=strides, padding="VALID", data_format="NCHW")
    # x2 = tf.cast(x2, orig_dtype)

    x = tf.nn.depthwise_conv2d(
        x, f, strides=strides, padding="SAME", data_format="NCHW"
    )
    x = tf.cast(x, orig_dtype)
    return x


def blur2d(x, f=[1, 2, 1], normalize=True):
    @tf.custom_gradient
    def func(in_x):
        y = _blur2d(in_x, f, normalize)

        @tf.custom_gradient
        def grad(dy):
            dx = _blur2d(dy, f, normalize, flip=True)
            return dx, lambda ddx: _blur2d(ddx, f, normalize)

        return y, grad

    return func(x)


#-------------Upscale2D
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


def upscale2d(x, factor=2):
    @tf.custom_gradient
    def func(in_x):
        y = _upscale2d(in_x, factor)

        @tf.custom_gradient
        def grad(dy):
            dx = _downscale2d(dy, factor, gain=factor ** 2)
            return dx, lambda ddx: _upscale2d(ddx, factor)

        return y, grad

    return func(x)


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
    return tf.nn.conv2d(x, w, strides=strides, padding="VALID", data_format="NCHW")


#-------------Upscale conv2d
def upscale2d_conv2d(x, fmaps, kernel, fused_scale="auto", **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, "auto"]
    if fused_scale == "auto":
        fused_scale = min(x.shape[2:]) * 2 >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return conv2d(upscale2d(x), fmaps, kernel, **kwargs)

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


#-------------conv2d_downscale2d
def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and DTYPE == tf.float32:
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


def downscale2d(x, factor=2):
    @tf.custom_gradient
    def func(in_x):
        y = _downscale2d(in_x, factor)

        @tf.custom_gradient
        def grad(dy):
            dx = _upscale2d(dy, factor, gain=1 / factor ** 2)
            return dx, lambda ddx: _downscale2d(ddx, factor)

        return y, grad

    return func(x)


def conv2d_downscale2d(x, fmaps, kernel, fused_scale="auto", **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, "auto"]
    if fused_scale == "auto":
        fused_scale = min(x.shape[2:]) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return downscale2d(conv2d(x, fmaps, kernel, **kwargs))

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

    loss_pix = tf.math.reduce_mean(tf.math.squared_difference(imgA, imgB))
    loss_fea = loss_fea_3 + loss_fea_6 + loss_fea_10 + loss_fea_14 + loss_fea_18

    losses = []
    losses.append(loss_pix)
    losses.append(loss_fea)
    losses.append(loss_fea_3)
    losses.append(loss_fea_6)
    losses.append(loss_fea_10)
    losses.append(loss_fea_14)
    losses.append(loss_fea_18)

    return losses