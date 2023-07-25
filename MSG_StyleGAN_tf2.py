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
import tensorflow as tf
import numpy as np
import os
import PIL
import time
import sys
import pathlib
import datetime

from parameters import *
from functions import *
from tensorflow.keras import layers, initializers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


#---------------------------define mapping
def make_mapping_model():

    # Options
    use_pixel_norm    = True      # Enable pixelwise feature vector normalization
    use_wscale        = True      # Enable equalized learning rate


    # Inputs
    latents_in = tf.keras.Input(shape=([LATENT_SIZE]), dtype=DTYPE)


    # Normalize
    if (use_pixel_norm):
        latents = pixel_norm(latents_in)
    else:
        latents = latents_in


    # Mapping layers.
    for ldx in range(8):
        dense   = layer_dense(latents, fmaps=LATENT_SIZE, gain=GAIN, use_wscale=use_wscale, lrmul=GM_LRMUL, name='dlatent_dense_%d' % ldx)
        latents = dense(latents)
        bias    = layer_bias(latents, lrmul=GM_LRMUL, name='dlatent_bias_%d' % ldx)
        latents = bias(latents)
        latents = layers.LeakyReLU()(latents)


    # create extended w+ latent space
    dlatents = tf.tile(latents[:, np.newaxis], [1, G_LAYERS, 1])

    mapping_model = Model(inputs=latents_in, outputs=dlatents)

    return mapping_model


#---------------------------define synthesis
def make_synthesis_model():

    # Options
    use_pixel_norm    = True         # Disable pixelwise feature vector normalization
    use_wscale        = True         # Enable equalized learning rate
    use_instance_norm = True         # Enable instance normalization
    use_noise         = True         # Enable noise inputs
    randomize_noise   = RANDOMIZE_NOISE  # True = randomize noise inputs every time (non-deterministic),
                                    # False = read noise inputs from variables.
    use_styles        = True         # Enable style inputs                             
    blur_filter       = BLUR_FILTER  # Low-pass filter to apply when resampling activations. 
                                    # None = no filtering.
    fused_scale       = False        # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.


    # Inputs
    dlatents = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)


    # Noise inputs
    noise_inputs = []
    for ldx in range(G_LAYERS):

        phi_init = tf.random_uniform_initializer(minval=0.0, maxval=2.0*np.pi, seed=ldx)
        phi_noise = tf.Variable(
            initial_value=phi_init([NC2_NOISE,1], dtype=DTYPE),
            trainable=False,
            name="input_phi_noise%d" % ldx,
            )

        noise_inputs.append(phi_noise)


    # Things to do at the end of each layer.
    def layer_epilogue(in_x, ldx):
        if use_noise:
            in_x = apply_noise(in_x, ldx, noise_inputs[ldx], randomize_noise=randomize_noise)

        bias = layer_bias(in_x)
        in_x = bias(in_x)
        in_x = layers.LeakyReLU()(in_x)
        if use_pixel_norm:
            in_x = pixel_norm(in_x)
        if use_instance_norm:
            in_x = instance_norm(in_x)
        if use_styles:
            in_x = style_mod(in_x, 
                            dlatents[:, ldx],
                            use_wscale=use_wscale,
                            name = "style_%d" % ldx)
        return in_x


    # define blur
    def blur(in_x):
        if blur_filter:
            blur2d = layer_blur2d()
            fx = blur2d(in_x)
        else:
            fx = in_x
        return fx


    # Early layers: we start from a constant input
    const = layer_const(dlatents)
    x = const(dlatents)
    x = tf.tile(x, [tf.shape(dlatents)[0], 1, 1, 1])
    x = layer_epilogue(x, 0)
    x = conv2d(x, fmaps=nf(1), kernel=3, gain=GAIN, use_wscale=use_wscale, name="Conv")
    x = layer_epilogue(x, 1)


    # Building blocks for remaining layers.
    def block(in_res, in_x):  # res = 3..RES_LOG2
        in_x = layer_epilogue(
            blur(
                upscale2d_conv2d(in_x,
                    fmaps=nf(in_res - 1),
                    kernel=3,
                    gain=GAIN,
                    use_wscale=use_wscale,
                    fused_scale=fused_scale,
                )
            ),
            in_res * 2 - 4,
        )
        in_x = layer_epilogue(
            conv2d(
                in_x,
                fmaps=nf(in_res - 1),
                kernel=3,
                gain=GAIN,
                use_wscale=use_wscale,
            ),
            in_res * 2 - 3,
        )
        return in_x


    # convert to RGB
    def torgb(in_res, in_x):  # res = 2 -> RES_LOG2-FIL
        in_lod = RES_LOG2 - in_res
        x = conv2d(in_x, fmaps=NUM_CHANNELS, kernel=1, gain=1, use_wscale=use_wscale, name ="ToRGB_lod%d" % in_lod)
        bias = layer_bias(x, name ="ToRGB_bias_lod%d" % in_lod)
        x = bias(x)
        x = normalize(x)
        return x

    # convert to RGB final
    def torgb_final(in_res, in_x):  # res = RES_LOG2-FIL -> RES_LOG2
        in_lod = RES_LOG2 - in_res
        x = conv2d(in_x, fmaps=NUM_CHANNELS, kernel=1, gain=1, use_wscale=use_wscale, name ="ToRGB_lod%d" % in_lod)
        bias = layer_bias(x, name ="ToRGB_bias_lod%d" % in_lod)
        x = bias(x)
        x_R = gaussian_filter(x[0,0,:,:], rs=1)
        x_G = gaussian_filter(x[0,1,:,:], rs=1)
        x_B = gaussian_filter(x[0,2,:,:], rs=1)
        x = tf.concat([x_R, x_G, x_B], axis=1)
        x = normalize(x)
        return x

    # Finally, arrange the computations for the layers
    images_out = []  # list will contain the output images at different resolutions
    images_out.append(torgb(2, x))
    for res in range(3, RES_LOG2):
        x = block(res, x)
        images_out.append(torgb(res, x))

    for res in range(RES_LOG2, RES_LOG2+1):
        x = block(res, x)
        images_out.append(torgb_final(res, x))
        
    synthesis_model = Model(inputs=dlatents, outputs=images_out)

    return synthesis_model




#-------------------------------------define filter
def make_filter_model(f_res, t_res):

    # inner parameters
    use_wscale  = True        # Enable equalized learning rate
    blur_filter = BLUR_FILTER # Low-pass filter to apply when resampling activations. 
                            # None = no filtering.
    fused_scale = False       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.

    # inner functions
    def blur(in_x):
        if blur_filter:
            blur2d = layer_blur2d()
            fx = blur2d(in_x)
        else:
            fx = in_x
        return fx

    def torgb(in_res, in_x):  # res = 2..RES_LOG2
        in_lod = RES_LOG2 - in_res
        x = conv2d(in_x, fmaps=1, kernel=1, gain=1, use_wscale=use_wscale, name ="ToRGB_lod%d" % in_lod)
        bias = layer_bias(x, name ="ToRGB_bias_lod%d" % in_lod)
        x = bias(x)
        return x

    # create model
    f_in = tf.keras.Input(shape=([1, OUTPUT_DIM, OUTPUT_DIM]), dtype=DTYPE)

    f_x = tf.identity(f_in)
    for res in range(f_res, t_res, -1):
        f_x = conv2d(f_x, fmaps=nf(res - 1), kernel=3, gain=GAIN, use_wscale=use_wscale, name="filter_conv_" + str(res))
        bias = layer_bias(f_x, name="filter_bias_0")
        f_x = bias(f_x)
        f_x = layers.LeakyReLU()(f_x)

        f_x = blur(f_x)
        f_x = conv2d_downscale2d(f_x, fmaps=nf(res - 2), kernel=3, gain=GAIN,
                use_wscale=use_wscale, fused_scale=fused_scale, name="filter_conv_1")
        bias = layer_bias(f_x, name="filter_bias_1")
        f_x = bias(f_x)
        f_x = layers.LeakyReLU()(f_x)
        f_x = torgb(res-1, f_x)

    fout_x = tf.identity(f_x)

    # Create model
    filter_model = Model(inputs=f_in, outputs=fout_x)

    return filter_model 




#-------------------------------------define discriminator
def make_discriminator_model():

    use_wscale         = True        # Enable equalized learning rate
    label_size         = 0           # Dimensionality of the labels, 0 if no labels. Overridden based on dataset
    mbstd_group_size   = 4           # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features = 1           # Number of features for the minibatch standard deviation layer.
    blur_filter        = BLUR_FILTER # Low-pass filter to apply when resampling activations. 
                                        # None = no filtering.
    fused_scale        = False       # True = fused convolution + scaling, 
                                        # False = separate ops, 'auto' = decide automatically.

    def blur(in_x):
        if blur_filter:
            blur2d = layer_blur2d()
            fx = blur2d(in_x)
        else:
            fx = in_x
        return fx

    def conv1x1(x_in, fmaps):
        return conv2d(x_in, fmaps, kernel=1, use_wscale=use_wscale)

    images_in = []

    for res in range(2, RES_LOG2 + 1):
        image = tf.keras.Input(shape=([NUM_CHANNELS, (2 ** res), (2 ** res)]), dtype=DTYPE)
        images_in.append(image)

    # Building blocks.
    def fromrgb(in_x, in_res, full_maps=False):  # res = 2..RES_LOG2
        if full_maps:
            tail = "FromRGB_lod%d" % (RES_LOG2 - in_res)
            in_x = conv2d(in_x, fmaps=nf(in_res - 1), kernel=1, use_wscale=use_wscale, name="Conv_" + tail)
            bias = layer_bias(in_x, name="Bias_" + tail)
            in_x = bias(in_x)
            in_x = layers.LeakyReLU()(in_x)
            return in_x

        tail = "lod%d" % (RES_LOG2 - in_res)
        in_x = conv2d(in_x, fmaps=nf(in_res - 1) // 2, kernel=1, use_wscale=use_wscale, name="Conv_" + tail)
        bias = layer_bias(in_x, name="Bias_" + tail)
        in_x = bias(in_x)
        in_x = layers.LeakyReLU()(in_x)
        return in_x

    def block(in_x, in_res, g_img=None):  # res = 2..RES_LOG2

        if g_img is not None:  # the combine function is a learnable 1x1 conv layer
            in_x = conv1x1(tf.concat((in_x, g_img), axis=1), nf(in_res - 1))

        if mbstd_group_size > 1:
            in_x = minibatch_stddev_layer(in_x, mbstd_group_size, mbstd_num_features)

        if in_res >= 3:  # 8x8 and up
            in_x = conv2d(in_x, fmaps=nf(in_res - 1), kernel=3, gain=GAIN, use_wscale=use_wscale, name="Conv0")
            bias = layer_bias(in_x, name="Bias0")
            in_x = bias(in_x)
            in_x = layers.LeakyReLU()(in_x)

            in_x = blur(in_x)
            in_x = conv2d_downscale2d(in_x, fmaps=nf(in_res - 2), kernel=3, gain=GAIN,
                use_wscale=use_wscale, fused_scale=fused_scale, name="Conv1_down")
            bias = layer_bias(in_x, name="Bias_down")
            in_x = bias(in_x)
            in_x = layers.LeakyReLU()(in_x)

        else:  # 4x4

            in_x = conv2d(in_x, fmaps=nf(in_res - 1), kernel=3, gain=GAIN, use_wscale=use_wscale, name="Conv")
            bias = layer_bias(in_x, name="Bias")
            in_x = bias(in_x)
            in_x = layers.LeakyReLU()(in_x)

            dense = layer_dense(in_x, fmaps=nf(in_res - 2), gain=GAIN, use_wscale=use_wscale, name="Dense0")
            in_x  = dense(in_x)
            bias  = layer_bias(in_x, name="Bias0")
            in_x  = bias(in_x)
            in_x  = layers.LeakyReLU()(in_x)

            dense = layer_dense(in_x, fmaps=max(label_size, 1), gain=1, use_wscale=use_wscale, name="Dense1")
            in_x  = dense(in_x)
            bias  = layer_bias(in_x, name="Bias1")
            in_x  = bias(in_x)

        return in_x


    # Fixed structure: simple and efficient, but does not support progressive growing.
    x = fromrgb(images_in[-1], RES_LOG2, full_maps=True)
    x = block(x, RES_LOG2)
    for (img, res) in zip(reversed(images_in[:-1]), range(RES_LOG2 - 1, 2, -1)):
        x = block(x, res, img)

    scores_out = block(x, 2, images_in[0])

    assert scores_out.dtype == tf.as_dtype(DTYPE)
    scores_out = tf.identity(scores_out, name="scores_out")


    # Create model
    discriminator_model = Model(inputs=images_in, outputs=scores_out)

    return discriminator_model    



#-------------------------------------define optimizer and loss functions
lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR_GEN,
    decay_steps=DECAY_STEPS_GEN,
    decay_rate=DECAY_RATE_GEN,
    staircase=STAIRCASE_GEN)

lr_schedule_fil = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR_FIL,
    decay_steps=DECAY_STEPS_FIL,
    decay_rate=DECAY_RATE_FIL,
    staircase=STAIRCASE_FIL)

lr_schedule_dis = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR_DIS,
    decay_steps=DECAY_STEPS_DIS,
    decay_rate=DECAY_RATE_DIS,
    staircase=STAIRCASE_DIS)

generator_optimizer     = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen, beta_1=BETA1_GEN, beta_2=BETA2_GEN, epsilon=SMALL)
filter_optimizer        = tf.keras.optimizers.Adam(learning_rate=lr_schedule_fil, beta_1=BETA1_FIL, beta_2=BETA2_FIL, epsilon=SMALL)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_dis, beta_1=BETA1_DIS, beta_2=BETA2_DIS, epsilon=SMALL)



#-------------------------------------create an instance of the generator and discriminator
mapping       = make_mapping_model()
synthesis     = make_synthesis_model()
discriminator = make_discriminator_model()

filter = make_filter_model(RES_LOG2, RES_LOG2-FIL)


#mapping.summary()
#synthesis.summary()
#discriminator.summary()
#for fil in range(RES_LOG2-RES_LOG2-FIL):
    #filter[fil].summary()


#plot_model(mapping,       to_file='images/mapping_graph.png',       show_shapes=True, show_layer_names=True)
#plot_model(synthesis,     to_file='images/synthesis_graph.png',     show_shapes=True, show_layer_names=True)
#plot_model(filter,        to_file='images/filter_graph.png',        show_shapes=True, show_layer_names=True)
#plot_model(discriminator, to_file='images/discriminator_graph.png', show_shapes=True, show_layer_names=True)


#-------------------------------------define checkpoint
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    filter_optimizer=filter_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    mapping=mapping,
                                    synthesis=synthesis,
                                    filter=filter,
                                    discriminator=discriminator)

def gradient_penalty(x):
    with tf.GradientTape() as t:
        t.watch(x)
        d_hat = discriminator(x, training=False)
        d_hat = d_hat * SCALING_UP   #loss scaling. Important for mixed precision training
    gradients = t.gradient(d_hat, x)
    r1_penalty = tf.reduce_sum(tf.square(tf.cast(gradients[-1], DTYPE)*SCALING_DOWN), axis=[1, 2, 3])
    for grad in gradients:
        r1_penalty = r1_penalty + tf.reduce_sum(tf.square(tf.cast(grad, DTYPE)*SCALING_DOWN), axis=[1, 2, 3])
    return r1_penalty



# find lists of coarse, medium and fine tunable noises
ltv_DNS = []
ltv_LES = []

for layer in synthesis.layers:
    if "layer_noise_constants" in layer.name:
        lname = layer.name
        ldx = int(lname.replace("layer_noise_constants",""))
        for variable in layer.trainable_variables:
            if (ldx<M_LAYERS):
                ltv_LES.append(variable)
            ltv_DNS.append(variable)

