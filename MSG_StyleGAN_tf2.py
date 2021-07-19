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


#---------------------------define mapping
def make_mapping_model():

    # Options
    use_pixel_norm    = True      # Enable pixelwise feature vector normalization
    use_wscale        = True      # Enable equalized learning rate
    truncation_psi    = 0.7,      # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff = 8,        # Number of layers for which to apply the truncation trick. None = disable.
    dlatent_avg_beta  = 0.995,    # Decay for tracking the moving average of W during training. None = disable.


    # Setup variables.
    v_init = tf.zeros_initializer()
    dlatent_avg = tf.Variable(
        initial_value=v_init(shape=[LATENT_SIZE], dtype="float32"),
        trainable=False,
        name="dlatent_avg",
    )


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

    dlatents = tf.tile(latents[:, np.newaxis], [1, G_LAYERS, 1])


    # Update moving average of W.
    if dlatent_avg_beta is not None:
        batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
        update_op = dlatent_avg, batch_avg - (dlatent_avg - batch_avg)*dlatent_avg_beta
        with tf.control_dependencies([update_op]):
            dlatents = tf.identity(dlatents)

    if truncation_psi is not None and truncation_cutoff is not None:
        layer_idx = np.arange(G_LAYERS)[np.newaxis, :, np.newaxis]
        ones = np.ones(layer_idx.shape, dtype=np.float32)
        coefs = tf.cast(tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones), tf.float32)
        dlatents = dlatent_avg + (dlatents - dlatent_avg) * coefs


    mapping_model     = Model(inputs=latents_in, outputs=dlatents)
    mapping_model_ave = Model(inputs=latents_in, outputs=dlatents)

    return mapping_model, mapping_model_ave


#---------------------------define synthesis
def make_synthesis_model():

    # Options
    use_pixel_norm    = False        # Disable pixelwise feature vector normalization
    use_wscale        = True         # Enable equalized learning rate
    use_instance_norm = True         # Enable instance normalization
    use_noise         = True         # Enable noise inputs
    randomize_noise   = False         # True = randomize noise inputs every time (non-deterministic),
                                     # False = read noise inputs from variables.
    use_styles        = True         # Enable style inputs                             
    blur_filter       = BLUR_FILTER  # Low-pass filter to apply when resampling activations. 
                                     # None = no filtering.
    fused_scale="auto"               # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.


    # Inputs
    dlatents = tf.keras.Input(shape=([G_LAYERS, LATENT_SIZE]), dtype=DTYPE)


    # Noise inputs
    noise_inputs = []
    for ldx in range(G_LAYERS):
        res = ldx // 2 + 2
        shape = [1, 2**res, 2**res]
        lnoise = layer_noise(dlatents, shape, name="input_noise%d" % ldx)
        noise = lnoise(dlatents)
        noise_inputs.append(noise)


    # Things to do at the end of each layer.
    def layer_epilogue(in_x, ldx):
        if use_noise:
            if noise_inputs[ldx] is None or randomize_noise:
                rnoise = tf.random.normal([tf.shape(in_x)[0], 1, in_x.shape[2], in_x.shape[3]],
                                          dtype=in_x.dtype,
                                          name="random_noise%d" % ldx)
            else:
                rnoise = tf.cast(noise_inputs[ldx], in_x.dtype)

            noise = apply_noise(in_x,
                                noise_inputs[ldx],
                                randomize_noise=randomize_noise,
                                name="noise%d" % ldx)
            in_x = noise(in_x, rnoise)

        bias = layer_bias(in_x, name ="Noise_bias%d" % ldx)
        in_x = bias(in_x)
        in_x = layers.LeakyReLU(name ="Noise_ReLU%d" % ldx)(in_x)
        if use_pixel_norm:
            in_x = pixel_norm(in_x)
        if use_instance_norm:
            in_x = instance_norm(in_x)
        if use_styles:
            in_x = style_mod(in_x, 
                             dlatents[:, ldx],
                             use_wscale=use_wscale,
                             name = "style_noise_%d" % ldx)
        return in_x


    # define blur
    def blur(in_x):
        return blur2d(in_x, blur_filter) if blur_filter else in_x


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
    def torgb(in_res, in_x):  # res = 2..RES_LOG2
        in_lod = RES_LOG2 - in_res
        in_x = conv2d(in_x, fmaps=NUM_CHANNELS, kernel=1, gain=1, use_wscale=use_wscale, name ="ToRGB_lod%d" % in_lod)
        bias = layer_bias(in_x, name ="ToRGB_bias_lod%d" % in_lod)
        in_x = bias(in_x)
        return in_x


    # Finally, arrange the computations for the layers
    images_out = []  # list will contain the output images at different resolutions
    images_out.append(torgb(2, x))
    for res in range(3, RES_LOG2 + 1):
        x = block(res, x)
        images_out.append(torgb(res, x))

    synthesis_model     = Model(inputs=dlatents, outputs=images_out)
    synthesis_model_ave = Model(inputs=dlatents, outputs=images_out)

    return synthesis_model, synthesis_model_ave




#-------------------------------------define discriminator
def make_discriminator_model():

    use_wscale         = True # Enable equalized learning rate
    label_size         = 0    # Dimensionality of the labels, 0 if no labels. Overridden based on dataset
    mbstd_group_size   = 4    # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features = 1    # Number of features for the minibatch standard deviation layer.
    blur_filter = BLUR_FILTER # Low-pass filter to apply when resampling activations. 
                              # None = no filtering.
    fused_scale="auto"        # True = fused convolution + scaling, 
                              # False = separate ops, 'auto' = decide automatically.

    def blur(in_x):
        return blur2d(in_x, blur_filter) if blur_filter else in_x

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
generator_optimizer     = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=DIS_LR, beta_1=0.0, beta_2=0.99, epsilon=1e-8)


#-------------------------------------create an instance of the generator and discriminator
mapping, mapping_ave     = make_mapping_model()
synthesis, synthesis_ave = make_synthesis_model()
discriminator            = make_discriminator_model()


#-------------------------------------define checkpoint
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                mapping=mapping,
                                mapping_ave=mapping_ave,
                                synthesis=synthesis,
                                synthesis_ave=synthesis_ave,
                                discriminator=discriminator)


def gradient_penalty(x, x_gen):
    with tf.GradientTape() as t:
        t.watch(x)
        d_hat = discriminator(x, training=False)
        d_hat = d_hat * SCALING_UP   #loss scaling. Important for mixed precision training
    gradients = t.gradient(d_hat, x)
    return tf.reduce_mean([tf.reduce_sum(tf.square(grad*SCALING_DOWN), axis=[1, 2, 3]) for grad in gradients])



#----------------------------------------------extra pieces----------------------------------------------

#---debug the NN with a given value
# y = tf.constant(2.0, shape=[1, 1, 1, 1], name="const2")
# yy = tf.tile(y, [tf.shape(dlatents)[0], 512, 4, 4])


#---take one image only
# image_in = Image.open(image_in)
# image_in = np.array(image_in)
# image_in = tf.image.convert_image_dtype(image_in, tf.float32)
# image = []
# for res in range(2, RES_LOG2 + 1):
#     r_img = tf.image.resize(image_in, [2**res, 2**res])
#     r_img = tf.transpose(r_img)
#     if (res==RES_LOG2):
#         image_ref = r_img
#     r_img = tf.expand_dims(r_img, axis=0).shape.as_list()
#     image.append(r_img)


#---for debug only
#for image in labeled_ds.take(10):
#  print("Image shape: ", image.numpy().shape)


#---mixed gradient penaly
# dim = x[0].shape[0]
# epsilon = tf.random.uniform([dim, 1, 1, 1], 0.0, 1.0)
# x_hat = []
# for i in range ( RES_LOG2-1):
#     x_hat.append(epsilon * x[i] + (1 - epsilon) * x_gen[i])


#---open a file
# image_in = Image.open('./data/defects_clean/defects5_clean.jpg')
# image_in = np.array(image_in)
# image_in = tf.image.convert_image_dtype(image_in, tf.float32)


#---set trainable variables
# list_traiv = generator.trainable_variables


# def update_average(G, Gs, beta):
#     list = G.trainable_variables
#     for var in list:
#         if 'dlatent_dense' or 'dlatent_bias' in var.name:
#             var = var * beta + (1-beta) * up_weight[j])
#             var.set_weights(new_weight)        
