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
from MSG_StyleGAN_tf2 import *
from tensorflow.keras import layers, initializers, regularizers
from tensorflow.keras.models import Model




#-------------------------------------define training step and loop
@tf.function
def train_step(input, images):
    with tf.GradientTape() as map_tape, tf.GradientTape() as syn_tape, tf.GradientTape() as disc_tape:
        dlatents = mapping_ave(input, training = True)
        g_images = synthesis_ave(dlatents, training = True)

        real_output = discriminator(images,   training=True)
        fake_output = discriminator(g_images, training=True)

        # find discriminator loss
        loss_real  = tf.reduce_mean(tf.math.softplus(-real_output))
        loss_fake  = tf.reduce_mean(tf.math.softplus(fake_output))
        r1_penalty = gradient_penalty(images)
        loss_disc  = loss_real + loss_fake + r1_penalty * (R1_GAMMA * 0.5)  #10.0 is the gradient penalty weight

        # find generator loss
        loss_gen  = tf.reduce_mean(tf.math.softplus(-fake_output))


    #apply gradients
    gradients_of_mapping       = map_tape.gradient(loss_gen,   mapping_ave.trainable_variables)
    gradients_of_synthetis     = syn_tape.gradient(loss_gen,   synthesis_ave.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(loss_disc, discriminator.trainable_variables)

    gradients_of_mapping       = [g if g is not None else tf.zeros_like(g) for g in gradients_of_mapping ]
    gradients_of_synthetis     = [g if g is not None else tf.zeros_like(g) for g in gradients_of_synthetis ]
    gradients_of_discriminator = [g if g is not None else tf.zeros_like(g) for g in gradients_of_discriminator ]

    generator_optimizer.apply_gradients(zip(gradients_of_mapping,           mapping.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_synthetis,         synthesis.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    for i in range(len(mapping.layers)):
        up_weight = mapping.layers[i].weights
        old_weight = mapping_ave.layers[i].weights
        for j in range(len(up_weight)):
            new_weight = old_weight[j] * Gs_beta + (1-Gs_beta) * up_weight[j]
            mapping_ave.layers[i].weights[j] = new_weight

    for i in range(len(synthesis.layers)):
        up_weight = synthesis.layers[i].weights
        old_weight = synthesis_ave.layers[i].weights
        for j in range(len(up_weight)):
            new_weight = old_weight[j] * Gs_beta + (1-Gs_beta) * up_weight[j]
            synthesis_ave.layers[i].weights[j] = new_weight

    metrics = [loss_disc, loss_gen, r1_penalty]

    return metrics



def train(dataset, GEN_LR, DIS_LR, train_summary_writer):

    # Plot models
    #mapping.summary()
    #synthesis.summary()
    #discriminator.summary()
    #tf.keras.utils.plot_model(generator,     "generator.png",     show_shapes=True)
    #tf.keras.utils.plot_model(discriminator, "discriminator.png", show_shapes=True)


    # Load latest checkpoint, if restarting
    if (IRESTART):
        checkpoint.restore(tf.train.latest_checkpoint(CHKP_DIR))


    # Create noise for sample images
    tf.random.set_seed(1)
    input_latent = tf.random.uniform([BATCH_SIZE, LATENT_SIZE])


    #save first images
    generate_and_save_images(mapping_ave, synthesis_ave, input_latent, 0)

    tstart = time.time()
    tint   = tstart
    for it in range(TOT_ITERATIONS):
        input_batch = tf.random.uniform([BATCH_SIZE, LATENT_SIZE])
        image_batch = next(iter(dataset))
        mtr = train_step(input_batch, image_batch)

        #print losses
        if it % PRINT_EVERY == 0:
            tend = time.time()
            print ('Total time {0:3.1f} h, Iteration {1:8d}, Time Step {2:6.1f} s, ' \
                'loss_disc {3:6.1e}, '  \
                'loss_gen {4:6.1e}, '   \
                'r1_penalty {5:6.1e}, ' \
                .format((tend-tstart)/3600, it, tend-tint, \
                mtr[0], \
                mtr[1], \
                mtr[2]))
            tint = tend

            # write losses tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss_disc',  mtr[0], step=it)
                tf.summary.scalar('loss_gen',   mtr[1], step=it)
                tf.summary.scalar('r1_penalty', mtr[2], step=it)

        #print images
        if (it+1) % IMAGES_EVERY == 0:    
            generate_and_save_images(mapping_ave, synthesis_ave, input_batch, it+1)

        #save the model
        if (it+1) % SAVE_EVERY == 0:    
            checkpoint.save(file_prefix = CHKP_PREFIX)

        #reduce learning rate
        if (it+1) % REDUCE_EVERY == 0:    
            # reduce learning rate
            if (GEN_LR>LR_THRS):
                GEN_LR = GEN_LR*1.0e-1
                DIS_LR = GEN_LR


    if (PROFILE):
        tf.summary.trace_export(name="Train", step=it,profiler_outdir='./logs_profile/train')

