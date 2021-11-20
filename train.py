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

from tensorflow.keras import layers, initializers, regularizers
from tensorflow.keras.models import Model

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from IO_functions import *
from LES_Solvers.testcases.HIT_2D.HIT_2D import uRef


#-------------------------------------define training step and loop
@tf.function
def train_step(input, images):
    with tf.GradientTape() as map_tape, \
         tf.GradientTape() as syn_tape, \
         tf.GradientTape() as fil_tape, \
         tf.GradientTape() as disc_tape:
        dlatents = mapping(input, training = True)
        g_images = synthesis(dlatents, training = True)
        f_images = filter(g_images[RES_LOG2-2], training = True)

        # # find vorticity loss
        # g_images_new = []
        # for res in range(RES_LOG2-1):
        #     U  = g_images[res][:,0,:,:]
        #     V  = g_images[res][:,1,:,:]
        #     Wt = ((tr(V, 1, 0)-tr(V, -1, 0)) - (tr(U, 0, 1)-tr(U, 0, -1)))
        #     WtMax = tf.math.reduce_max(Wt) 
        #     WtMin = tf.math.reduce_min(Wt) 
        #     Wt = (Wt - WtMin)/(WtMax - WtMin + 1.e-20)
        #     g_images_new.append(tf.concat([g_images[res][:,0:2,:,:], Wt[:,np.newaxis,:,:]], 1))

        real_output = discriminator(images,   training=True)
        fake_output = discriminator(g_images, training=True)

        # find discriminator loss
        loss_real  = tf.reduce_mean(tf.math.softplus(-real_output))
        loss_fake  = tf.reduce_mean(tf.math.softplus(fake_output))
        r1_penalty = gradient_penalty(images)
        loss_disc  = loss_real + loss_fake + r1_penalty * (R1_GAMMA * 0.5)  #10.0 is the gradient penalty weight

        # find generator loss
        loss_gen  = tf.reduce_mean(tf.math.softplus(-fake_output))

        # find filter loss
        loss_fil = tf.reduce_mean(tf.math.squared_difference(f_images, g_images[RES_LOG2-5]))

        # # find vorticity loss
        # U  = g_images[RES_LOG2-2][:,0,:,:]*uRef
        # V  = g_images[RES_LOG2-2][:,1,:,:]*uRef
        # Wt =  ((tr(V, 1, 0)-tr(V, -1, 0)) - (tr(U, 0, 1)-tr(U, 0, -1)))
        # Wt  = (Wt - tf.math.reduce_min(Wt))/(tf.math.reduce_max(Wt) - tf.math.reduce_min(Wt) + 1.e-20)
        # W  = g_images[RES_LOG2-2][:,2,:,:]  # we want the difference between W inferred and W calculated
        # W  = (W - tf.math.reduce_min(W))/(tf.math.reduce_max(W) - tf.math.reduce_min(W) + 1.e-20)

        loss_vor = 0. #tf.reduce_mean(tf.math.squared_difference(W, Wt))
        loss_gen_vor = loss_gen + loss_vor

    #apply gradients
    gradients_of_mapping       = map_tape.gradient(loss_gen, mapping.trainable_variables)
    gradients_of_synthetis     = syn_tape.gradient(loss_gen, synthesis.trainable_variables)
    gradients_of_filter        = fil_tape.gradient(loss_fil,          filter.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(loss_disc,        discriminator.trainable_variables)

    gradients_of_mapping       = [g if g is not None else tf.zeros_like(g) for g in gradients_of_mapping ]
    gradients_of_synthetis     = [g if g is not None else tf.zeros_like(g) for g in gradients_of_synthetis ]
    gradients_of_filter        = [g if g is not None else tf.zeros_like(g) for g in gradients_of_filter ]
    gradients_of_discriminator = [g if g is not None else tf.zeros_like(g) for g in gradients_of_discriminator ]

    generator_optimizer.apply_gradients(zip(gradients_of_mapping,           mapping.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_synthetis,         synthesis.trainable_variables))
    filter_optimizer.apply_gradients(zip(gradients_of_filter,               filter.trainable_variables))
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

    metrics = [loss_disc, loss_gen, loss_vor, loss_fil, r1_penalty, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)]

    return metrics



def train(dataset, LR, train_summary_writer):

    # Load latest checkpoint, if restarting
    if (IRESTART):
        checkpoint.restore(tf.train.latest_checkpoint(CHKP_DIR))


    # Create noise for sample images
    tf.random.set_seed(1)
    input_latent = tf.random.uniform([BATCH_SIZE, LATENT_SIZE])
    lr = LR
    mtr = np.zeros([5], dtype=DTYPE)


    #save first images
    div, momU, momV = generate_and_save_images(mapping_ave, synthesis_ave, input_latent, 0)
    with train_summary_writer.as_default():
        for res in range(RES_LOG2-1):
            pow = 2**(res+2)
            var_name = "divergence/" + str(pow) + "x" + str(pow)
            tf.summary.scalar(var_name, div[res], step=0)
            var_name = "dUdt/" + str(pow) + "x" + str(pow)
            tf.summary.scalar(var_name, momU[res], step=0)
            var_name = "dVdt/" + str(pow) + "x" + str(pow)
            tf.summary.scalar(var_name, momV[res], step=0)

    tstart = time.time()
    tint   = tstart
    for it in range(TOT_ITERATIONS):
    
        # take next batch
        input_batch = tf.random.uniform([BATCH_SIZE, LATENT_SIZE])
        image_batch = next(iter(dataset))
        mtr = train_step(input_batch, image_batch)

        # print losses
        if it % PRINT_EVERY == 0:
            tend = time.time()
            lr = lr_schedule(it)
            print ('Total time {0:3.1f} h, Iteration {1:8d}, Time Step {2:6.1f} s, ' \
                'ld {3:6.1e}, ' \
                'lg {4:6.1e}, ' \
                'lv {5:6.1e}, ' \
                'lf {6:6.1e}, ' \
                'r1 {7:6.1e}, ' \
                'sr {8:6.1e}, ' \
                'sf {9:6.1e}, ' \
                'lr {10:6.1e}, ' \
                .format((tend-tstart)/3600, it, tend-tint, \
                mtr[0], \
                mtr[1], \
                mtr[2], \
                mtr[3], \
                mtr[4], \
                mtr[5], \
                mtr[6], \
                lr))
            tint = tend

            # write losses to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('a/loss_disc',   mtr[0], step=it)
                tf.summary.scalar('a/loss_gen',    mtr[1], step=it)
                tf.summary.scalar('a/loss_vor',    mtr[2], step=it)
                tf.summary.scalar('a/loss_filter', mtr[3], step=it)
                tf.summary.scalar('a/r1_penalty',  mtr[4], step=it)
                tf.summary.scalar('a/score_real',  mtr[5], step=it)
                tf.summary.scalar('a/score_fake',  mtr[6], step=it)                                
                tf.summary.scalar('a/lr',              lr, step=it)


        # print images
        if (it+1) % IMAGES_EVERY == 0:    
            div, momU, momV = generate_and_save_images(mapping_ave, synthesis_ave, input_batch, it+1)
            with train_summary_writer.as_default():
                for res in range(RES_LOG2-1):
                    pow = 2**(res+2)
                    var_name = "b/divergence_" + str(pow) + "x" + str(pow)
                    tf.summary.scalar(var_name, div[res], step=it)
                    var_name = "c/dUdt_" + str(pow) + "x" + str(pow)
                    tf.summary.scalar(var_name, momU[res], step=it)
                    var_name = "d/dVdt_" + str(pow) + "x" + str(pow)
                    tf.summary.scalar(var_name, momV[res], step=it)

        #save the model
        if (it+1) % SAVE_EVERY == 0:    
            checkpoint.save(file_prefix = CHKP_PREFIX)


    # end of the training
    print("Total divergencies, dUdt and dVdt for each resolution are:")
    for reslog in range(RES_LOG2-1):
        res = 2**(reslog+2)
        print("{:4d}x{:4d}:   {:03e}   {:03e}   {:03e}".format(res, res, div[reslog], momU[reslog], momV[reslog]))
    print("\n")


    # profile
    if (PROFILE):
        tf.summary.trace_export(name="Train", step=it,profiler_outdir='./logs_profile/train')

