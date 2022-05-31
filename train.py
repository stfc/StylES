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

iNN = 1.0/(OUTPUT_DIM*OUTPUT_DIM)


with mirrored_strategy.scope():
    def compute_losses(real_output_per_sample, fake_output_per_sample, f_images, g_images, images):

        # discriminator loss
        loss_real_per_sample  = tf.math.softplus(-real_output_per_sample)

        loss_fake_per_sample  = tf.math.softplus(fake_output_per_sample)
        r1_penalty_per_sample = gradient_penalty(images)

        # generator loss
        loss_gen_per_sample = tf.math.softplus(-fake_output_per_sample)

        # filter loss
        loss_fil_per_sample = tf.math.squared_difference(f_images, g_images[RES_LOG2_FIL-2])

        loss_real   = tf.nn.compute_average_loss(loss_real_per_sample,   global_batch_size=GLOBAL_BATCH_SIZE)
        loss_fake   = tf.nn.compute_average_loss(loss_fake_per_sample,   global_batch_size=GLOBAL_BATCH_SIZE)
        loss_gen    = tf.nn.compute_average_loss(loss_gen_per_sample,    global_batch_size=GLOBAL_BATCH_SIZE)
        loss_fil    = tf.nn.compute_average_loss(loss_fil_per_sample,    global_batch_size=GLOBAL_BATCH_SIZE)
        r1_penalty  = tf.nn.compute_average_loss(r1_penalty_per_sample,  global_batch_size=GLOBAL_BATCH_SIZE)
        real_output = tf.nn.compute_average_loss(real_output_per_sample, global_batch_size=GLOBAL_BATCH_SIZE)
        fake_output = tf.nn.compute_average_loss(fake_output_per_sample, global_batch_size=GLOBAL_BATCH_SIZE)

        return loss_real, loss_fake, loss_gen, loss_fil, r1_penalty, real_output, fake_output

 


#-------------------------------------define training step and loop
@tf.function
def train_step(input, inputVariances, images):
    with tf.GradientTape() as map_tape, \
        tf.GradientTape() as syn_tape, \
        tf.GradientTape() as fil_tape, \
        tf.GradientTape() as disc_tape:
        dlatents = mapping(input, training = True)
        g_images = synthesis([dlatents, inputVariances], training = True)
        f_images = filter(g_images[RES_LOG2-2], training = True)

        real_output = discriminator(images,   training=True)
        fake_output = discriminator(g_images, training=True)

        # find losses
        loss_real, loss_fake, loss_gen, loss_fil, r1_penalty, real_output, fake_output = compute_losses(real_output, fake_output, f_images, g_images, images)

        # find loss discriminator
        loss_disc  = loss_real + loss_fake + r1_penalty * (R1_GAMMA * 0.5)  #10.0 is the gradient penalty weight

    #apply gradients
    gradients_of_mapping       = map_tape.gradient(loss_gen,   mapping.trainable_variables)
    gradients_of_synthetis     = syn_tape.gradient(loss_gen,   synthesis.trainable_variables)
    gradients_of_filter        = fil_tape.gradient(loss_fil,   filter.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(loss_disc, discriminator.trainable_variables)

    gradients_of_mapping       = [g if g is not None else tf.zeros_like(g) for g in gradients_of_mapping ]
    gradients_of_synthetis     = [g if g is not None else tf.zeros_like(g) for g in gradients_of_synthetis ]
    gradients_of_filter        = [g if g is not None else tf.zeros_like(g) for g in gradients_of_filter ]
    gradients_of_discriminator = [g if g is not None else tf.zeros_like(g) for g in gradients_of_discriminator ]

    generator_optimizer.apply_gradients(zip(gradients_of_mapping,           mapping.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_synthetis,         synthesis.trainable_variables))
    filter_optimizer.apply_gradients(zip(gradients_of_filter,               filter.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    metrics = [loss_disc, loss_gen, loss_fil, r1_penalty, real_output, fake_output]

    return metrics


@tf.function
def distributed_train_step(input, inputVariances, images):
    losses = mirrored_strategy.run(train_step, args=(input, inputVariances, images))
    mtr=[]
    for i in range(len(losses)):
        mtr.append(mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses[i], axis=None))

    return mtr




def train(dataset, train_summary_writer):

    # Load latest checkpoint, if restarting
    if (IRESTART):
        with mirrored_strategy.scope():
            checkpoint.restore(tf.train.latest_checkpoint(CHKP_DIR))


    # Create noise for sample images
    tf.random.set_seed(1)
    input_latent = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
    inputVariances = tf.constant(1.0, shape=(1, G_LAYERS), dtype=DTYPE)
    mtr = np.zeros([5], dtype=DTYPE)


    #save first images
    div, momU, momV = generate_and_save_images(mapping, synthesis, input_latent, inputVariances, 0)
    with train_summary_writer.as_default():
        for res in range(RES_LOG2-1):
            pow = 2**(res+2)
            var_name = "b/divergence_" + str(pow) + "x" + str(pow)
            tf.summary.scalar(var_name, div[res], step=0)
            var_name = "c/dUdt_" + str(pow) + "x" + str(pow)
            tf.summary.scalar(var_name, momU[res], step=0)
            var_name = "d/dVdt_" + str(pow) + "x" + str(pow)
            tf.summary.scalar(var_name, momV[res], step=0)

    tstart = time.time()
    tint   = tstart
    for it in range(TOT_ITERATIONS):
    
        # take next batch
        input_batch = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
        image_batch = next(iter(dataset))
        mtr = distributed_train_step(input_batch, inputVariances, image_batch)

        # print losses
        if it % PRINT_EVERY == 0:
            tend = time.time()
            lr_gen = lr_schedule_gen(it)
            lr_dis = lr_schedule_dis(it)
            print ('Total time {0:3.2f} h, Iteration {1:8d}, Time Step {2:6.2f} s, ' \
                'ld {3:6.2e}, ' \
                'lg {4:6.2e}, ' \
                'lf {5:6.2e}, ' \
                'r1 {6:6.2e}, ' \
                'sr {7:6.2e}, ' \
                'sf {8:6.2e}, ' \
                'lr_gen {9:6.2e}, ' \
                'lr_dis {10:6.2e}, ' \
                .format((tend-tstart)/3600, it, tend-tint, \
                mtr[0], \
                mtr[1], \
                mtr[2], \
                mtr[3], \
                mtr[4], \
                mtr[5], \
                lr_gen, \
                lr_dis))
            tint = tend

            # write losses to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('a/loss_disc',   mtr[0], step=it)
                tf.summary.scalar('a/loss_gen',    mtr[1], step=it)
                tf.summary.scalar('a/loss_filter', mtr[2], step=it)
                tf.summary.scalar('a/r1_penalty',  mtr[3], step=it)
                tf.summary.scalar('a/score_real',  mtr[4], step=it)
                tf.summary.scalar('a/score_fake',  mtr[5], step=it)                                
                tf.summary.scalar('a/lr_gen',      lr_gen, step=it)
                tf.summary.scalar('a/lr_dis',      lr_dis, step=it)                


        # print images
        if (it+1) % IMAGES_EVERY == 0:
            div, momU, momV = generate_and_save_images(mapping, synthesis, input_latent, inputVariances, it+1)

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




# Extra pieces.............

# update average weights

        # # update average weights
        # if (it >= 0 and it % G_SMOOTH_RATE == 0):
        #     with tf.device('/device:gpu:0'):
        #         for i in range(len(mapping.layers)):
        #             up_weights = mapping.layers[i].get_weights()
        #             old_weights = mapping_ave.layers[i].get_weights()
        #             new_weights = []
        #             for j in range(len(up_weights)):
        #                 new_weights.append(old_weights[j] * Gs_beta + (1-Gs_beta) * up_weights[j])
        #             if (len(new_weights)>0):
        #                 mapping_ave.layers[i].set_weights(new_weights)

        #         for i in range(len(synthesis.layers)):
        #             up_weights = synthesis.layers[i].get_weights()
        #             old_weights = synthesis_ave.layers[i].get_weights()
        #             new_weights = []
        #             for j in range(len(up_weights)):
        #                 new_weights.append(old_weights[j] * Gs_beta + (1-Gs_beta) * up_weights[j])
        #             if (len(new_weights)>0):
        #                 synthesis_ave.layers[i].set_weights(new_weights)


                # # find divergence loss
        # loss_div_per_sample=0.
        # for res in range(2,RES_LOG2-1):
        #     U  = g_images[res][:,0,:,:]*2*uRef - uRef
        #     V  = g_images[res][:,1,:,:]*2*uRef - uRef
        #     loss_div_per_sample = loss_div_per_sample + tf.math.reduce_sum(tf.abs(((tr(U, 1, 0)-tr(U, -1, 0)) + (tr(V, 0, 1)-tr(V, 0, -1)))))
        # loss_div_per_sample = loss_div_per_sample*iNN

        # # find vorticity loss
        # Wt = ((tr(V, 1, 0)-tr(V, -1, 0)) - (tr(U, 0, 1)-tr(U, 0, -1)))
        # Wt = (Wt - tf.math.reduce_min(Wt))/(tf.math.reduce_max(Wt) - tf.math.reduce_min(Wt) + 1.e-20)
        # W  = g_images[RES_LOG2-2][:,2,:,:]  # we want the difference between W inferred and W calculated
        # W  = (W - tf.math.reduce_min(W))/(tf.math.reduce_max(W) - tf.math.reduce_min(W) + 1.e-20)
        # loss_vor_per_sample = tf.reduce_mean(tf.math.squared_difference(W, Wt))*iNN

        # total loss for the generator
        #loss_gen_per_sample = loss_gen_per_sample + loss_div_per_sample + loss_vor_per_sample

