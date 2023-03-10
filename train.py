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

from tensorflow.keras import layers, initializers, regularizers
from tensorflow.keras.models import Model

from parameters import *
from functions import *
from MSG_StyleGAN_tf2 import *
from IO_functions import *

iNN = 1.0/(OUTPUT_DIM*OUTPUT_DIM)


def compute_losses(real_output_per_sample, fake_output_per_sample, f_images, g_images, images):

    # discriminator loss
    loss_real_per_sample  = tf.math.softplus(-real_output_per_sample)

    loss_fake_per_sample  = tf.math.softplus(fake_output_per_sample)
    r1_penalty_per_sample = gradient_penalty(images)

    # generator loss
    loss_gen_per_sample = tf.math.softplus(-fake_output_per_sample)

    # filter loss
    loss_fil_per_sample = []
    for fil in range(RES_LOG2-RES_LOG2_FIL):
        loss_fil_per_sample.append(tf.math.squared_difference(f_images[fil], g_images[RES_LOG2 -(fil+1) -2]))

    loss_real   = tf.math.reduce_mean(loss_real_per_sample)
    loss_fake   = tf.math.reduce_mean(loss_fake_per_sample)
    loss_gen    = tf.math.reduce_mean(loss_gen_per_sample)

    loss_fil = []
    for fil in range(RES_LOG2-RES_LOG2_FIL):
        loss_fil.append(tf.math.reduce_mean(loss_fil_per_sample[fil]))

    r1_penalty  = tf.math.reduce_mean(r1_penalty_per_sample)
    real_output = tf.math.reduce_mean(real_output_per_sample)
    fake_output = tf.math.reduce_mean(fake_output_per_sample)

    return loss_real, loss_fake, loss_gen, loss_fil, r1_penalty, real_output, fake_output

 


#-------------------------------------define training step and loop
@tf.function
def train_step(input, images):
    with tf.GradientTape() as map_tape, \
        tf.GradientTape() as syn_tape, \
        tf.GradientTape() as fil_tape_128, \
        tf.GradientTape() as fil_tape_64, \
        tf.GradientTape() as fil_tape_32, \
        tf.GradientTape() as fil_tape_16, \
        tf.GradientTape() as disc_tape:
        dlatents = mapping(input, training = True)
        g_images = synthesis(dlatents, training = True)

        f_images = []
        for fil in range(RES_LOG2-RES_LOG2_FIL):
            f_images.append(filter[fil](g_images[RES_LOG2-2], training = True))

        real_output = discriminator(images,   training=True)
        fake_output = discriminator(g_images, training=True)

        # find losses
        loss_real, loss_fake, loss_gen, loss_fil, r1_penalty, real_output, fake_output = compute_losses(real_output, fake_output, f_images, g_images, images)

        # find loss discriminator
        loss_disc  = loss_real + loss_fake + r1_penalty * (R1_GAMMA * 0.5)  #10.0 is the gradient penalty weight

    #apply gradients
    gradients_of_mapping       = map_tape.gradient(loss_gen,   mapping.trainable_variables)
    gradients_of_synthetis     = syn_tape.gradient(loss_gen,   synthesis.trainable_variables)

    gradients_of_filter = []
    gradients_of_filter.append(fil_tape_128.gradient(loss_fil[0], filter[0].trainable_variables))
    gradients_of_filter.append(fil_tape_64.gradient(loss_fil[1],  filter[1].trainable_variables))
    gradients_of_filter.append(fil_tape_32.gradient(loss_fil[2],  filter[2].trainable_variables))
    gradients_of_filter.append(fil_tape_16.gradient(loss_fil[3],  filter[3].trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(loss_disc, discriminator.trainable_variables)

    gradients_of_mapping       = [g if g is not None else tf.zeros_like(g) for g in gradients_of_mapping ]
    gradients_of_synthetis     = [g if g is not None else tf.zeros_like(g) for g in gradients_of_synthetis ]

    for fil in range(RES_LOG2-RES_LOG2_FIL):
        gradients_of_filter[fil] = [g if g is not None else tf.zeros_like(g) for g in gradients_of_filter[fil] ]

    gradients_of_discriminator = [g if g is not None else tf.zeros_like(g) for g in gradients_of_discriminator ]

    generator_optimizer.apply_gradients(zip(gradients_of_mapping,           mapping.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_synthetis,         synthesis.trainable_variables))

    for fil in range(RES_LOG2-RES_LOG2_FIL):
        filter_optimizer.apply_gradients(zip(gradients_of_filter[fil],  filter[fil].trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    metrics = [loss_disc, loss_gen, loss_fil, r1_penalty, real_output, fake_output]

    return metrics



def train(dataset, train_summary_writer):

    # Load latest checkpoint, if restarting
    managerCheckpoint = tf.train.CheckpointManager(checkpoint, CHKP_DIR, max_to_keep=None)
    if (IRESTART):
        checkpoint.restore(managerCheckpoint.latest_checkpoint)


    # Create noise for sample images
    tf.random.set_seed(1)
    input_latent = tf.random.uniform([BATCH_SIZE, LATENT_SIZE], dtype=DTYPE, minval=MINVALRAN, maxval=MAXVALRAN)
    mtr = np.zeros([5], dtype=DTYPE)


    #save first images
    div, momU, momV = generate_and_save_images(mapping, synthesis, input_latent, 0)
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
        mtr = train_step(input_batch, image_batch)

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
                mtr[2][RES_LOG2-RES_LOG2_FIL-1], \
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
                for fil in range(RES_LOG2-RES_LOG2_FIL):
                    res = 2**(RES_LOG2_FIL+fil)
                    tf.summary.scalar('a/loss_filter ' +str(res), mtr[2][fil], step=it)
                tf.summary.scalar('a/r1_penalty',  mtr[3], step=it)
                tf.summary.scalar('a/score_real',  mtr[4], step=it)
                tf.summary.scalar('a/score_fake',  mtr[5], step=it)
                tf.summary.scalar('a/lr_gen',      lr_gen, step=it)
                tf.summary.scalar('a/lr_dis',      lr_dis, step=it)                


        # print images
        if (it+1) % IMAGES_EVERY == 0:
            div, momU, momV = generate_and_save_images(mapping, synthesis, input_latent, it+1)

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
            managerCheckpoint.save()


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

