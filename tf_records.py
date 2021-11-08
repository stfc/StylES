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
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from parameters import *

image_paths = glob.glob(DATASET + '*.jpg')



def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def decode_img(img):
    #convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    if (NUM_CHANNELS==1):
        img = tf.image.rgb_to_grayscale(img)
    
    #Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, DTYPE)
    
    #resize the image to the desired size.
    img_out = []
    for reslog in range(2, RES_LOG2 + 1):
        r_img = tf.image.resize(img, [2**reslog, 2**reslog])
        r_img = tf.transpose(r_img)
        img_out.append(r_img)
        
    return img_out



# Define functions for serialization
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Serialize records
def serialize_record(images):
    features = {
                   'image_4':   _bytes_feature(images[0]),
                   'image_8':   _bytes_feature(images[1]),
                   'image_16':  _bytes_feature(images[2]),
                   'image_32':  _bytes_feature(images[3]),
                   'image_64':  _bytes_feature(images[4]),
                   'image_128': _bytes_feature(images[5]),
                   'image_256': _bytes_feature(images[6]),
               }

    #  Create a Features message using tf.train.Example.
    proto = tf.train.Example(features=tf.train.Features(feature=features))
    
    return proto.SerializeToString()



# Write records
def write_tf_records():
    with tf.io.TFRecordWriter(TF_REC_DATASET) as writer:
        for image_path in image_paths:
            
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            if (NUM_CHANNELS==1):
                img = tf.image.rgb_to_grayscale(img)
            img = tf.image.convert_image_dtype(img, DTYPE)
            img_out = []
            for reslog in range(2, RES_LOG2 + 1):
                r_img = tf.image.resize(img, [2**reslog, 2**reslog])
                r_img = tf.transpose(r_img)
                r_img = tf.io.serialize_tensor(r_img)
                img_out.append(r_img)

            record = serialize_record(img_out)
            writer.write(record)

    print ('\n Tensorflow ' + TF_REC_DATASET + ' created!')


# Read records
def read_tfrecord(data_record):
    feature_description = {
        'image_4':    tf.io.FixedLenFeature((), tf.string),
        'image_8':    tf.io.FixedLenFeature((), tf.string),
        'image_16':   tf.io.FixedLenFeature((), tf.string),
        'image_32':   tf.io.FixedLenFeature((), tf.string),
        'image_64':   tf.io.FixedLenFeature((), tf.string),
        'image_128':  tf.io.FixedLenFeature((), tf.string),
        'image_256':  tf.io.FixedLenFeature((), tf.string),
    }

    record = tf.io.parse_single_example(data_record, feature_description)
    
    images_out = []
    for reslog in range(2, RES_LOG2 + 1):
        dim = 2**reslog
        rec_name = 'image_' + str(dim)
        image = tf.io.parse_tensor(record[rec_name],  out_type = DTYPE)
        image = tf.reshape(image, [3, dim, dim])
        images_out.append(image)

    return images_out




#---------------- extra pieces

# plt.figure(figsize=(10,10))
# for i, data in enumerate(parsed_dataset.take(9)):
#     print (data)
#     img = tf.keras.preprocessing.image.array_to_img(data[0])
#     plt.subplot(3,3,i+1)
#     plt.imshow(img)
# plt.show()  