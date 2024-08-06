import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model


print("TensorFlow version:", tf.__version__)


tf.config.run_functions_eagerly(True)

RAND_NOISE = True

class layer_dummy(layers.Layer):
    def __init__(self, x, **kwargs):
        super(layer_dummy, self).__init__(**kwargs)

        c_init = tf.ones_initializer()
        self.c = tf.Variable(
            initial_value=c_init(shape=[x.shape[-2],x.shape[-1]]),
            trainable=False,
            name="dummy_var",
        )

    def call(self, vin, xshape):
        x = vin + tf.reshape(self.c, shape=xshape)
        return x



def make_dummy_model():
    vin = tf.keras.Input(shape=([1, 10]))

    ldummy = layer_dummy(vin)
    vout = ldummy(vin, [vin.shape[-2],vin.shape[-1]])

    dummy_model = Model(inputs=vin, outputs=vout)
    return dummy_model


dummy = make_dummy_model()
dummy.summary()

exit()











# #-------------------------- test: noise generation---------------------------- 
# DTYPE = "float32"
# LATENT_SIZE = 512
# N  = 100
# N2 = int(N/2)

# zl_init = tf.ones_initializer()
# c = tf.Variable(
#     initial_value=zl_init(shape=[1,N2], dtype="float32"),
#     trainable=True,
#     name="zlatent_k1"
# )

# # zl_init = tf.range(-250,250,dtype="float32")
# # z2 = zl_init #*zl_init
# # z2 = z2[tf.newaxis,:]
# # c = tf.Variable(z2,
# #     trainable=True,
# #     name="zlatent_k1"
# # )

# T  = LATENT_SIZE-1
# Dt = T/(LATENT_SIZE-1)
# t  = Dt*tf.range(LATENT_SIZE, dtype="float32")
# t  = t[tf.newaxis,:]
# t  = tf.tile(t, [N2, 1])
# k  = tf.range(1,int(N2+1), dtype=DTYPE)
# f  = k/T
# f  = f[:,tf.newaxis]

# phi = tf.random.uniform([N2,1], maxval=2.0*np.pi, dtype=DTYPE)
# #phi = tf.ones([N2,1])

# freq = f * t

# argsin = tf.math.sin(2*np.pi*freq + phi)
# x = tf.matmul(c,argsin)

# minRan = tf.math.reduce_min(x)
# x = x - minRan

# plt.plot(x[0,:].numpy())
# plt.savefig('dummy_test1.png')
# plt.close()

# plt.hist(x[0,:].numpy(), 50)
# plt.savefig('dummy_test2.png')





# DTYPE = "float32"
# LATENT_SIZE = 512
# NC_NOISE = 100
# NC2_NOISE = int(NC_NOISE/2)
# DTYPE    = "float32"

# class layer_noise(tf.keras.layers.Layer):
#     def __init__(self, x, ldx, **kwargs):
#         super(layer_noise, self).__init__(**kwargs)

#         self.NSIZE = x.shape[-2] * x.shape[-1]
#         self.N     = NC_NOISE
#         self.N2    = int(self.N/2)
#         self.T     = self.NSIZE-1
#         self.Dt    = self.T/(self.NSIZE-1)
#         self.t     = self.Dt*tf.range(self.NSIZE, dtype=DTYPE)
#         self.t     = self.t[tf.newaxis,:]
#         self.t     = tf.tile(self.t, [self.N2, 1])
#         self.k     = tf.range(1,int(self.N2+1), dtype=DTYPE)
#         self.f     = self.k/self.T
#         self.f     = self.f[:,tf.newaxis]

#         c_init = tf.ones_initializer()
#         self.c = tf.Variable(
#             initial_value=c_init(shape=[1,self.N2], dtype=DTYPE),
#             trainable=True,
#             name="noise_%d" % ldx
#         )

#     def call(self, x, phi):

#         freq = self.f * self.t
#         argsin = tf.math.sin(2*np.pi*freq + phi)
#         noise = tf.matmul(self.c,argsin)
#         noise = tf.reshape(noise, shape=x.shape)

#         return noise


# style_in = tf.ones(shape=[1, LATENT_SIZE], dtype=DTYPE)
# #phi      = tf.ones([NC2_NOISE,1], dtype=DTYPE)
# phi      = tf.random.uniform([NC2_NOISE,1], maxval=2.0*np.pi, dtype=DTYPE)
# lnoise   = layer_noise(style_in, 0)
# z        = lnoise(style_in, phi)

# filename = "z_latent.png"
# plt.plot(z.numpy()[0,:])
# plt.savefig(filename)
# plt.close()




# LATENT_SIZE = 512
# N   = 10
# N2  = int(N/2)
# T   = LATENT_SIZE-1
# Dt  = T/(LATENT_SIZE-1)
# t   = Dt*np.arange(1,LATENT_SIZE)
# k   = np.arange(1,int(N2))
# c   = np.ones(N2)
# phi = np.random.uniform(0,2*np.pi,N2)
# f   = k/T
# C0  = 0

# v = C0/2*np.ones(LATENT_SIZE)
# for i in range(LATENT_SIZE):
#     for k in range(1,N2):
#         v[i-1] = v[i-1] + c[k-1]*np.sin(2*np.pi*f[k-1]*t[i-1] + phi[k-1])

# plt.plot(v)
# plt.savefig('dummy_test1.png')






# for k in range(1):
#     filename = "results_reconstruction/wf_" + str(k) + ".npz"
#     data = np.load(filename)
#     wf = data['wf']
#     print(wf.shape)
#     for j in range(10,11):
#         plt.plot(wf[0,j,:])

#     plt.savefig('test1.png', linewidth=0.01)

# exit()

# a = tf.random.uniform([3,1],maxval=5, dtype="int32")
# b = tf.random.uniform([1,3,2],maxval=5, dtype="int32")
# c = a[:,:]*b[:,:,:]
# print(a)
# print(b)
# print(c)

# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model, layers


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # Add a channels dimension
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")


# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000).batch(32)

# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# #-------------Layer Noise
# def apply_noise(x):
#     w_init = tf.zeros_initializer()
#     weight = tf.Variable(
#     initial_value=w_init(shape=[1]),
#     trainable=True,
#     )
#     return x*weight


# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10)

#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     x = self.d2(x)
#     x = apply_noise(x)
#     return x



# # Create an instance of the model
# model = MyModel()

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# optimizer = tf.keras.optimizers.Adam()


# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# #@tf.function
# def train_step(images, labels):
#   with tf.GradientTape() as tape:
#     # training=True is only needed if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     predictions = model(images, training=True)
#     loss = loss_object(labels, predictions)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#   train_loss(loss)
#   train_accuracy(labels, predictions)



# #@tf.function
# def test_step(images, labels):
#   # training=False is only needed if there are layers with different
#   # behavior during training versus inference (e.g. Dropout).
#   predictions = model(images, training=False)
#   t_loss = loss_object(labels, predictions)

#   test_loss(t_loss)
#   test_accuracy(labels, predictions)




# EPOCHS = 1

# for epoch in range(EPOCHS):
#   # Reset the metrics at the start of the next epoch
#   train_loss.reset_states()
#   train_accuracy.reset_states()
#   test_loss.reset_states()
#   test_accuracy.reset_states()

#   for images, labels in train_ds:
#     train_step(images, labels)

#   for test_images, test_labels in test_ds:
#     test_step(test_images, test_labels)

#   print(
#     f'Epoch {epoch + 1}, '
#     f'Loss: {train_loss.result()}, '
#     f'Accuracy: {train_accuracy.result() * 100}, '
#     f'Test Loss: {test_loss.result()}, '
#     f'Test Accuracy: {test_accuracy.result() * 100}'
#   )

# model.summary()




  
# # x = tf.Variable(3.0)
# # c = tf.Variable(4.0)

# with tf.GradientTape() as tape:
#   x = c*5
#   y = x**2

# dx_dc = tape.gradient(x, c)

# print(dx_dc.numpy())





# layer = tf.keras.layers.Dense(2, activation='relu')
# c = tf.constant([[1., 2., 3.]])
# x = tf.Variable([[1., 2., 3.]], trainable=True)

# with tf.GradientTape() as tape:
#   # Forward pass
#   x = c*2
#   y = layer(x)
#   loss = tf.reduce_mean(y**2)

# # Calculate gradients with respect to every trainable variable
# grad = tape.gradient(loss, layer.trainable_variables)


# for var, g in zip(layer.trainable_variables, grad):
#   print(f'{var.name}, shape: {g.shape}')






# mnist = tf.keras.datasets.mnist

# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)






# #-------------Layer Noise
# class layer_noise(layers.Layer):
#     def __init__(self, **kwargs):
#         super(layer_noise, self).__init__(**kwargs)

#         w_init = tf.ones_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=[1]),
#             trainable=False,
#             **kwargs
#         )

#     def call(self, x):
#         return tf.cast(self.w, x.dtype)

# # def apply_noise(x):
# #     # w_init = tf.zeros_initializer()
# #     # weight = tf.Variable(
# #     # initial_value=w_init(shape=[1]),
# #     # trainable=False,
# #     # )
# #     # return x*weight
# #     #lnoise = layer_noise(x,name="layer_noise")
# #     nweights = lnoise(x)
# #     return x*nweights


# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10)
#     self.n1 = layer_noise(name="layer_noise")
#     #self.var = tf.Variable(2., trainable=False)

#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     x = self.d2(x)
#     w = self.n1(x)
#     x = x*w
#     return x
