import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time


#--------------------------- create random tuning noise

# parameters
DTYPE     = "float32"
NSIZE     = 512*17
NC_NOISE  = 50
NC2_NOISE = int(NC_NOISE/2)

# initialization
N2 = int(NC_NOISE/2)
T  = NSIZE-1
Dt = T/(NSIZE-1)
t  = Dt*tf.cast(tf.random.uniform([NSIZE], maxval=NSIZE, dtype="int32"), DTYPE)
t  = t[tf.newaxis,:]
t  = tf.tile(t, [N2, 1])
k  = tf.range(1,int(N2+1), dtype=DTYPE)
f  = k/T
f  = f[:,tf.newaxis]
#c  = tf.constant(1.0, shape=[1,N2], dtype=DTYPE)

# find noise
for it in range(10):
    phi    = tf.random.uniform([NC2_NOISE,1], minval=0.0, maxval=2.0*np.pi, seed=0, dtype=DTYPE) 
    freq   = f * t
    c      = tf.random.uniform([1,N2], minval=0, maxval=1.0, seed=0, dtype=DTYPE) 

    argsin = tf.math.sin(2*np.pi*freq + phi)
    noise  = tf.matmul(c,argsin)
    noise  = 2.0 * (noise - tf.math.reduce_min(noise))/(tf.math.reduce_max(noise) - tf.math.reduce_min(noise)) - 1.0
    noise  = noise - tf.math.reduce_mean(noise)

    print(noise.shape)
    count, bins, ignored = plt.hist(noise, 20, density=True)

    ax = plt.gca()
    plt.savefig("tests.png")
    input()

exit(0)    
    

# N = 512

# a = -1*np.ones(N)
# b = np.ones(N)

# for i in range(100):
#     k = np.random.normal(loc=np.random.uniform(), scale=np.random.uniform(), size=N)
#     k = np.clip(k, 0, 1)

#     c = a*k + b*(1-k)
#     print(np.mean(a), np.mean(b), np.mean(c))
    
#     # count, bins, ignored = plt.hist(a, 15, density=True)
#     # plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')

#     # count, bins, ignored = plt.hist(b, 15, density=True)
#     # plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')

#     count, bins, ignored = plt.hist(c, 100, density=True)
#     plt.plot(bins, np.ones_like(bins), linewidth=2, color='y')
#     ax = plt.gca()
#     ax.set_ylim([0,1])
#     plt.savefig("tests.png")
#     input()
    
    
# N=16
# for i in range(10):
#     N=N*2
#     a = np.arange(N*N, dtype="float64")
#     a = np.reshape(a, [N,N])

#     tstart = time.time()
#     b = tf.convert_to_tensor(a)
#     print(str(N) + "CPU-GPU ", time.time() - tstart)

#     tstart = time.time()
#     c = b.numpy()
#     print(str(N) + "GPU-CPU ", time.time() - tstart)
    
#     print("\n")