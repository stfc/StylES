import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fftn

N = 20
x = np.arange(0.0,2.0*np.pi+2.0*np.pi/(N-1),2.0*np.pi/(N-1))
y = np.cos(1*x)

print(x[N-1])

plt.plot(x,y)
plt.savefig("test.png")
plt.close()

yf = fftn(y)

for i in range(len(yf)):
    print(i, int(yf[i].real))

plt.plot(yf)
plt.savefig("test2.png")