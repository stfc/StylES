# switch between numpy and cupy
USE_CUPY = False

if (USE_CUPY):
    import cupy as nc
    from cupy import asnumpy as convert
else:
    import numpy as nc
    def convert(val):
        return val

