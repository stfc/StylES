# switch between numpy and cupy
USE_GPU = False

if (USE_GPU):
    import cupy as nc
    from cupy import asnumpy as convert
else:
    import numpy as nc
    def convert(val):
        return val

