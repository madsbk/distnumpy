"""
pyHPC is a collection of useful numerical operations for distributed
NumPy arrays.

All communication is implemented in Python using DistNumPy in SPMD mode.
Therefore, any application that uses pyHPC must run in SPMD mode.
Please note that most operations in pyHPC do not support arbitrary
views of arrays. In most cases, array views must be aligned with the
global block-size.

"""
import numpy

if not numpy.SPMD_MODE:
    raise Exception("DistNumPy must run in SPMD mode\n")

from summa import summa
from lu import lu
from transpose import transpose
from fft import *

#Default matrix multiplication
matmul = summa


__all__ = ['summa', 'matmul', 'lu', 'transpose']
