#N-dimensional transpose
import numpy as np
import math
from itertools import izip as zip


def transpose(a, axis=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    Returns
    -------
    p : ndarray
        `a` with its axes permuted.

    Examples
    --------
    >>> x = np.ones((1, 2, 3))
    >>> np.transpose(x, (1, 0, 2)).shape
    (2, 1, 3)

    """
    if not a.dist():
        raise Exception("The array must be distributed")

    for d in a.shape:
        if(d % np.BLOCKSIZE != 0):
            raise Exception("The array dimensions must be divisible "\
                            "with np.BLOCKSIZE(%d)"%np.BLOCKSIZE)
    ndims = len(a.shape)

    if axis is None:
        axis = [d for d in reversed(xrange(len(a.shape)))]

    if len(axis) != ndims:
        raise Exception("If not None, axis must have length equal the "\
                        "number of dimensions in a")

    BS = np.BLOCKSIZE
    BSf = float(BS)
    out = np.zeros([a.shape[d] for d in axis], dtype=float, dist=True)

    #Transpose all distributed array blocks
    finish = False
    coord = [0]*ndims
    while not finish:
        s = "out[%d:%d+BS"%(coord[axis[0]],coord[axis[0]])
        for shape in axis[1:]:
            s += ",%d:%d+BS"%(coord[shape],coord[shape])
        s += "] = a[%d:%d+BS"%(coord[0], coord[0])
        for shape in coord[1:]:
            s += ",%d:%d+BS"%(shape,shape)
        s += "]"
        exec s

        for i in xrange(ndims):
            coord[i] += BS
            if coord[i] >= a.shape[i]:
                if i == ndims-1:
                    finish = True
                    break
                coord[i] = 0
            else:
                break

    #Transpose all local array blocks
    local = out.local()
    finish = False
    coord = [0]*ndims
    while not finish:
        s = "block = local[%d:%d+BS"%(coord[0],coord[0])
        for shape in coord[1:]:
            s += ",%d:%d+BS"%(shape,shape)
        s += "]"
        exec s
        block[:] = np.transpose(block.copy(),axis)

        for i in xrange(ndims):
            coord[i] += BS
            if coord[i] >= local.shape[i]:
                if i == ndims-1:
                    finish = True
                    break
                coord[i] = 0
            else:
                break

    return out
