import numpy as np
from scipy import linalg
import pyHPC
from itertools import izip as zip

def lu(matrix):
    """
    Compute LU decompostion of a matrix.

    Parameters
    ----------
    a : array, shape (M, M)
        Array to decompose

    Returns
    -------
    p : array, shape (M, M)
        Permutation matrix
    l : array, shape (M, M)
        Lower triangular or trapezoidal matrix with unit diagonal.
    u : array, shape (M, M)
        Upper triangular or trapezoidal matrix
    """
    SIZE = matrix.shape[0]
    BS = np.BLOCKSIZE

    if matrix.shape[0] != matrix.shape[0]:
        raise Exception("LU only supports squared matricis")
    if not matrix.dist():
        raise Exception("The matrix is not distributed")

    if(SIZE % np.BLOCKSIZE != 0):
        raise Exception("The matrix dimensions must be divisible "\
                        "with np.BLOCKSIZE(%d)"%np.BLOCKSIZE)

    (prow,pcol) = matrix.pgrid()
    A = np.zeros((SIZE,SIZE), dtype=matrix.dtype, dist=True);A += matrix
    L = np.zeros((SIZE,SIZE), dtype=matrix.dtype, dist=True)
    U = np.zeros((SIZE,SIZE), dtype=matrix.dtype, dist=True)
    tmpL = np.zeros((SIZE,SIZE), dtype=matrix.dtype, dist=True)
    tmpU = np.zeros((SIZE,SIZE), dtype=matrix.dtype, dist=True)
    for k in xrange(0,SIZE,BS):
        bs = min(BS,SIZE - k) #Current block size
        kb = k / BS # k as block index

        #Compute vertical multiplier
        slice = ((kb,kb+1),(kb,kb+1))
        for a,l,u in zip(A.blocks(slice), L.blocks(slice), U.blocks(slice)):
            (p,tl,tu) = linalg.lu(a)
            if not (np.diag(p) == 1).all():#We do not support pivoting
                raise Exception("Pivoting was needed!")
            #There seems to be a transpose bug in SciPy's LU
            l[:] = tl.T
            u[:] = tu.T

        #Replicate diagonal block horizontal and vertical
        for tk in xrange(k+bs,SIZE,BS):
            tbs = min(BS,SIZE - tk) #Current block size
            L[tk:tk+tbs,k:k+bs] = U[k:k+tbs,k:k+bs]
            U[k:k+bs,tk:tk+tbs] = L[k:k+bs,k:k+tbs]

        if k+bs < SIZE:
            #Compute horizontal multiplier
            slice = ((kb,kb+1),(kb+1,SIZE/BS))
            for a,u in zip(A.blocks(slice), U.blocks(slice)):
                u[:] = np.linalg.solve(u.T,a.T).T

            #Compute vertical multiplier
            slice = ((kb+1,SIZE/BS),(kb,kb+1))
            for a,l in zip(A.blocks(slice), L.blocks(slice)):
                l[:] = np.linalg.solve(l,a)

            #Apply to remaining submatrix
            tmp = pyHPC.summa(L[:,:k+bs],U[:k+bs,:], ao=(k+bs,k),
                              bo=(k,k+bs), co=(k+bs,k+bs))
            A += tmp

    return (L, U)
