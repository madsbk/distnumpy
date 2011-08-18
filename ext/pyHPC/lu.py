import numpy as np
from scipy import linalg
import pyHPC
from itertools import izip as zip

def luSEQ(A):
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

    if A.shape[0] != A.shape[0]:
        raise Exception("LU only supports squared matricis")
    SIZE = A.shape[0]
    BS = np.BLOCKSIZE
    L = np.zeros((SIZE,SIZE), dtype=float)
    U = np.zeros((SIZE,SIZE), dtype=float)

    for k in xrange(0,SIZE,BS):
        bs = min(BS,SIZE - k) #Current block size

        diagA = A[k:k+bs,k:k+bs]
        (p,diagL,diagU) = linalg.lu(diagA)
        if not (np.diag(p) == 1).all():#We do not support pivoting
            raise Exception("Pivoting was needed!")

        #There seems to be a transpose bug in SciPy's LU
        diagL = diagL.T
        diagU = diagU.T

        L[k:k+bs,k:k+bs] = diagL
        U[k:k+bs,k:k+bs] = diagU

        if k+bs < SIZE:
            #Compute multipliers
            for i in xrange(k+bs,SIZE,BS):
                tbs = min(BS,SIZE - i) #Block size of the i'th block
                L[i:i+tbs,k:k+bs] = np.linalg.solve(diagU.T, A[i:i+tbs,k:k+bs].T).T
                U[k:k+bs,i:i+tbs] = np.linalg.solve(diagL , A[k:k+bs,i:i+tbs])
            #Apply to remaining submatrix
            A[k+bs:, k+bs:] -= np.dot(L[k+bs:,k:k+bs], U[k:k+bs,k+bs:])

    A = L + U - np.identity(SIZE) #Merge L and U into A.

    return (L, U)

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
    A = np.zeros((SIZE,SIZE), dtype=float, dist=True);A += matrix
    L = np.zeros((SIZE,SIZE), dtype=float, dist=True)
    U = np.zeros((SIZE,SIZE), dtype=float, dist=True)
    tmpL = np.zeros((SIZE,SIZE), dtype=float, dist=True)
    tmpU = np.zeros((SIZE,SIZE), dtype=float, dist=True)
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
            A[k+bs:,k+bs:] -= pyHPC.summa(L[k+bs:,k:k+bs],U[k:k+bs,k+bs:])

    return (L, U)
