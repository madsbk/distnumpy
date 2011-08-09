import numpy as np
from scipy import linalg
import pyHPC

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

        #Compute diagonal block
        if A.pgrid_incoord((kb%prow,kb%pcol)):
            l1 = kb/prow * BS#Convert to local index
            l2 = kb/pcol * BS#Convert to local index
            l_A = A.local()[l1:l1+bs,l2:l2+bs]
            (l_P,l_L,l_U) = linalg.lu(l_A)
            if not (np.diag(l_P) == 1).all():#We do not support pivoting
                raise Exception("Pivoting was needed!")
            #There seems to be a transpose bug in SciPy's LU
            l_L = l_L.T
            l_U = l_U.T
            #Place L and U in there distributed array
            L.local()[l1:l1+bs,l2:l2+bs] = l_L
            U.local()[l1:l1+bs,l2:l2+bs] = l_U
        #Replicate diagonal block horizontal and vertical
        for tk in xrange(k+bs,SIZE,BS):
            tbs = min(BS,SIZE - tk) #Current block size
            L[tk:tk+tbs,k:k+bs] = U[k:k+tbs,k:k+bs]
            U[k:k+bs,tk:tk+tbs] = L[k:k+bs,k:k+tbs]

        #Compute multipliers
        for tk in xrange(k+bs,SIZE,BS):
            tbs = min(BS,SIZE - tk) #Current block size
            tkb = tk / BS # k as block index
            #Horizontal
            if A.pgrid_incoord((kb%prow,tkb%pcol)):
                l1 = kb/prow * BS#Convert to local index
                l2 = tkb/pcol * BS#Convert to local index
                l_U = U.local()[l1:l1+bs,l2:l2+tbs]
                l_A = A.local()[l1:l1+bs,l2:l2+tbs]
                l_U = linalg.solve(l_U.T, l_A.T).T
                U.local()[l1:l1+bs,l2:l2+tbs] = l_U
            #Vertical
            if A.pgrid_incoord((tkb%prow,kb%pcol)):
                l1 = tkb/prow * BS#Convert to local index
                l2 = kb/pcol * BS#Convert to local index
                l_L = L.local()[l1:l1+tbs,l2:l2+bs]
                l_A = A.local()[l1:l1+tbs,l2:l2+bs]
                l_L = linalg.solve(l_L , l_A)
                L.local()[l1:l1+tbs,l2:l2+bs] = l_L

        SUMMA = 1
        if SUMMA:
            A[k+bs:,k+bs:] -= pyHPC.summa(L[k+bs:,k:k+bs],U[k:k+bs,k+bs:])
        else:
            #Apply to remaining submatrix
            if k+bs < SIZE:
                #Replicate k'th block in L and U
                for tk in xrange(k+bs,SIZE,BS):
                    tbs = min(BS,SIZE - tk) #Current block size
                    tmpL[k+bs:,tk:tk+tbs] = L[k+bs:,k:k+bs]
                    #Replicate k'th row block in U vertically
                    tmpU[tk:tk+tbs,k+bs:] = U[k:k+bs,k+bs:]

                #Apply to remaining submatrix via matrix multiplication
                for tk1 in xrange(k+bs,SIZE,BS):
                    tbs1 = min(BS,SIZE - tk1) #Current block size
                    tkb1 = tk1 / BS # k as block index
                    for tk2 in xrange(k+bs,SIZE,BS):
                        tbs2 = min(BS,SIZE - tk2) #Current block size
                        tkb2 = tk2 / BS # k as block index
                        if A.pgrid_incoord((tkb1%prow,tkb2%pcol)):
                            l1 = tkb1/prow * BS#Convert to local index
                            l2 = tkb2/pcol * BS#Convert to local index
                            l_L = tmpL.local()[l1:l1+tbs1,l2:l2+tbs2]
                            l_U = tmpU.local()[l1:l1+tbs1,l2:l2+tbs2]
                            l_A = A.local()[l1:l1+l1,l2:l2+tbs2]
                            if l_A.size > 0:
                                l_A -= np.dot(l_L, l_U)
    return (L, U)
