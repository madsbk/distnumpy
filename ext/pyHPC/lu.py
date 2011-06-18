import numpy as np
from scipy import linalg

def lu(A):
    """
    Compute LU decompostion of a matrix.

    Parameters
    ----------
    a : array, shape (M, N)
        Array to decompose

    Returns
    -------
    p : array, shape (M, M)
        Permutation matrix
    l : array, shape (M, K)
        Lower triangular or trapezoidal matrix with unit diagonal.
        K = min(M, N)
    u : array, shape (K, N)
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

        L[k:k+bs,k:k+bs] = diagL
        U[k:k+bs,k:k+bs] = diagU

        if k+bs < SIZE:
            #Compute multipliers
            for i in xrange(k+bs,SIZE,BS):
                tbs = min(BS,SIZE - i) #Block size of the i'th block
                L[i:i+tbs,k:k+bs] = linalg.solve(diagU.T, A[i:i+tbs,k:k+bs].T).T
                U[k:k+bs,i:i+tbs] = linalg.solve(diagL , A[k:k+bs,i:i+tbs])

            #Apply to remaining submatrix
            A[k+bs:, k+bs:] -= np.dot(L[k+bs:,k:k+bs], U[k:k+bs,k+bs:])

    A = L + U - np.identity(SIZE) #Merge L and U into A.

    return (L, U)

