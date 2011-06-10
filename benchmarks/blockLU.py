#Blocked LU factorization
import numpy as np
from scipy import linalg

SIZE = 4;
BS = 2;
Nblock = SIZE / BS;
Nlocal = BS;

# generate matrix to factorize
A = np.zeros((SIZE,SIZE), dtype=float);
for row in xrange(Nblock):
    for col in xrange(Nblock):
        for n in xrange(BS):
            for m in xrange(BS):
                t1 = float(BS)
                r = row*t1 + n
                c = col*t1 + m
                A[row*BS+n,col*BS+m] = r*c / (Nblock*t1*t1*Nblock)
        if row == col:
            for n in xrange(BS):
                A[row*BS+n,col*BS+n] = A[row*BS+n,col*BS+n] + 10

lapack_getrf = linalg.lapack.get_lapack_funcs(('getrf',),(A,))[0]

def DGETRF(A):
    (M,N) = A.shape
    N = min(M,N)
    piv = np.zeros(N, dtype=int)

    for k in xrange(0, N, BS):
        bs = min(BS, N - k)#Current block size

        #Factor diagonal and subdiagonal blocks
        A[k:N,k:k+bs], lpiv, info = lapack_getrf(A[k:N,k:k+bs], overwrite_a=False)

        #Adjust local pivots to global pivots
        piv[k:k+bs] = lpiv[0:bs] + k

        #Pivots columns not already done in LU on current block column
        for i in xrange(k, k+bs):
            tmp = A[i,0:k].copy()
            A[i,0:k] = A[piv[i],0:k]
            A[piv[i],0:k] = tmp

            tmp = A[i,k+bs:N].copy()
            A[i,k+bs:N] = A[piv[i],k+bs:N]
            A[piv[i],k+bs:N] = tmp

        #Triangular solve with matrix right hand side
        triled = linalg.tril(A[k:k+bs,k:k+bs], -1)
        triled += np.ma.identity(bs,dtype=float)

        A[k:k+bs,k+bs:N] = linalg.solve(triled, A[k:k+bs,k+bs:N])

        #Update trailing submatrix
        A[k+bs:N,k+bs:N] = A[k+bs:N,k+bs:N] - np.dot(A[k+bs:N,k:k+bs] , A[k:k+bs,k+bs:N])

    return A


lu = DGETRF(A.copy())
print lu

lu, lpiv, info = lapack_getrf(A.copy(), overwrite_a=False)
print lu
