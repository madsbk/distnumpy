#A direct port from scimark2's LU.c
#Translated by Brian Vinter

import numpy as np
import time
import sys

DIST=int(sys.argv[1])
N = int(sys.argv[2])
I = int(sys.argv[3])

def LU_factor(A):
    M, N = np.shape(A)
    minMN =  np.min([M, N])
    pivot = np.zeros(minMN, dtype=float, dist=DIST)
    temp =  np.empty(minMN, dtype=float, dist=DIST)

    for j in xrange(I):
        jp= 0#np.argmax(np.abs(A[:,j]))
        pivot[j] = jp;

        if A[jp][j]==0:
            return False

        if jp != j: #Do full pivot!!!
            temp[:]=A[jp,:]
            A[jp,:]=A[j,:]
            A[j,:]=temp[:]

        if j < M-1:
            recp =  1.0 / A[j][j]
            A[j:,j] *= recp

        if (j < minMN-1):
            t1 = A[j+1:,j+1:] - A[j+1:,j] * A[j,j+1:]
            A[j+1:,j+1:] = t1

    return A, pivot

A = np.ufunc_random(np.empty((N,N), dtype=float, dist=DIST))
np.core.multiarray.timer_reset()
np.core.multiarray.evalflush()
t1 = time.time()
LU_factor(A)
np.core.multiarray.evalflush()
t1 = time.time() - t1

print 'Iter: ', I, ' size: ', N,' time: ', t1,
if A.dist():
    print "(Dist) notes: %s"%sys.argv[4]
else:
    print "(Non-Dist) notes: %s"%sys.argv[4]

