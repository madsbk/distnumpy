import numpy as np
import pyHPC
import time
import sys

DIST=int(sys.argv[1])
M=int(sys.argv[2])
N=int(sys.argv[3])
K=int(sys.argv[4])
C=int(sys.argv[5])

A = np.empty([M,K], dtype=float, dist=DIST)
np.ufunc_random(A,A)

B = np.empty([K,M], dtype=float, dist=DIST)
np.ufunc_random(B,B)

if DIST:
    matmul = pyHPC.summa
else:
    matmul = np.dot

np.timer_reset()
np.evalflush()
start=time.time()
for i in range(C):
    tmp = matmul(A,B)
np.evalflush()
stop=time.time()

if np.RANK == 0:
    print "SUMMA M:%d, N:%d, K:%d, C:%d, time:"%(M,N,K,C), stop-start,
    if DIST:
        print "(Dist) notes: %s"%sys.argv[6]
    else:
        print "(Non-Dist) notes: %s"%sys.argv[6]

