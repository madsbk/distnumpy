import numpy as np
import util
import pyHPC

parser = util.Parsing()
DIST=parser.dist
M=int(parser.argv[0])
N=int(parser.argv[1])
K=int(parser.argv[2])
C=int(parser.argv[3])

A = np.empty([M,K], dtype=float, dist=DIST)
np.ufunc_random(A,A)

B = np.empty([K,M], dtype=float, dist=DIST)
np.ufunc_random(B,B)

if DIST:
    matmul = pyHPC.summa
else:
    matmul = np.dot

np.timer_reset()
for i in range(C):
    tmp = matmul(A,B)
timing = np.timer_getdict()

if np.RANK == 0:
    print "SUMMA M:%d, N:%d, K:%d, C:%d"%(M,N,K,C)
    parser.pprint(timing)
    parser.write_dict(timing)

