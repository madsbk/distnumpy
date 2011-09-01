import time
import numpy as np
import pyHPC
import util
from scipy import linalg

def gen_matrix(SIZE,dist):
    A = np.zeros((SIZE,SIZE), dtype=float, dist=dist);
    np.ufunc_random(A,A)
    for i in xrange(SIZE):
        A[i,i] *= 2**30
    return A

parser = util.Parsing()
DIST = parser.dist
SIZE = int(parser.argv[0])
A = gen_matrix(SIZE, DIST)

if DIST:
    f = pyHPC.lu
else:
    f = linalg.lu

np.timer_reset()
f(A)
timing = np.timer_getdict()

if np.RANK == 0:
    print 'hpcLU - size:,',np.shape(A)

parser.pprint(timing)
parser.write_dict(timing)

