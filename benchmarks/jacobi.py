import sys
import numpy as np
import util

parser = util.Parsing(sys.argv[1:])

def jacobi(A, B, tol=0.005, forcedIter=0):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    h = np.zeros(np.shape(B), float, dist=A.dist())
    dmax = 1.0
    n = 0
    tmp0 = np.empty(np.shape(A), float, dist=A.dist())
    tmp1 = np.empty(np.shape(B), float, dist=A.dist())
    AD = np.diagonal(A)
    np.timer_reset()
    while (forcedIter and forcedIter > n) or \
          (forcedIter == 0 and dmax > tol):
        n += 1
        np.multiply(A,h,tmp0)
        np.add.reduce(tmp0,1,out=tmp1)
        tmp2 = AD
        np.subtract(B, tmp1, tmp1)
        np.divide(tmp1, tmp2, tmp1)
        hnew = h + tmp1
        np.subtract(hnew,h,tmp2)
        np.divide(tmp2,h,tmp1)
        np.absolute(tmp1,tmp1)
        dmax = np.maximum.reduce(tmp1)
        h = hnew

    timing = np.timer_getdict()
    print timing
    print 'Iter: ', n, ' size:', np.shape(A)
    parser.pprint(timing)

    return h

size = int(parser.argv[0])
iter = int(parser.argv[1])

#A = array([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 4, -1], [0, -1, -1, 4]], float, dist=d)
#B = array([1,2,0,1], float, dist=d)

A = np.zeros([size,size], dtype=float, dist=parser.dist)
np.ufunc_random(A,A)

B = np.zeros([size], dtype=float, dist=parser.dist)
np.ufunc_random(B,B)

C = jacobi(A, B, forcedIter=iter)

