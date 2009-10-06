import time
import sys
from numpy import *

def jacobi(A, B, tol=0.005, forcedIter=0):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    print "jacobi solver - dist:", A.dist()
    h = zeros(shape(B), float, dist=A.dist())
    dmax = 1.0
    n = 0
    tmp0 = empty(shape(A), float, dist=A.dist())
    tmp1 = empty(shape(B), float, dist=A.dist())
    t1 = time.time()
    while (forcedIter and forcedIter > n) or \
          (forcedIter == 0 and dmax > tol):
        n += 1
        multiply(A,h,tmp0)
        add.reduce(tmp0,1,out=tmp1)
        #tmp2 = diag(A)
        tmp2 = zeros(shape(B), float, dist=A.dist())
        subtract(B, tmp1, tmp1)
        divide(tmp1, tmp2, tmp1)
        hnew = h + tmp1
        subtract(hnew,h,tmp2)
        divide(tmp2,h,tmp1)
        absolute(tmp1,tmp1)
        dmax = maximum.reduce(tmp1)
        h = hnew

    print "SOLVED at: Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
    print "                           Time = ", time.time() - t1, "seconds"

    return h


def gauss_seidel(A, B, tol=0.005, init_val=0.0):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    dist = False
    if(A.dist()):
        dist = True

    print "gauss_siedel - dist:", dist
    if init_val == 0.0:
        h = zeros(shape(B), float, dist=dist)
        hnew = zeros(shape(B), float, dist=dist)
    else:
        h = init_val
        hnew[:] = h[:]
    dmax = 1.0
    n = 0

    t1 = t = time.time()
    while dmax > tol:
        n += 1
        #print "  Iteration = ", n
        for i in range(len(B)):
            #print "add.reduce(multiply(A,h)) ", add.reduce(multiply(A,h))
            hnew[i] = h[i] + (B[i] - add.reduce(multiply(A[i,:],hnew)))/ A[i,i]
            #divide(add.reduce(subtract(B, multiply(hnew, A[i,:]))), A[i,i])
            #( B - h * A[i,:]) / A[i,i]
        dmax = max(abs(divide(subtract(h,hnew), h)))
        #print "Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
##        print_arr(h, "h")
##        print_arr(hnew, "hnew")
        if (time.time() - t) > 2.0:
            print "Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
            t = time.time()
        h[:] = hnew[:]
##        print n, ":", dmax,  h
    print "SOLVED at: Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
    print "                           Time = ", time.time() - t1, "seconds"

    return h

def sor(A, B, rf=1.3, tol=0.005, forcedIter=0):
    '''itteratively solving for matrix A with solution vector B
       rf = relaxation factor (rf > 1.0; eg rf = 1.1)
       tol = tolerance for dh/h
    '''

    print "successive over-relaxation (sor) - dist:", A.dist(), ""
    h = zeros(shape(B), float, dist=A.dist())
    hnew = zeros(shape(B), float, dist=A.dist())
    dmax = 1.0
    n = 0

    t = time.time()
    while (forcedIter and forcedIter > n) or \
          (forcedIter == 0 and dmax > tol):
        n += 1

        for i in range(len(B)):
            t1 = multiply(A[i],hnew)
            t2 = add.reduce(t1)
            hnew[i] = h[i] + (B[i] - t2) / A[i,i]

        #over-relaxation
        dh = subtract(hnew, h)
        multiply(rf,dh,dh)
        add(h,dh,hnew)
        subtract(h,hnew,dh)
        divide(dh,h,dh)
        absolute(dh,dh)
        dmax = maximum.reduce(dh)
        h[:] = hnew[:]

    print "SOLVED at: Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
    print "                           Time = ", time.time() - t, "seconds"

    return h


d = int(sys.argv[1])
size = int(sys.argv[2])
iter = int(sys.argv[3])

#A = array([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 4, -1], [0, -1, -1, 4]], float, dist=d)
#B = array([1,2,0,1], float, dist=d)


A = zeros([size,size], dtype=float, dist=d)
ufunc_random(A,A)

B = zeros([size], dtype=float, dist=d)
ufunc_random(B,B)

C = jacobi(A, B, forcedIter=iter)

