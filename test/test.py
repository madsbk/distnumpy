#Test and demonstration of DistNumPy.
import numpy as np
import random
import sys
import time

try:
    size = int(sys.argv[1])
except IndexError:
    size = 5
try:
    seed = int(sys.argv[2])
except IndexError:
    seed = time.time()
random.seed(seed)

pydebug = True
try:
    sys.gettotalrefcount()
except AttributeError:
    pydebug = False

def funtest(fun, str):
    print "*"*100
    print "Testing %s"%str
    if pydebug:
        r1 = sys.gettotalrefcount()
        out = fun(6)
        r2 = sys.gettotalrefcount()
        if r2-2 != r1:
            print "Memory leak - totrefcount: from %d to %d"%(r1,r2)
    else:
        out = fun(6)
    if out is None:
        print "Succes"
    else:
        print "Error in %s! Random seed: %d"%(str, seed)
        for o in out[2:]:
            print "Input array:"
            print o
        print "The result from DistNumPy:"
        print out[0]
        print "The result from NumPy:"
        print out[1]
        sys.exit()

def array_equal(A,B):
    if type(A) is not type(B):
        return False
    elif (not type(A) == type(np.array([]))) and (not type(A) == type([])):
        if A == B:
            return True
        else:
            return False

    A = A.flatten()
    B = B.flatten()
    if not len(A) == len(B):
        return False

    for i in range(len(A)):
        if not A[i] == B[i]:
            return False
    return True

def random_list(dims):
    if len(dims) == 0:
        return random.randint(0,100000)

    list = []
    for i in range(dims[-1]):
        list.append(random_list(dims[0:-1]))
    return list

def ufunc(max_ndim):
    for i in xrange(1,max_ndim+1):
        src = random_list(random.sample(xrange(1, 10),i))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        ran = random.randint(0,i-1)
        if i > 1 and ran > 0:
            for j in range(0,ran):
                src = src[0]
        Bd = np.array(src, dtype=float, dist=True)
        Bf = np.array(src, dtype=float, dist=False)
        Cd = Ad + Bd + 42 + Bd[-1]
        Cf = Af + Bf + 42 + Bf[-1]
        Cd = Cd[::2] + Cd[::2,...] + Cd[0,np.newaxis]
        Cf = Cf[::2] + Cf[::2,...] + Cf[0,np.newaxis]
        Dd = np.array(Cd, dtype=float, dist=True)
        Df = np.array(Cf, dtype=float, dist=False)
        Dd[1:] = Cd[:-1]
        Df[1:] = Cf[:-1]
        Cd = Dd + Bd[np.newaxis,-1]
        Cf = Df + Bf[np.newaxis,-1]
        Cd[1:] = Cd[:-1]
        Cf[1:] = Cf[:-1]
        if not array_equal(Cd,Cf):
            return (Cd, Cf, Af, Bf)

    for i in xrange(1,max_ndim+1):
        src = random_list([max_ndim,max_ndim,max_ndim])
        Ad = np.array(src, dist=True, dtype=float)
        Af = np.array(src, dist=False, dtype=float)
        Bd = np.array(src, dist=True, dtype=float)
        Bf = np.array(src, dist=False, dtype=float)
        Cd = Ad[::2, ::2, ::2] + Bd[::2, ::2, ::2] + Ad[::2,1,2]
        Cf = Af[::2, ::2, ::2] + Bf[::2, ::2, ::2] + Af[::2,1,2]
        if not array_equal(Cd,Cf):
            return (Cd, Cf, Af, Bf)

    return None

def ufunc_reduce(max_ndim):
    for i in range(1,max_ndim+1):
        src = random_list(random.sample(range(1, 10),i))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        for j in range(len(Ad.shape)):
            Cd = np.add.reduce(Ad,j)
            Cf = np.add.reduce(Af,j)
            if not array_equal(Cd,Cf):
                return (Cd,Cf,Af)
    return None

def diagonal(niters):
    niters *= 10
    for i in xrange(niters):
        src = random_list([random.randint(1, 20), \
                           random.randint(1, 20)])
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        Cd = Ad.diagonal()
        Cf = Af.diagonal()
        if not array_equal(Cd,Cf):
            return (Cd,Cf,Af)
    return None

def matmul(niter):
    for m in range(2,niter+2):
        for n in range(2,niter+2):
            for k in range(2,niter+2):
                Asrc = random_list([k,m])
                Bsrc = random_list([m,k])
                Ad = np.array(Asrc, dtype=float, dist=True)
                Af = np.array(Asrc, dtype=float, dist=False)
                Bd = np.array(Bsrc, dtype=float, dist=True)
                Bf = np.array(Bsrc, dtype=float, dist=False)
                Cd = np.dot(Ad,Bd)
                Cf = np.dot(Af,Bf)
                if not array_equal(Cd,Cf):
                    return (Cd, Cf, Af, Bf)
    return None


funtest(ufunc, "ufunc")
funtest(ufunc_reduce, "ufunc reduce (no views)")
funtest(diagonal, "diagonal (no views)")
funtest(matmul, "matrix multiplication (no views)")
