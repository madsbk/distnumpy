#Test and demonstration of DistNumPy.

import numpy as np
import random

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
    

def ufunc(max_ndim=5):
    for i in range(1,max_ndim+1):
        src = random_list(random.sample(range(1, 10),i))
        Ad = np.array(src, dtype=float, isdist=True)
        Af = np.array(src, dtype=float, isdist=False)
        
        ran = random.randint(0,i-1)
        if i > 1 and ran > 0:
            for j in range(0,ran):
                src = src[0]

        Bd = np.array(src, dtype=float, isdist=True)
        Bf = np.array(src, dtype=float, isdist=False)
        print "1"
        Cd = Ad + Bd + 42 + Bd[-1]
        print "2"
        Cf = Af + Bf + 42 + Bf[-1]
        print "3"
        Cd = Cd[::2] + Cd[::2,...] + Cd[0,np.newaxis]
        print "4"
        Cf = Cf[::2] + Cf[::2,...] + Cf[0,np.newaxis]
        print "5"
        Dd = np.array(Cd, dtype=float, isdist=True)
        Df = np.array(Cf, dtype=float, isdist=False)
        
        Dd[1:] = Cd[:-1]
        Df[1:] = Cf[:-1]

        if not array_equal(Dd,Df):
            print "Error in ufunc_broadcast!"
            print Dd
            print "The distributed array no equal to facit"
            print Df
            return False
    return True

def ufunc_reduce(max_ndim=6):
    for i in range(1,max_ndim+1):
        src = random_list(random.sample(range(1, 10),i))
        Ad = np.array(src, dtype=float, isdist=True)
        Af = np.array(src, dtype=float, isdist=False)
        for j in range(len(Ad.shape)):
            Cd = np.add.reduce(Ad,j)
            Cf = np.add.reduce(Af,j)
            if not array_equal(Cd,Cf):
                return False
    return True





print "*"*100
print "Testing ufunc"
if ufunc(6):
    print "Succes"
else:
    print "Fail!"



print "*"*100
print "Testing ufunc reduce (no views)"
if ufunc_reduce(6):
    print "Succes"

