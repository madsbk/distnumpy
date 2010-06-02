#Test and demonstration of DistNumPy.
import numpy as np
import random
import sys
import time
import subprocess
import os

DataSetDir = os.path.join(os.path.join(\
             os.path.dirname(sys.argv[0]), "datasets"), "")

TmpSetDir = os.path.join("/",os.path.join("tmp", ""))

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


if __name__ == "__main__":
    pydebug = True
    try:
        sys.gettotalrefcount()
    except AttributeError:
        pydebug = False
    try:
        seed = int(sys.argv[1])
    except IndexError:
        seed = time.time()
    random.seed(seed)
    
    print "*"*100
    print "*"*31, "Testing Distributed Numerical Python", "*"*31
    for f in os.listdir(os.path.dirname(sys.argv[0])):
        if f.startswith("test_") and f.endswith("py"):
            m = f[:-3]#Remove ".py"
            m = __import__(m)
            print "*"*100
            print "Testing %s"%f
            if pydebug:
                r1 = sys.gettotalrefcount()
                (err, msg) = m.run()
                r2 = sys.gettotalrefcount()
                if r2 != r1:
                    print "Memory leak - totrefcount: from %d to %d"%(r1,r2)
            else:
                (err, msg) = m.run()
            if err:
                print "Error in %s! Random seed: %d"%(f, seed)
                print msg
            else:
                print "Succes"

    print "*"*100
    print "*"*46, "Finish", "*"*46
    print "*"*100
    
