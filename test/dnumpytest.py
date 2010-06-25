#Test and demonstration of DistNumPy.
import numpy as np
import random
import sys
import time
import subprocess
import os
import getopt

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
    seed = time.time()
    script_list = os.listdir(os.path.dirname(sys.argv[0]))

    try:
        sys.gettotalrefcount()
    except AttributeError:
        pydebug = False

    try:
        opts, args = getopt.getopt(sys.argv[1:],"s:f:",["seed=", "file="])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)

    for o, a in opts:
        if o == "-s":
            verbose = True
        elif o in ("-s", "--seed"):
            seed = int(a)
        elif o in ("-f", "--file"):
            script_list = [a]
        else:
            assert False, "unhandled option"

    random.seed(seed)

    script_list.append("test_sor.py")


    print "*"*100
    print "*"*31, "Testing Distributed Numerical Python", "*"*31
    for i in xrange(len(script_list)):
        f = script_list[i]
        if f.startswith("test_") and f.endswith("py"):
            m = f[:-3]#Remove ".py"
            m = __import__(m)
            print "*"*100
            print "Testing %s"%f
            if pydebug:
                r1 = sys.gettotalrefcount()
                try:
                    (err, msg) = m.run()
                except:
                    err = True
                    msg = "Error message: %s"%sys.exc_info()[1]
                r2 = sys.gettotalrefcount()
                if i == 0:
                    r2 -= 6 #The first load has 6 extra references.
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

