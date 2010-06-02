import numpy as np
import dnumpytest
import random

def run():
    max_ndim = 6
    for i in xrange(1,max_ndim+1):
        src = dnumpytest.random_list(random.sample(xrange(1, 10),i))
        A = np.array(src, dtype=float, dist=True)
        fname = "%sdistnumpt_test_matrix.npy"%dnumpytest.TmpSetDir
        np.save(fname,A)
        Bf = np.load(fname, dist=False)
        Bd = np.load(fname, dist=True)
                
        if not dnumpytest.array_equal(Bf,Bd):
            err = "Input array:\n %s\n"%str(A)
            err += "The loaded array from DistNumPy:\n%s\n"%str(Bd)
            err += "The loaded array from NumPy:\n%s\n"%str(Bf)
            return (True, err)
    return (False, "")

if __name__ == "__main__":
    (err,msg) = run()
    print msg
