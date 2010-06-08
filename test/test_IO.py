import numpy as np
import dnumpytest
import random
import subprocess

def run():
    import zlib#Make sure that it fails here if zlib is not available
    max_ndim = 6
    for i in xrange(1,max_ndim+1):
        src = dnumpytest.random_list(random.sample(xrange(1, 10),i))
        A = np.array(src, dtype=float, dist=True)
        fname = "distnumpt_test_matrix.npy"
        np.save(fname,A)
        Bf = np.load(fname, dist=False)
        Bd = np.load(fname, dist=True)
                
        if not dnumpytest.array_equal(Bf,Bd):
            err = "Input array:\n %s\n"%str(A)
            err += "The loaded array from DistNumPy:\n%s\n"%str(Bd)
            err += "The loaded array from NumPy:\n%s\n"%str(Bf)
            subprocess.check_call(('rm %s'%fname), shell=True)
            return (True, err)
        subprocess.check_call(('rm %s'%fname), shell=True)
    return (False, "")

if __name__ == "__main__":
    (err,msg) = run()
    print msg
