import numpy as np
import dnumpytest
import random
import subprocess

def run():
    if np.SPMD_MODE:
        print "[rank %d] Warning - ignored in SPMD mode\n"%(np.RANK),
        return

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
            subprocess.check_call(('rm %s'%fname), shell=True)
            raise Exception("Uncorrect result array\n")
        subprocess.check_call(('rm %s'%fname), shell=True)

if __name__ == "__main__":
    run()
