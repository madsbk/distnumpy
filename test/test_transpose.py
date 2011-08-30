import numpy as np
import dnumpytest
from itertools import izip as zip
from itertools import permutations

def run():
    if not np.SPMD_MODE:
        print "[rank %d] Warning - ignored in non-SPMD mode\n"%(np.RANK),
        return
    try:#This test requires the pyHPC module
        import pyHPC
    except:
        print "[rank %d] Warning - ignored pyHPC not found\n"%(np.RANK),
        return
    if np.BLOCKSIZE > 10:
        print "[rank %d] Warning - ignored np.BLOCKSIZE too high\n"%(np.RANK),
        return

    niter = 5
    for m in range(np.BLOCKSIZE,niter*np.BLOCKSIZE, np.BLOCKSIZE):
        for n in range(np.BLOCKSIZE,niter*np.BLOCKSIZE, np.BLOCKSIZE):
            for k in range(np.BLOCKSIZE,niter*np.BLOCKSIZE, np.BLOCKSIZE):
                for axis in permutations(xrange(3)):
                    Asrc = dnumpytest.random_list([m,n,k])
                    Ad = np.array(Asrc, dtype=float, dist=True)
                    Af = np.array(Asrc, dtype=float, dist=False)
                    Bd = pyHPC.transpose(Ad, axis)
                    Bf = np.transpose(Af, axis)
                    if not dnumpytest.array_equal(Bd,Bf):
                        raise Exception("Uncorrect result matrix\n")

if __name__ == "__main__":
    run()
