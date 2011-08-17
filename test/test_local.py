import numpy as np
import dnumpytest

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

    max_ndim = 4
    for i in xrange(2, max_ndim+2):
        src = dnumpytest.random_list(range(np.BLOCKSIZE, i*np.BLOCKSIZE, np.BLOCKSIZE))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        Bd = Ad
        Bf = Af
        Cd = Bd.local()
        Cf = Bf
        Cd += 42.0
        Cf += 42.0
        if not dnumpytest.array_equal(Cd,Cf):
            raise Exception("Uncorrect result array\n")

if __name__ == "__main__":
    run()
