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

    max_ndim = 3
    for i in xrange(4, max_ndim+4):
        SIZE = i*np.BLOCKSIZE
        src = dnumpytest.random_list(range(np.BLOCKSIZE*2, SIZE, np.BLOCKSIZE))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)

        slice = [((1,Ad.shape[0]/np.BLOCKSIZE))]
        for d in xrange(Ad.ndim-1):
            slice.append((0,Ad.shape[d+1]/np.BLOCKSIZE))

        for a in Ad.blocks(slice):
            a += 100.0

        Af[np.BLOCKSIZE:] += 100.0

        if not dnumpytest.array_equal(Ad,Af):
            raise Exception("Uncorrect result array\n")

    max_ndim = 3
    for i in xrange(4, max_ndim+4):
        SIZE = i*np.BLOCKSIZE
        src = dnumpytest.random_list(range(np.BLOCKSIZE*2, SIZE, np.BLOCKSIZE))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)

        slice = [((0,(Ad.shape[0]/np.BLOCKSIZE)-1))]
        for d in xrange(Ad.ndim-1):
            slice.append((0,Ad.shape[d+1]/np.BLOCKSIZE))

        for a in Ad.blocks(slice):
            a += 100.0

        Af[:-np.BLOCKSIZE] += 100.0

        if not dnumpytest.array_equal(Ad,Af):
            raise Exception("Uncorrect result array\n")


if __name__ == "__main__":
    run()
