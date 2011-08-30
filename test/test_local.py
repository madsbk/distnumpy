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
    BS = np.BLOCKSIZE
    for i in xrange(2, max_ndim+2):
        src = dnumpytest.random_list(range(BS, i*BS, BS))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        Bd = Ad[BS:,...]
        Bf = Af[BS:,...]
        Cd = Bd.local()
        Cf = Bf
        Cd += 42.0
        Cf += 42.0
        Bd = Ad[BS*2:,...]
        Bf = Af[BS*2:,...]
        Cd = Bd.local()
        Cf = Bf
        Cd += 4.0
        Cf += 4.0
        Bd = Ad[:BS,...]
        Bf = Af[:BS,...]
        Cd = Bd.local()
        Cf = Bf
        Cd += 142.0
        Cf += 142.0
        Bd = Ad[:BS*2,...]
        Bf = Af[:BS*2,...]
        Cd = Bd.local()
        Cf = Bf
        Cd += 143.0
        Cf += 143.0
        Bd = Ad[...,:BS]
        Bf = Af[...,:BS]
        Cd = Bd.local()
        Cf = Bf
        Cd += 1042.0
        Cf += 1042.0
        if not dnumpytest.array_equal(Ad,Af):
            raise Exception("Uncorrect result array\n")

if __name__ == "__main__":
    run()
