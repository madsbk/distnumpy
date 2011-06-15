import os, sys
sys.path.append(\
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../pyHPC'))
import numpy as np
import dnumpytest
import summa

def run():
    if not np.SPMD_MODE:
        print "[rank %d] Warning - ignored in non-SPMD mode\n"%(np.RANK),
        return

    #Non-view test - identical to the one in test_dot.py
    niter = 6
    for m in range(2,niter+2):
        for n in range(2,niter+2):
            for k in range(2,niter+2):
                Asrc = dnumpytest.random_list([k,m])
                Bsrc = dnumpytest.random_list([n,k])
                Ad = np.array(Asrc, dtype=float, dist=True)
                Af = np.array(Asrc, dtype=float, dist=False)
                Bd = np.array(Bsrc, dtype=float, dist=True)
                Bf = np.array(Bsrc, dtype=float, dist=False)
                Cd = summa.summa(Ad,Bd)
                Cf = np.dot(Af,Bf)
                if not dnumpytest.array_equal(Cd,Cf):
                    raise Exception("Uncorrect result matrix\n")
    niter *= 2
    Asrc = dnumpytest.random_list([niter,niter])
    Bsrc = dnumpytest.random_list([niter,niter])
    Ad = np.array(Asrc, dtype=float, dist=True)
    Af = np.array(Asrc, dtype=float, dist=False)
    Bd = np.array(Bsrc, dtype=float, dist=True)
    Bf = np.array(Bsrc, dtype=float, dist=False)
    Cd = np.zeros((niter,niter),dtype=float, dist=True)
    BS = np.BLOCKSIZE
    for m in xrange(0,niter-BS, BS):
        for n in xrange(0,niter-BS,BS):
            for k in xrange(0,niter-BS,BS):
                tAd = Ad[m:,k:]
                tAf = Af[m:,k:]
                tBd = Bd[k:,n:]
                tBf = Bf[k:,n:]
                tCd = Cd[m:,n:]
                tCd = summa.summa(tAd,tBd)
                tCf = np.dot(tAf,tBf)
                if not dnumpytest.array_equal(tCd,tCf):
                    raise Exception("Uncorrect result matrix\n")
if __name__ == "__main__":
    run()
