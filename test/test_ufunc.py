import numpy as np
import dnumpytest
import random

def run():
    max_ndim = 6
    for i in xrange(1,max_ndim+1):
        src = dnumpytest.random_list(random.sample(xrange(1, 10),i))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        ran = random.randint(0,i-1)
        if i > 1 and ran > 0:
            for j in range(0,ran):
                src = src[0]
        Bd = np.array(src, dtype=float, dist=True)
        Bf = np.array(src, dtype=float, dist=False)
        Cd = Ad + Bd + 42 + Bd[-1]
        Cf = Af + Bf + 42 + Bf[-1]
        Cd = Cd[::2] + Cd[::2,...] + Cd[0,np.newaxis]
        Cf = Cf[::2] + Cf[::2,...] + Cf[0,np.newaxis]
        Dd = np.array(Cd, dtype=float, dist=True)
        Df = np.array(Cf, dtype=float, dist=False)
        Dd[1:] = Cd[:-1]
        Df[1:] = Cf[:-1]
        Cd = Dd + Bd[np.newaxis,-1]
        Cf = Df + Bf[np.newaxis,-1]
        Cd[1:] = Cd[:-1]
        Cf[1:] = Cf[:-1]
        if not dnumpytest.array_equal(Cd,Cf):
            err = "Input A:\n %s\n Input B:\n%s\n"%(str(Af),str(Bf))
            err += "The result from DistNumPy:\n%s\n"%str(Cd)
            err += "The result from NumPy:\n%s\n"%str(Cf)
            return (True, err)

    for i in xrange(3,max_ndim+3):
        src = dnumpytest.random_list([i,i,i])
        Ad = np.array(src, dist=True, dtype=float)
        Af = np.array(src, dist=False, dtype=float)
        Bd = np.array(src, dist=True, dtype=float)
        Bf = np.array(src, dist=False, dtype=float)
        Cd = Ad[::2, ::2, ::2] + Bd[::2, ::2, ::2] + Ad[::2,1,2]
        Cf = Af[::2, ::2, ::2] + Bf[::2, ::2, ::2] + Af[::2,1,2]
        if not dnumpytest.array_equal(Cd,Cf):
            err = "Input A:\n %s\n Input B:\n%s\n"%(str(Af),str(Bf))
            err += "The result from DistNumPy:\n%s\n"%str(Cd)
            err += "The result from NumPy:\n%s\n"%str(Cf)
            return (True, err)

    return (False, "")

if __name__ == "__main__":
    print run()
