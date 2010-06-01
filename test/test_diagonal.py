import numpy as np
import dnumpytest
import random

def run():
    niters = 50
    for i in xrange(niters):
        src = dnumpytest.random_list([random.randint(1, 20), \
                                      random.randint(1, 20)])
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        Cd = Ad.diagonal()
        Cf = Af.diagonal()
        if not dnumpytest.array_equal(Cd,Cf):
            err = "Input A:\n %s\n"%(str(Af))
            err += "The result from DistNumPy:\n%s\n"%str(Cd)
            err += "The result from NumPy:\n%s\n"%str(Cf)
            return (True, err)
    return (False, "")

if __name__ == "__main__":
    run()
