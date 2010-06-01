import numpy as np
import dnumpytest
import random

def run():
    max_ndim = 6
    for i in range(1,max_ndim+1):
        src = dnumpytest.random_list(random.sample(range(1, 10),i))
        Ad = np.array(src, dtype=float, dist=True)
        Af = np.array(src, dtype=float, dist=False)
        for j in range(len(Ad.shape)):
            Cd = np.add.reduce(Ad,j)
            Cf = np.add.reduce(Af,j)
            if not dnumpytest.array_equal(Cd,Cf):
                err = "Input A:\n %s\n Input B:\n%s\n"%(str(Af),str(Bf))
                err += "The result from DistNumPy:\n%s\n"%str(Cd)
                err += "The result from NumPy:\n%s\n"%str(Cf)
                return (True, err)
    return (False, "")

if __name__ == "__main__":
    run()
