import time
import numpy as np
import random

def MonteCarlo_num_Mflops(Num_samples):
    # 3 flops in x^2+y^2 and 1 flop in random routine
    return  (Num_samples* 4.0)/10E6

def MC_int(c, s, dist):
    x = np.empty([s], dtype=np.double, dist=dist)
    y = np.empty([s], dtype=np.double, dist=dist)
    sum=0.0
    start=time.time()
    np.ufunc_random(x,x)
    np.ufunc_random(y,y)
    for i in range(c):
        np.square(x,x)
        np.square(y,y)
        np.add(x,y,x)
        x = np.less_equal(x, 1)
        sum += np.add.reduce(x)*4.0/s
    stop=time.time()
    print 'Pi: ', sum, ' in sec: ', stop-start,
    if dist:
        print "(Distributed)"
    else:
        print "(Not distributed)"
		    

N=10000000

MC_int(1, N, True)
MC_int(1, N, False)
MC_int(1, N, True)
MC_int(1, N, False)

