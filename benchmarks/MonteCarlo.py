import time
import numpy as np
import random
import sys

def MC_int(c, s, dist):
    x = np.empty([s], dtype=np.double, dist=dist)
    y = np.empty([s], dtype=np.double, dist=dist)
    sum=0.0
    np.core.multiarray.timer_reset()
    np.core.multiarray.evalflush()
    start=time.time()
    for i in range(c):
        np.ufunc_random(x,x)
        np.ufunc_random(y,y)
        np.square(x,x)
        np.square(y,y)
        np.add(x,y,x)
        z = np.less_equal(x, 1)
        sum += np.add.reduce(z)*4.0/s
    sum = sum / c
    np.core.multiarray.evalflush()
    stop=time.time()
    print 'Pi: ', sum, ' with ', s,' samples in sec: ', stop-start,
    if dist:
        print "(Dist) notes: %s"%sys.argv[4]
    else:
        print "(Non-Dist) notes: %s"%sys.argv[4]


D=int(sys.argv[1])
N=int(sys.argv[2])
C=int(sys.argv[3])


MC_int(C, N, D)

