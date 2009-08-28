import time
import numpy as np
import random

square=np.square
add=np.add
less_equal=np.less_equal
sum=np.sum


def MonteCarlo_num_Mflops(Num_samples):
    # 3 flops in x^2+y^2 and 1 flop in random routine
    return  (Num_samples* 4.0)/10E6

def random_list(dims):
	if len(dims) == 0:
		return random.randint(0,100000)
	
	list = []
	for i in range(dims[-1]):
		list.append(random_list(dims[0:-1]))
	return list	

def MC_int(s, dist):
    x = np.empty([s,s], dtype=float, dist=dist)
    y = np.empty([s,s], dtype=float, dist=dist)

    start=time.time()
    square(x,x)
    square(y,y)
    add(x,y,x)
    less_equal(x, 1, x)
    add.reduce(x)*4.0/s
    stop=time.time()
    print 'Performance of MC_int (in sec): ', stop-start,
    if dist:
        print "(Distributed)"
    else:
        print "(Not distributed)"
		    

N=8000

MC_int(N, True)
MC_int(N, False)
MC_int(N, True)
MC_int(N, False)

