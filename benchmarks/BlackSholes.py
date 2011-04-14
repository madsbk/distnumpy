import numpy as np
import sys
import time

def CND(X):
    (a1,a2,a3,a4,a5) = (0.31938153, -0.356563782, 1.781477937, \
                        -1.821255978, 1.330274429)
    L = np.abs(X)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / np.sqrt(2*np.pi)*np.exp(-L*L/2.) * \
        (a1*K + a2*(K**2) + a3*(K**3) + a4*(K**4) + a5*(K**5))

    mask = X<0
    w = w * ~mask + (1.0-w)*mask
    return w

# Black Sholes Function
def BlackSholes(CallPutFlag,S,X,T,r,v):
    d1 = (np.log(S/X)+(r+v*v/2.)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if CallPutFlag=='c':
        return S*CND(d1)-X*np.exp(-r*T)*CND(d2)
    else:
        return X*np.exp(-r*T)*CND(-d2)-S*CND(-d1)

DIST=int(sys.argv[1])

N=int(sys.argv[2])

S = np.empty((N), dtype=float, dist=DIST)
np.ufunc_random(S,S)
S = S*4.0-2.0 + 60.0 #Price is 58-62

X=65.0
#T=0.25 Moved to loop
r=0.08
v=0.3

year=int(sys.argv[3])
day=1.0/year
T=day

np.core.multiarray.timer_reset()
np.core.multiarray.evalflush()
stime = time.time()
for t in xrange(year):
    np.sum(BlackSholes('c', S, X, T, r, v))/N
    T+=day
np.core.multiarray.evalflush()
print 'N: ', N, ' iter: ', year, 'in sec: ', time.time() - stime,
if DIST:
    print " (Dist) notes: %s"%sys.argv[4]
else:
    print " (Non-Dist) notes: %s"%sys.argv[4]
