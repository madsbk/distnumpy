import numpy as np
import time
import sys

def compute_targets(base, targets):
    b1 = base[:,np.newaxis,:]
    d0 = b1-targets
    d1 = d0**2
    d2 = np.add.reduce(d1, 2)
    d3 = np.sqrt(d2)
    r  = np.max(d2, axis=1)
    return r

DIST = int(sys.argv[1])
ndims = int(sys.argv[2])
db_length = int(sys.argv[3])
niter = int(sys.argv[4])

targets = []
for i in xrange(niter):
    targets.append(np.ufunc_random(np.empty((db_length, ndims), \
                                             dtype=float, dist=DIST)))
base = np.ufunc_random(np.empty((db_length, ndims), dtype=float,\
                                dist=DIST))

np.core.multiarray.timer_reset()
np.core.multiarray.evalflush()
t1 = time.time()
for t in targets:
    compute_targets(base, t)
np.core.multiarray.evalflush()
t2 = time.time()

if DIST:
    print "(Par)",
else:
    print "(Seq)",
print "db has %d dims and %d*%d entries - time: %f sec."\
       %(ndims, db_length, niter, t2-t1),
if len(sys.argv) > 5:
    print " notes: ", sys.argv[5]
else:
    print ""
