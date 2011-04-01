import numpy as np
import time
import sys

def compute_targets_pyloop(base, targets):
    dist = (base[0]-targets[0])**2
    for i in range(1,len(base)):
        dist += (base[i]-targets[i])**2
    dist = np.sqrt(dist)
    r  = np.max(dist, axis=0)
    return r

def compute_targets(base, target):
    dist = (base-target[:,np.newaxis])**2
    dist = np.add.reduce(dist)
    dist = np.sqrt(dist)
    r  = np.max(dist, axis=0)
    return r

DIST = int(sys.argv[1])
ndims = int(sys.argv[2])
db_length = int(sys.argv[3])
niters = int(sys.argv[4])

targets = np.ufunc_random(np.empty((ndims, db_length), dtype=float, dist=DIST))
base = np.ufunc_random(np.empty((ndims,db_length), dtype=float, dist=DIST))

np.core.multiarray.timer_reset()
np.core.multiarray.evalflush()
t1 = time.time()
for i in xrange(niters):
    compute_targets_pyloop(base, targets[:,i])
np.core.multiarray.evalflush()
t2 = time.time()

print "ndims: %d, dbsize: %d, niters: %d - time: %f sec."\
      %(ndims, db_length, niters, t2-t1),
if DIST:
    print "(Dist) notes: %s"%sys.argv[5]
else:
    print "(Non-Dist) notes: %s"%sys.argv[5]

