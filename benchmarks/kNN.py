import numpy as np
import time
import sys

def compute_targets_pyloop(base, target):
    tmp = base-target[:,np.newaxis]
    dist= np.add.reduce(tmp)**2
    dist = np.sqrt(dist)
    r  = np.max(dist, axis=0)
    return r

def compute_targets(base, target, tmp1, tmp2):
    base = base[:,np.newaxis]
    target = target[:,:,np.newaxis]
    t = base - target
    #print t.shape, tmp1.shape
    np.subtract(base, target, tmp1)
    tmp1 **= 2
    np.add.reduce(tmp1, out=tmp2)
    np.sqrt(tmp2, tmp2)
    r  = np.max(tmp2, axis=0)
    return r

DIST = int(sys.argv[1])
ndims = int(sys.argv[2])
db_length = int(sys.argv[3])
step = int(sys.argv[4])

targets = np.ufunc_random(np.empty((ndims,db_length), dtype=float, dist=DIST))
base = targets #np.ufunc_random(np.empty((ndims,db_length), dtype=float, dist=DIST))

tmp1 = np.empty((ndims,step,step), dtype=float, dist=DIST)
tmp2 = np.empty((step,step), dtype=float, dist=DIST)

np.core.multiarray.timer_reset()
np.core.multiarray.evalflush()
t1 = time.time()
for i in xrange(0, db_length, step):
    for j in xrange(0, db_length, step):
        compute_targets(base[:,i:i+step], targets[:,j:j+step], tmp1, tmp2)
np.core.multiarray.evalflush()
t2 = time.time()

print "ndims: %d, dbsize: %d, step: %d - time: %f sec."\
      %(ndims, db_length, step, t2-t1),
if DIST:
    print "(Dist) notes: %s"%sys.argv[5]
else:
    print "(Non-Dist) notes: %s"%sys.argv[5]

