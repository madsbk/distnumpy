import numpy as np
import sys
import time

DIST=int(sys.argv[1])

W = int(sys.argv[2])
H = int(sys.argv[2])

forcedIter = int(sys.argv[3])

full = np.zeros((W+2,H+2), dtype=np.double, dist=DIST)
work = np.zeros((W,H), dtype=np.double, dist=DIST)
diff = np.zeros((W,H), dtype=np.double, dist=DIST)
tmpdelta = np.zeros((W), dtype=np.double, dist=DIST)

cells = full[1:-1, 1:-1]
up    = full[1:-1, 0:-2]
left  = full[0:-2, 1:-1]
right = full[2:  , 1:-1]
down  = full[1:-1, 2:  ]

full[:,0]  += -273.15
full[:,-1] += -273.15
full[0,:]  +=  40.0
full[-1,:] += -273.13

np.core.multiarray.timer_reset()
np.core.multiarray.evalflush()
t1 = time.time()

epsilon=W*H*0.010
delta=epsilon+1
i=0
while (forcedIter and forcedIter > i) or \
      (forcedIter == 0 and epsilon<delta):
  i+=1
  work[:] = cells
  work += up
  work += left
  work += right
  work += down
  work *= 0.2
  np.subtract(cells,work,diff)
  np.absolute(diff, diff)
  np.add.reduce(diff,out=tmpdelta)
  delta = np.add.reduce(tmpdelta)
  cells[:] = work

np.core.multiarray.evalflush()
t2 = time.time()
print 'Iter: ', i, ' size: ', H,' time: ', t2-t1,
if DIST:
    print "(Dist) notes: %s"%sys.argv[4]
else:
    print "(Non-Dist) notes: %s"%sys.argv[4]

