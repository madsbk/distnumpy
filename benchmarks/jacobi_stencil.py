import numpy as np
import util

parser = util.Parsing()

DIST=parser.dist

W = int(parser.argv[0])
H = int(parser.argv[0])

forcedIter = int(parser.argv[1])

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

np.timer_reset()

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

timing = np.timer_getdict()

if np.RANK == 0:
    print 'jacobi_stencil - Iter: ', i, ' size:', np.shape(work)
    parser.pprint(timing)
    parser.write_dict(timing)
