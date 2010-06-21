import numpy as np
import time
import random
import sys

W = int(sys.argv[2])
H = int(sys.argv[2])
LIVING_LOW= 2
LIVING_HIGH = 3
ALIVE = 3
DIST = int(sys.argv[1])

DISPLAY = False

full      = np.zeros((W+2,H+2), dtype=np.long, dist=DIST)
dead      = np.zeros((W,H),     dtype=np.long, dist=DIST)
live      = np.zeros((W,H),     dtype=np.long, dist=DIST)
live2     = np.zeros((W,H),     dtype=np.long, dist=DIST)
neighbors = np.zeros((W,H),     dtype=np.long, dist=DIST)

cells = full[1:W+1,1:H+1]
ul = full[0:W, 0:H]
um = full[0:W, 1:H+1]
ur = full[0:W, 2:H+2]
ml = full[1:W+1, 0:H]
mr = full[1:W+1, 2:H+2]
ll = full[2:W+2, 0:H]
lm = full[2:W+2, 1:H+1]
lr = full[2:W+2, 2:H+2]

random.seed(time.time())
for i in range(W):
  for j in range(H):
      if random.random() > .8:
          cells[i][j] = 1

t1 = time.time()

for i in range(int(sys.argv[3])):
    print i
    # zero neighbors
    np.bitwise_and(neighbors,0,neighbors)
    # count neighbors
    neighbors += ul
    neighbors += um
    neighbors += ur
    neighbors += ml
    neighbors += mr
    neighbors += ll
    neighbors += lm
    neighbors += lr
    # extract live cells neighbors
    np.multiply(neighbors, cells, live)
    # find all living cells among the already living
    np.equal(live, LIVING_LOW, live2)
    np.equal(live, LIVING_HIGH, live)
    # merge living cells into 'live'
    np.bitwise_or(live, live2, live)
    # extract dead cell neighbors
    np.equal(cells, 0, dead)
    dead *= neighbors
    np.equal(dead,ALIVE,dead)
    # make sure all threads have read their values
    np.bitwise_or(live, dead, cells)
    # ABGR is the order...
    if DISPLAY:
        np.save("%s.%08d"%(sys.argv[2],i), full)
t2 = time.time()
print "Time spent:", t2-t1
