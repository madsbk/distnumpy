import time
import numpy as np
import pyHPC
import util

np.datalayout([(3,1,1)])

parser = util.Parsing()
DIST = parser.dist
SIZE = int(parser.argv[0])
A = np.empty((SIZE,SIZE,SIZE), dtype=np.complex, dist=DIST)

if DIST:
    f = pyHPC.fft3d
else:
    f = np.fft.fftn

np.timer_reset()
f(A)
timing = np.timer_getdict()

if np.RANK == 0:
    print 'fft3d - size:,',np.shape(A)

parser.pprint(timing)
parser.write_dict(timing)

