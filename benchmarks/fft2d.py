import time
import numpy as np
import pyHPC
import util

np.datalayout([(2,1,1)])

parser = util.Parsing()
DIST = parser.dist
SIZE1 = int(parser.argv[0])
SIZE2 = int(parser.argv[1])
A = np.empty((SIZE1,SIZE2), dtype=np.complex, dist=DIST)

if DIST:
    f = pyHPC.fft2d
else:
    f = np.fft.fft2

np.timer_reset()
f(A)
timing = np.timer_getdict()

if np.RANK == 0:
    print 'fft2d - size:,',np.shape(A)

parser.pprint(timing)
parser.write_dict(timing)

