import numpy as np
import util

def stencil(iters, work, center, up, left, right, down):
    for i in xrange(iters):
        work[:] = center
        work += up
        work += left
        work += right
        work += down
        work *= 0.2

parser = util.Parsing()
DIST=parser.dist
nonaligned = bool(eval(parser.argv[0]))
N = int(parser.argv[1])
iters=int(parser.argv[2])
BS=1

full = np.empty((N+BS*2,N+BS*2), dtype=np.double, dist=DIST)
align = np.empty((N,N), dtype=np.double, dist=DIST)
work   = np.zeros((N,N), dtype=np.double, dist=DIST)
np.ufunc_random(full,full)
np.ufunc_random(align,align)

center = full[1*BS:-1*BS, 1*BS:-1*BS]
up     = full[1*BS:-1*BS,     :-2*BS]
left   = full[:-2*BS    , 1*BS:-1*BS]
right  = full[2*BS:     , 1*BS:-1*BS]
down   = full[1*BS:-1*BS, 2*BS:     ]

np.timer_reset()
if nonaligned:
    stencil(iters, work, center, up, left, right, down)
else:
    stencil(iters, work, align, align, align, align, align)
timing = np.timer_getdict()

print 'alignment - nonaligned: %s, iters: %d'%(nonaligned, iters),\
      ' size:,',np.shape(work)
parser.pprint(timing)
parser.write_dict(timing)
