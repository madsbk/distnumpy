import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import os

TMPDIR = "/tmp/distnumpy_viewer/"
try:
    os.makedirs(TMPDIR)
except OSError:
    pass

for i in xrange(10000):
    try:
        x = np.load("%s.%08d.npy"%(sys.argv[1],i))
        print "Processing iteration: %d\r"%i,
        sys.stdout.flush()
        plt.pcolor(x)
        plt.colorbar()
        plt.savefig("%s%s.%08d.png"%(TMPDIR, sys.argv[1],i), dpi=100)    
        plt.clf()
    except IOError:
        continue
    except KeyboardInterrupt:
        break;


command = ('mencoder',
           'mf://%s.*.png'%sys.argv[1],
           '-mf',
           'type=png:w=800:h=600:fps=10',
           '-ovc',
           'lavc',
           '-lavcopts',
           'vcodec=mpeg4',
           '-oac',
           'copy',
           '-o',
           'tmp.avi')

print "\n\nabout to execute:\n%s\n\n" % ' '.join(command)
subprocess.check_call(command, cwd=TMPDIR)
subprocess.check_call(('mv %s/tmp.avi %s.avi'%(TMPDIR, sys.argv[1])), shell=True)
subprocess.check_call(('rm -R %s'%TMPDIR), shell=True)

print "\n\n The movie was written to '%s.avi'"%sys.argv[1]
