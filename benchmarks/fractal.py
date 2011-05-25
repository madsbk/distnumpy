import numpy as np
import time
import sys

DEBUG = False
DIST = int(sys.argv[1])

DENSITY = int(sys.argv[2]) # warning: execution speed decreases with square of DENSITY
ITERATIONS = int(sys.argv[3])

x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

c = np.zeros((DENSITY,DENSITY), dtype=np.complex, dist=DIST)
z = np.zeros((DENSITY,DENSITY), dtype=np.complex, dist=DIST)

#Original: c = lx + 1j*ly
if DEBUG:
    lx, ly = np.meshgrid(np.linspace(x_min, x_max, DENSITY),
                         np.linspace(y_min, y_max, DENSITY))
    for x in xrange(len(lx)):
        for y in xrange(len(ly)):
            c[x,y] += lx[x,y] + 1j*ly[x,y]
    del lx
    del ly
else:
    c += 42 + 1j*7
    z += 42 + 1j*7

z[:] = c
fractal = np.zeros(z.shape, dtype=np.uint8, dist=DIST) + 255

np.timer_reset()
np.evalflush()
t1 = time.time()

for n in range(ITERATIONS):
    z *= z
    z += c
    mask = (fractal == 255) & (abs(z) > 10)
    fractal *= ~mask
    fractal += mask * 254 * n / float(ITERATIONS)

np.evalflush()
t2 = time.time()

if DEBUG and not DIST:
    import matplotlib.pyplot as plt
    plt.imshow(np.log(fractal), cmap=plt.cm.hot,
               extent=(x_min, x_max, y_min, y_max))
    plt.title('Mandelbrot Set')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

print 'DENSITY: ', DENSITY, ' ITERATIONS ',ITERATIONS,' in sec: ', t2-t1,
if DIST:
    print "(Dist) notes: %s"%sys.argv[4]
else:
    print "(Non-Dist) notes: %s"%sys.argv[4]
