import numpy as np
import sys
import time

d = int(sys.argv[1]) #Distributed
n = int(sys.argv[2]) #Number of bodies
k = int(sys.argv[3]) #Number of iterations

G = 1     #Gravitational constant
dT = 0.01 #Time increment

M  = np.empty((n,1), dist=d)
np.ufunc_random(M,M)
MT = np.empty((1,n), dist=d)
np.ufunc_random(MT,MT)
Px = np.empty((n,1), dist=d)
np.ufunc_random(Px,Px)
Py = np.empty((n,1), dist=d)
np.ufunc_random(Py,Py)
Pz = np.empty((n,1), dist=d)
np.ufunc_random(Pz,Pz)
PxT= np.empty((1,n), dist=d)
np.ufunc_random(PxT,PxT)
PyT= np.empty((1,n), dist=d)
np.ufunc_random(PyT,PyT)
PzT= np.empty((1,n), dist=d)
np.ufunc_random(PzT,PzT)
Vx = np.empty((n,1), dist=d)
np.ufunc_random(Vx,Vx)
Vy = np.empty((n,1), dist=d)
np.ufunc_random(Vy,Vy)
Vz = np.empty((n,1), dist=d)
np.ufunc_random(Vz,Vz)

OnesCol = np.zeros((n,1), dtype=float, dist=d)+1.0
OnesRow = np.zeros((1,n), dtype=float, dist=d)+1.0
#Identity= array(diag([1]*n), dtype=double, dist=d)

np.core.multiarray.evalflush()
stime = time.time()
for i in xrange(k):
    #distance between all pairs of objects
    Fx = np.dot(OnesCol, PxT) - np.dot(Px, OnesRow)
    Fy = np.dot(OnesCol, PyT) - np.dot(Py, OnesRow)
    Fz = np.dot(OnesCol, PzT) - np.dot(Pz, OnesRow)

    Dsq = Fx * Fx + Fy * Fy + Fx * Fz #+ Identity
    D = np.sqrt(Dsq)

    #mutual forces between all pairs of objects
    F = G * np.dot(M, MT) / Dsq

    #F = F - diag(diag(F))#set 'self attraction' to 0
    Fx = (Fx / D) * F
    Fy = (Fy / D) * F
    Fz = (Fz / D) * F

    #net force on each body
    Fnet_x = np.add.reduce(Fx,1)
    Fnet_y = np.add.reduce(Fy,1)
    Fnet_z = np.add.reduce(Fz,1)

    Fnet_x = Fnet_x[:,np.newaxis]
    Fnet_y = Fnet_y[:,np.newaxis]
    Fnet_z = Fnet_z[:,np.newaxis]

    Fnet_x *= dT
    Fnet_y *= dT
    Fnet_z *= dT

    #change in velocity:
    Vx += Fnet_x / M
    Vy += Fnet_y / M
    Vz += Fnet_z / M

    #change in position
    Px += Vx * dT
    Py += Vy * dT
    Pz += Vz * dT

np.core.multiarray.evalflush()
print 'nbody with #bodies: ', n,', iter: ', i+1, 'in sec: ', time.time() - stime,
if d:
    print " (Dist) notes: %s"%sys.argv[4]
else:
    print " (Non-Dist) notes: %s"%sys.argv[4]




"""Paper version:
    #distance between all pairs of objects
    Fx = dot(OnesCol, PxT) - dot(Px, OnesRow)
    Fy = dot(OnesCol, PyT) - dot(Py, OnesRow)
    Fz = dot(OnesCol, PzT) - dot(Pz, OnesRow)

    Dsq = Fx * Fx + Fy * Fy + Fx * Fz #+ Identity
    D = sqrt(Dsq)

    #mutual forces between all pairs of objects
    F = G * dot(M, MT) / Dsq

    #F = F - diag(diag(F))#set 'self attraction' to 0
    Fx = (Fx / D) * F
    Fy = (Fy / D) * F
    Fz = (Fz / D) * F

    #net force on each body
    Fnet_x = add.reduce(Fx,1)
    Fnet_x = add.reduce(Fy,1)
    Fnet_x = add.reduce(Fz,1)

    #change in velocity:
    Vx += Fnet_x[:,newaxis] * dT / M
    Vy += Fnet_y[:,newaxis] * dT / M
    Vz += Fnet_z[:,newaxis] * dT / M

    #change in position
    Px += Vx * dT
    Py += Vy * dT
    Pz += Vz * dT
"""
