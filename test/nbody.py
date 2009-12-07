from numpy import *
import sys
import time

d = int(sys.argv[1]) #Distributed
n = int(sys.argv[2]) #Number of bodies
k = int(sys.argv[3]) #Number of iterations

G = 1     #Gravitational constant
dT = 0.01 #Time increment

M  = empty((n,1), dist=d)
ufunc_random(M,M)
MT = empty((1,n), dist=d)
ufunc_random(MT,MT)
Px = empty((n,1), dist=d)
ufunc_random(Px,Px)
Py = empty((n,1), dist=d)
ufunc_random(Py,Py)
Pz = empty((n,1), dist=d)
ufunc_random(Pz,Pz)
PxT= empty((1,n), dist=d)
ufunc_random(PxT,PxT)
PyT= empty((1,n), dist=d)
ufunc_random(PyT,PyT)
PzT= empty((1,n), dist=d)
ufunc_random(PzT,PzT)
Vx = empty((n,1), dist=d)
ufunc_random(Vx,Vx)
Vy = empty((n,1), dist=d)
ufunc_random(Vy,Vy)
Vz = empty((n,1), dist=d)
ufunc_random(Vz,Vz)

OnesCol = zeros((n,1), dtype=double, dist=d)+1.0
OnesRow = zeros((1,n), dtype=double, dist=d)+1.0
#Identity= array(diag([1]*n), dtype=double, dist=d)

stime = time.time()
for i in range(k):
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
    Fnet_x = dot(Fx, OnesCol)
    Fnet_y = dot(Fy, OnesCol)
    Fnet_z = dot(Fz, OnesCol)

    #change in velocity:
    Vx += Fnet_x * dT / M
    Vy += Fnet_y * dT / M
    Vz += Fnet_z * dT / M

    #change in position
    Px += Vx * dT
    Py += Vy * dT
    Pz += Vz * dT

print 'nbody with #bodies: ', n,', iter: ', i, 'in sec: ', time.time() - stime,
if d:
    print " (Dist) notes: %s"%sys.argv[4]
else:
    print " (Non-Dist) notes: %s"%sys.argv[4]
