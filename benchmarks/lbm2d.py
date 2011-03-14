
'''
Channel flow past a cylindrical obstacle, using a LB method
Copyright (C) 2006 Jonas Latt
Address: Rue General Dufour 24,  1211 Geneva 4, Switzerland
E-mail: Jonas.Latt@cui.unige.ch
'''
import numpy as np
import time
import sys

"""
# Initialise Tk ...
import matplotlib
import matplotlib.figure
import matplotlib.backends.backend_tkagg
import Tkinter
figure = matplotlib.figure.Figure()
figure.set_size_inches((8, 6))
axes1 = figure.add_subplot(311)
axes2 = figure.add_subplot(312)
axes3 = figure.add_subplot(313)
tk = Tkinter.Tk()
canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(figure, master = tk)
canvas.get_tk_widget().pack(expand = True, fill = Tkinter.BOTH)
"""

DIST=int(sys.argv[1])

# General flow constants
lx = int(sys.argv[2]) #250
ly = int(sys.argv[3]) #51
obst_x = lx/5.+1               # position of the cylinder; (exact
obst_y = ly/2.+1               # y-symmetry is avoided)
obst_r = 10 #ly/10.+1              # radius of the cylinder
uMax   = 0.02                  # maximum velocity of Poiseuille inflow
Re     = 100                   # Reynolds number
nu     = uMax * 2.*obst_r / Re # kinematic viscosity
omega  = 1. / (3*nu+1./2.)     # relaxation parameter
maxT   = int(sys.argv[4])#10   # total number of iterations
tPlot  = 5                     # cycles

# D2Q9 Lattice constants
t  = np.array([4/9., 1/9.,1/9.,1/9.,1/9., 1/36.,1/36.,1/36.,1/36.], dtype=float, dist=DIST)
cx = np.array([  0,   1,  0, -1,  0,    1,  -1,  -1,   1], dtype=float, dist=DIST)
cy = np.array([  0,   0,  1,  0, -1,    1,   1,  -1,  -1], dtype=float, dist=DIST)
opp = np.array([ 0,   3,  4,  1,  2,    7,   8,   5,   6], dtype=float, dist=DIST)
col = np.array(xrange(2,ly), dtype=float, dist=DIST)

bbRegion = np.zeros((lx,ly), dtype=float, dist=DIST)
not_bbRegion = np.zeros((lx,ly), dtype=float, dist=DIST)
not_bbRegion += 1.0
for x in xrange(lx):
    for y in xrange(ly):
        if (x-obst_x)**2 + (y-obst_y)**2 <= obst_r**2:
            bbRegion[x,y] += 1.0
            not_bbRegion[x,y] *= 0.0
bbRegion[:,0] += 1.0
bbRegion[:,-1] += 1.0
not_bbRegion[:,0] *= 0.0
not_bbRegion[:,-1] *= 0.0

# Initial condition: (rho=0, u=0) ==> fIn[i] = t[i]
fIn = (np.zeros([9,lx,ly], dtype=float, dist=DIST)+1) * t[:,np.newaxis,np.newaxis]
fEq = np.zeros([9,lx,ly], dtype=float, dist=DIST)+1
fOut = np.zeros([9,lx,ly], dtype=float, dist=DIST)+1

def lbm2d():
    # Main loop (time cycles)
    for cycle in xrange(maxT):
        # Macroscopic variables
        rho = fIn.sum(axis = 0)
        ux = np.add.reduce(cx[:, np.newaxis, np.newaxis] * fIn) / rho
        uy = np.add.reduce(cy[:, np.newaxis, np.newaxis] * fIn) / rho

        # Macroscopic (Dirichlet) boundary conditions
        # Inlet: Poisseuille profile
        L = ly-2.0
        y = col-0.5
        ux[0,2:] = 4 * uMax / (L ** 2) * (y * L - y ** 2)
        uy[0,2:] *= 0
        t1 = np.empty(fIn[0:5:2,0,2:].shape, dtype=float, dist=DIST)
        t2 = np.empty(fIn[3:8:2,0,2:].shape, dtype=float, dist=DIST)
        t1[:] = fIn[0:5:2,0,2:]
        t2[:] = fIn[3:8:2,0,2:]
        rho[0,2:] = 1 / (1-ux[0,2:]) * (np.add.reduce(t1) + 2 * np.add.reduce(t2))
        # Outlet: Zero gradient on rho/ux
        rho[-1,2:] = rho[-2,2:]
        uy[-1,2:] *= 0
        ux[-1,2:] = ux[-2, 2:]

        for i in xrange(0, 9):
            cu = 3 * (cx[i] * ux + cy[i] * uy)
            fEq[i] = rho * t[i] * (1 + cu + 0.5 * cu ** 2 - 1.5 * (ux ** 2 + uy ** 2))
            fOut[i] = fIn[i] - omega * (fIn[i] - fEq[i])

        # Microscopic boundary conditions
        for i in xrange(0, 9):
            # Left boundary:
            fOut[i, 0, 2:] = fEq[i,0,2:] + 18 * t[i] * cx[i] * cy[i] * (fIn[7,0,2:] - fIn[6,0,2:] - fEq[7,0,2:] + fEq[6,0,2:])
            # Right boundary:
            fOut[i,-1,2:] = fEq[i,-1,2:] + 18 * t[i] * cx[i] * cy[i] *(fIn[5,-1,2:] - fIn[8,-1,2:] - fEq[5,-1,2:] + fEq[8,-1,2:])
            # Bounce back region:
            BB = np.zeros((lx,ly), dist=DIST)
            BB[:] = fIn[opp[i]]
            BB *= bbRegion
            fOut[i] *= not_bbRegion
            fOut[i] += BB

        # Streaming step
        for i in xrange(0,9):
            if cx[i] == 1:
                t1 = np.empty(fOut[i].shape, dtype=float, dist=DIST)
                t1[1:] = fOut[i][:-1]
                t1[0] = fOut[i][-1]
                fOut[i] = t1
            elif cx[i] == -1:
                t1 = np.empty(fOut[i].shape, dtype=float, dist=DIST)
                t1[:-1] = fOut[i][1:]
                t1[-1] = fOut[i][0]
                fOut[i] = t1
            if cy[i] == 1:
                t1 = np.empty(fOut[i].shape, dtype=float, dist=DIST)
                t1[:,1:] = fOut[i][:,:-1]
                t1[:,0] = fOut[i][:,-1]
                fIn[i] = t1
            elif cy[i] == -1:
                t1 = np.empty(fOut[i].shape, dtype=float, dist=DIST)
                t1[:,:-1] = fOut[i][:,1:]
                t1[:,-1] = fOut[i][:,0]
                fIn[i] = t1
            else:
                fIn[i] = fOut[i]
        """
        if not cycle%tPlot:
            u = np.sqrt(ux**2+uy**2)
            #u[bbRegion] = np.nan
            axes1.clear()
            axes2.clear()
            axes3.clear()
            axes1.imshow(u.T)
            axes2.imshow(ux.T)
            axes3.imshow(uy.T)
            canvas.show()
        """

np.core.multiarray.timer_reset()
np.core.multiarray.evalflush()
t1 = time.time()
lbm2d()
np.core.multiarray.evalflush()
t2 = time.time()
print 'Iter: %d size: (%d, %d) time: '%(maxT,lx,ly), t2-t1,
if DIST:
    print "(Dist) notes: %s"%sys.argv[5]
else:
    print "(Non-Dist) notes: %s"%sys.argv[5]
