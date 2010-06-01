import numpy as np
import dnumpytest

def sor(W,H,Dist):
    full = np.zeros((W+2,H+2), dtype=np.double, dist=Dist)
    work = np.zeros((W,H), dtype=np.double, dist=Dist)
    diff = np.zeros((W,H), dtype=np.double, dist=Dist)

    cells = full[1:W+1,1:H+1]
    up = full[1:W+1, 0:H]
    left = full[0:W, 1:H+1]
    right = full[2:W+2, 1:H+1]
    down = full[1:W+1, 2:H+2]

    for i in range(1,W+1):
      full[i][0]=-273.15
      full[i][-1]=-273.15

    full[0,:] += 40.0
    full[W+1,:] += -273.13

    epsilon=W*H*0.002
    delta=epsilon+1
    i=0
    while epsilon<delta:
      i+=1
      np.add(cells,0,work)
      work += up
      work += left
      work += right
      work += down
      work *= 0.2                            
      np.subtract(cells,work,diff)
      diff=np.absolute(diff)
      delta=np.sum(diff)
      np.add(work,0,cells)
    return cells

def run():
    Seq = sor(5,5,False)
    Par = sor(5,5,True)

    if not dnumpytest.array_equal(Seq,Par):
        return (True, "Uncorrect result matrix\n")
    return (False, "")

if __name__ == "__main__":
    print run()
