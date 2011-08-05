#Test Blocked LU factorization
import numpy as np
import dnumpytest

def gen_matrix(SIZE,dist):
    BS = np.BLOCKSIZE
    Nblock = SIZE / BS
    Nlocal = BS

    # generate matrix to factorize
    A = np.zeros((SIZE,SIZE), dtype=float, dist=dist);
    for row in xrange(0,SIZE,BS):
        rbs = min(BS,SIZE - row)
        for col in xrange(0,SIZE,BS):
            cbs = min(BS,SIZE - col)
            for n in xrange(rbs):
                for m in xrange(cbs):
                    r = row + n
                    c = col + m
                    t1 = float(SIZE) / BS
                    A[row+n,col+m] = r*c / (t1*BS*BS*t1)
            if row == col:
                for n in xrange(rbs):
                    A[row+n,col+n] = A[row+n,col+n] + 10
    return A


def run():
    if not np.SPMD_MODE:
        print "[rank %d] Warning - ignored in non-SPMD mode\n"%(np.RANK),
        return
    try:#This test requires the scipy module
        from scipy import linalg
    except:
        print "[rank %d] Warning - ignored scipy not found\n"%(np.RANK),
        return
    try:#This test requires the pyHPC module
        import pyHPC
    except:
        print "[rank %d] Warning - ignored pyHPC not found\n"%(np.RANK),
        return

    for SIZE in xrange(np.BLOCKSIZE,100,np.BLOCKSIZE):
        (Ld, Ud) = pyHPC.lu(gen_matrix(SIZE,True))
        (P, Lf, Uf) = linalg.lu(gen_matrix(SIZE,False))

        #There seems to be a transpose bug in SciPy's LU
        Lf = Lf.T
        Uf = Uf.T

        if not (np.diag(P) == 1).all():#We do not support pivoting
            raise Exception("Pivoting was needed!")

        if not dnumpytest.array_equal(Ld,Lf,maxerror=1e-13):
            raise Exception("Uncorrect L matrix\n")

        if not dnumpytest.array_equal(Ud,Uf,maxerror=1e-13):
            raise Exception("Uncorrect U matrix\n")



if __name__ == "__main__":
    run()
