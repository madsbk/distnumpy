#SUMMA: matrix multiplication algorithm
import numpy as np

def summa(a,b,out=None):
    """
    Matrix multiplication using the SUMMA algorithm.

    """
    ksize = a.shape[1] #Number of rows and columns in the match dimension

    if out is None:
        c = np.zeros((a.shape[0],b.shape[1]), dtype=a.dtype, dist=True)
    else:
        c = out

    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise Exception("All arrays must have two dimensions")

    if a.shape[1] != b.shape[0]:
        raise Exception("Shape of a(%d,%d) and b(%d,%d) does not match"\
                         %(a.shape[0],a.shape[1],b.shape[0],b.shape[1]))

    if c.shape[0] != a.shape[0] or c.shape[1] != b.shape[1]:
        raise Exception("Shape of output (%d,%d) do not match input "\
                        "a(%d,%d) and b(%d,%d)"\
                        %(c.shape[0],c.shape[1],a.shape[0],\
                        a.shape[1],b.shape[0],b.shape[1]))

    if not (a.dist() and b.dist() and c.dist()):
        raise Exception("All arrays must be distributed")

    (prow,pcol) = a.pgrid()
    BS = np.BLOCKSIZE
    a_work = np.zeros((a.shape[0],BS*pcol), dtype=a.dtype, dist=True)
    b_work = np.zeros((BS*prow,b.shape[1]), dtype=a.dtype, dist=True)

    for k in xrange(0,ksize,BS):
        bs = min(BS, ksize - k)#Current block size

        #Replicate colum-block horizontal
        for p in xrange(pcol):
            a_work[:,p*BS:p*BS+bs] = a[:,k:k+bs]
        #Replicate row-block vertical
        for p in xrange(prow):
            b_work[p*BS:p*BS+bs,:] = b[k:k+bs,:]

        #Apply local outer dot product
        l_a_work = a_work.local()[:,:bs]
        l_b_work = b_work.local()[:bs,:]
        l_c = c.local()
        l_c += np.dot(l_a_work, l_b_work)

    return c
