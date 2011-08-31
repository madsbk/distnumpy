#N-dimensional transpose
import numpy as np
import pyHPC

def fft2d(A):

    if not A.dist():
        raise Exception("The array must be distributed")

    for d in A.shape:
        if(d % np.BLOCKSIZE != 0):
            raise Exception("The array dimensions must be divisible "\
                            "with np.BLOCKSIZE(%d)"%np.BLOCKSIZE)

    #Find an axis that is not distributed.
    localaxis = -1
    for i,p in enumerate(A.pgrid()):
        if p == 1:
            localaxis = i
            break
    if localaxis == -1:
        raise Exception("One dimension in the process grid must not "
                        "be distributed.")

    #Convert to a complex array
    B = np.empty(A.shape, dtype=np.complex, dist=True)
    B[:] = A

    ## 1-D FFT on the X dimension
    l_A = A.local()

    if len(l_A) > 0:
        l_A[:] = np.fft.fft(l_A,axis=localaxis) ## should be local FFTs

    ## Transpose
    A = pyHPC.transpose(A)

    ## 1-D FFt on the Y dimension
    l_A = A.local()
    if len(l_A) > 0:
        l_A[:] = np.fft.fft(l_A,axis=localaxis) ## should be local FFTs

    ## Transpose
    A = pyHPC.transpose(A)

    return A
