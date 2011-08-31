#N-dimensional transpose
import numpy as np
import pyHPC

def fft2d(A):
    """
    Compute the 2-dimensional discrete Fourier Transform

    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array, can be complex

    Returns
    -------
    out : complex ndarray
    """

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
        l_A[:] = np.fft.fft(l_A,axis=localaxis) ## local FFTs

    ## Transpose
    A = pyHPC.transpose(A)

    ## 1-D FFt on the Y dimension
    l_A = A.local()
    if len(l_A) > 0:
        l_A[:] = np.fft.fft(l_A,axis=localaxis) ## local FFTs

    ## Transpose
    A = pyHPC.transpose(A)

    return A


def fft3d(A):
    """
    Compute the 3-dimensional discrete Fourier Transform

    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array, can be complex

    Returns
    -------
    out : complex ndarray
    """

    if not A.dist():
        raise Exception("The array must be distributed")

    for d in A.shape:
        if(d % np.BLOCKSIZE != 0):
            raise Exception("The array dimensions must be divisible "\
                            "with np.BLOCKSIZE(%d)"%np.BLOCKSIZE)

    #Find an axis that is not distributed.
    localaxis=0
    if A.pgrid()[0] != 1:
        raise Exception("The first dimension in the process grid must not "
                        "be distributed.")

    #Convert to a complex array
    B = np.empty(A.shape, dtype=np.complex, dist=True)
    B[:] = A

    ## 1-D FFT on data along the X dimension, for which the data
    ## should be local and contiguous
    l_A = A.local()
    if len(l_A) > 0:
        l_A[:] = np.fft.fft(l_A,axis=localaxis) ## local FFTs

    ## Transpose the X,Y planes: X'=Y, Y'=X, Z'=Z, (Y,X,Z)
    A = pyHPC.transpose(A,(1,0,2))

    ## 1-D FFT on data along the Y dimension (now is X', for which the
    ## data should be local and contiguous)
    l_A = A.local()
    if len(l_A) > 0:
        l_A[:] = np.fft.fft(l_A,axis=localaxis) ## local FFTs

    ## Transpose the X',Z' planes: X''=Z'=Z, Y''=Y'=X, Z''=X'=Y, (Z,X,Y)
    A = pyHPC.transpose(A,(2,1,0))

    ## 1-D FFt on data along the Z dimension (now is X'', for which
    ## data should be local and contiguous)
    l_A = A.local()
    if len(l_A) > 0:
        l_A[:] = np.fft.fft(l_A,axis=localaxis) ## local FFTs

    ## Transpose the X',Z' planes: X''=Z'=Z, Y''=Y'=X, Z''=X'=Y, (Z,X,Y)
    A = pyHPC.transpose(A,(1,2,0))

    return A
