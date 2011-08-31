#Test FFT
import numpy as np
import dnumpytest


def run():
    #Make sure we have one non-distributed dimension.
    np.datalayout([(2,1,1),(3,1,1)])
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
    if np.BLOCKSIZE > 10:
        print "[rank %d] Warning - ignored np.BLOCKSIZE too high\n"%(np.RANK),
        return

    #2D FFT
    for SIZE in xrange(np.BLOCKSIZE,np.BLOCKSIZE*10,np.BLOCKSIZE):
        src = dnumpytest.random_list([SIZE,SIZE])
        Ad = np.array(src, dtype=np.complex, dist=True)
        Af = np.array(src, dtype=np.complex, dist=False)
        Bd = pyHPC.fft2d(Ad)
        Bf = np.fft.fft2(Af)

        if not dnumpytest.array_equal(Bf,Bd,maxerror=1e-6):
            raise Exception("Uncorrect result array\n")

    #3D FFT
    for SIZE in xrange(np.BLOCKSIZE,np.BLOCKSIZE*5,np.BLOCKSIZE):
        src = dnumpytest.random_list([SIZE,SIZE,SIZE])
        Ad = np.array(src, dtype=np.complex, dist=True)
        Af = np.array(src, dtype=np.complex, dist=False)
        Bd = pyHPC.fft3d(Ad)
        Bf = np.fft.fftn(Af)

        if not dnumpytest.array_equal(Bd,Bf,maxerror=1e-6):
            raise Exception("Uncorrect result array\n")

if __name__ == "__main__":
    run()
