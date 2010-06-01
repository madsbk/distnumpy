from numpy import *
import dnumpytest

def jacobi(A, B, tol=0.005):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    h = zeros(shape(B), float, dist=A.dist())
    dmax = 1.0
    n = 0
    tmp0 = empty(shape(A), float, dist=A.dist())
    tmp1 = empty(shape(B), float, dist=A.dist())
    AD = diagonal(A)
    while dmax > tol:
        n += 1
        multiply(A,h,tmp0)
        add.reduce(tmp0,1,out=tmp1)
        tmp2 = AD
        subtract(B, tmp1, tmp1)
        divide(tmp1, tmp2, tmp1)
        hnew = h + tmp1
        subtract(hnew,h,tmp2)
        divide(tmp2,h,tmp1)
        absolute(tmp1,tmp1)
        dmax = maximum.reduce(tmp1)
        h = hnew
    return h

def run():
    Ad = array([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 4, -1], [0, -1, -1, 4]], float, dist=True)
    Bd = array([1,2,0,1], float, dist=True)
    Af = array([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 4, -1], [0, -1, -1, 4]], float, dist=False)
    Bf = array([1,2,0,1], float, dist=False)
        
    Seq = jacobi(Af,Bf)
    Par = jacobi(Ad,Bd)

    if not dnumpytest.array_equal(Seq,Par):
        return (True, "Uncorrect result matrix\n")
    return (False, "")

if __name__ == "__main__":
    print run()
