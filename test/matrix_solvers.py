from numpy import *
import time

def jacobi(A, B, tol=0.005, init_val=0.0):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    dist = False
    if(A.dist()):
        dist = True
    
    print "jacobi solver - dist:", dist
    h = zeros(shape(B), float, dist=dist)
    dmax = 1.0
    n = 0

    t1 = t = time.time()
    while dmax > tol:
        n += 1
        #print "  Iteration = ", n
        tmp1 = add.reduce(multiply(A,h),1)
        tmp2 = diag(A)
        tmp3 = subtract(B, tmp1)
        print "TMP2: ", tmp2, " dist:",tmp2.dist()
        print "TMP3: ", tmp3, " dist:",tmp3.dist()
        tmp4 = divide(tmp3, tmp2)
        hnew = h + tmp4
##        print_arr(multiply(A,h), "multiply(A,h)")
##        print_arr(add.reduce(multiply(A,h),1), "add.reduce(multiply(A,h))")
##        print_arr(subtract(B, add.reduce(multiply(A,h),1)), "subtract(B, add.reduce(multiply(A,h)))")
##        print_arr(hnew, "hnew")
        dmax = max(abs(divide(subtract(h,hnew), h)))
        #print "   dmax = ", dmax, ": tol = ", tol
        #print "subtract", subtract(h,hnew)
        if (time.time() - t) > 2.0:
            print "Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
            t = time.time()
        h = hnew
##        print n, dmax,  h
##        print " "
    #print_arr(h, "h")
    print "SOLVED at: Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
    print "                           Time = ", time.time() - t1, "seconds"

    return h


def gauss_seidel(A, B, tol=0.005, init_val=0.0):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    dist = False
    if(A.dist()):
        dist = True
    
    print "gauss_siedel - dist:", dist
    if init_val == 0.0:
        h = zeros(shape(B), float, dist=dist)
        hnew = zeros(shape(B), float, dist=dist)
    else:
        h = init_val
        hnew[:] = h[:]
    dmax = 1.0
    n = 0

    t1 = t = time.time()
    while dmax > tol:
        n += 1
        #print "  Iteration = ", n
        for i in range(len(B)):
            #print "add.reduce(multiply(A,h)) ", add.reduce(multiply(A,h))
            hnew[i] = h[i] + (B[i] - add.reduce(multiply(A[i,:],hnew)))/ A[i,i]
            #divide(add.reduce(subtract(B, multiply(hnew, A[i,:]))), A[i,i])
            #( B - h * A[i,:]) / A[i,i]
        dmax = max(abs(divide(subtract(h,hnew), h)))
        #print "Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
##        print_arr(h, "h")
##        print_arr(hnew, "hnew")
        if (time.time() - t) > 2.0:
            print "Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
            t = time.time()
        h[:] = hnew[:]
##        print n, ":", dmax,  h   
    print "SOLVED at: Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
    print "                           Time = ", time.time() - t1, "seconds"
    
    return h

def sor(A, B, rf, tol=0.005, forcedIter=0):
    '''itteratively solving for matrix A with solution vector B
       rf = relaxation factor (rf > 1.0; eg rf = 1.1)
       tol = tolerance for dh/h       
    '''
    
    print "successive over-relaxation (sor) - dist:", A.dist(), ""
    h = zeros(shape(B), float, dist=A.dist())
    hnew = zeros(shape(B), float, dist=A.dist())
    dmax = 1.0
    n = 0

    t = time.time()
    while (forcedIter and forcedIter > n) or \
          (forcedIter == 0 and dmax > tol):
        n += 1

        for i in range(len(B)):
            t1 = multiply(A[i],hnew)

            #t2 = add.reduce(t1)
            #hnew[i] = h[i] + (B[i] - t2) / A[i,i]
        
        #over-relaxation
        #dh = subtract(hnew, h)
        #multiply(rf,dh,dh)
        #add(h,dh,hnew)
        #subtract(h,hnew,dh)
        #divide(dh,h,dh)
        #absolute(dh,dh)
        #dmax = maximum.reduce(dh)
        #h[:] = hnew[:]
        
    print "SOLVED at: Itteration = ", n, ":   dmax = ", dmax, ": tol = ", tol
    print "                           Time = ", time.time() - t, "seconds"
    
    return h


def print_arr(arr, title):
    print title, shape(arr)
    print arr

def random_list(dims):
    if len(dims) == 0:
        return random.randint(0,100000)
    
    list = []
    for i in range(dims[-1]):
        list.append(random_list(dims[0:-1]))
    return list  

d = True
size = 100
iter = 10

#A = array([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 4, -1], [0, -1, -1, 4]], float, dist=d)

#A = array(random_list([size,size]), float, dist=d)
A = zeros([size,size], dtype=float, dist=d)
add(A,42,A)

#B = array(random_list([size]), float, dist=d)
B = zeros([size], dtype=float, dist=d)
add(B,42,B)


"""
C = jacobi(A,B)
print_arr(C, "C jacobi")
C = gauss_seidel(A,B)
print_arr(C, "C gauss_seidel")
"""


C = sor(A, B, rf=1.3,forcedIter=iter)


