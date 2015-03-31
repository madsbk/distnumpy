DistNumPy is a new version of NumPy that parallelizes array operations in a manner completely transparent to the user - from the perspective of the user, the difference between NumPy and DistNumPy is minimal. DistNumPy can use multiple processors through the communication library Message Passing Interface (MPI). In DistNumPy MPI communication is fully transparent and the user needs no knowledge of MPI or any parallel programming model. However, the user is required to use the array operations in DistNumPy to obtain any kind of speedup.

The only difference in the API of NumPy and DistNumPy is the array creation routines. DistNumPy allow both distributed and non-distributed arrays to co-exist thus the user must specify, as an optional parameter, if the array should be distributed. The following illustrates the only difference between the creation of a standard array and a distributed array:
```
#Non-Distributed
A = numpy.array([1,2,3])
#Distributed
B = numpy.array([1,2,3], dist=True)
```

Please note that the current implementation of DistNumPy is in alpha state - only a fraction of NumPy interface is implemented.

This project is continued through the project **Bohrium**: http://www.bh107.org